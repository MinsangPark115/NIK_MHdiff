# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import click
from tqdm.auto import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import tensorflow as tf
import io
from torchvision.utils import make_grid, save_image
import classifier_lib
import random
import time

#----------------------------------------------------------------------------
# Proposed EDM-G++ sampler.

def edm_sampler(
    boosting, time_min, time_max, vpsde, dg_weight_1st_order, dg_weight_2nd_order, discriminator,
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    backsteps=0, min_backsteps=0, max_backsteps=18, mode='default', outdir=None, adaptive_pickle=None, adaptive_pickle2=None,
    class_idx=None, batch_size=100, num_samples=50000, iter_warmup=10, max_iter=999999, do_seed=1, snr=0, litrs=0,
):

    def sampling_loop(x_next, lst_idx, log_ratio_prev, per_sample_nfe, labels, warmup=False):
        t_cur = t_steps[lst_idx]
        t_next = t_steps[lst_idx+1]

        x_cur = x_next
        t_hat = t_cur
        x_hat = x_cur

        bool_zero = lst_idx == 0
        if warmup:
            if bool_zero.sum() != 0:
                log_ratio_prev[bool_zero] = classifier_lib.get_grad_log_ratio(discriminator, vpsde, x_cur[bool_zero], t_steps[lst_idx][bool_zero], net.img_resolution, time_min, time_max, labels, log_only=True).detach().cpu()

                for i in range(len(log_ratio_prev[bool_zero])):
                    lst_adaptive[0].append(log_ratio_prev[bool_zero][i].cpu())
        else:
            if min_backsteps == 0:
                while bool_zero.sum() != 0:
                    x_check = x_cur[bool_zero]
                    labels_ = labels[bool_zero] if labels is not None else None
                    log_ratio_prev_check = log_ratio_prev[bool_zero]
                    log_ratio = classifier_lib.get_grad_log_ratio(discriminator, vpsde, x_check, t_steps[lst_idx][bool_zero], net.img_resolution, time_min, time_max, labels_, log_only=True).detach().cpu()
                    bool_neg_log_ratio = log_ratio < adaptive[lst_idx][bool_zero] + torch.log(torch.rand_like(log_ratio) + 1e-7)
                    bool_reject = torch.arange(len(bool_zero), device=bool_zero.device)[bool_zero][bool_neg_log_ratio]
                    bool_accept = torch.arange(len(bool_zero), device=bool_zero.device)[bool_zero][~bool_neg_log_ratio]

                    if bool_neg_log_ratio.sum() != 0:
                        eps_rand = torch.randn_like(x_check[bool_neg_log_ratio]).to(torch.float64)
                        eps_rand = torch.randn_like(x_check[bool_neg_log_ratio]).to(torch.float64)
                        x_back = t_steps[0] * eps_rand
                        x_hat[bool_reject] = x_back

                    log_ratio_prev_check[~bool_neg_log_ratio] = log_ratio[~bool_neg_log_ratio]
                    log_ratio_prev[bool_zero] = log_ratio_prev_check
                    bool_zero[bool_accept] = False

        # Euler step.
        denoised = net(x_hat, t_hat, labels).to(torch.float64)
        per_sample_nfe += 1
        if mode == 'debug':
            nonlocal total_nfe
            total_nfe += len(denoised)
        d_cur = (x_hat - denoised) / t_hat[:, None, None, None]
        x_next = x_hat + (t_next - t_hat)[:, None, None, None] * d_cur

        # Apply 2nd order correction.
        bool_2nd = lst_idx < num_steps - 1
        if bool_2nd.sum() != 0:
            labels_ = labels[bool_2nd] if labels is not None else None
            denoised = net(x_next[bool_2nd], t_next[bool_2nd], labels_).to(torch.float64)
            per_sample_nfe[bool_2nd] += 1
            if mode == 'debug':
                total_nfe += len(denoised)
            d_prime = (x_next[bool_2nd] - denoised) / t_next[bool_2nd][:, None, None, None]
            x_next[bool_2nd] = x_hat[bool_2nd] + (t_next - t_hat)[bool_2nd][:, None, None, None] * (0.5 * d_cur[bool_2nd] + 0.5 * d_prime)

        lst_idx = lst_idx + 1

        if warmup:
            assert adaptive_pickle == 'None'
            log_ratio = classifier_lib.get_grad_log_ratio(discriminator, vpsde, x_next, t_steps[lst_idx], net.img_resolution, time_min, time_max, labels, log_only=True).detach().cpu()
            for i in range(len(log_ratio)):
                lst_adaptive[lst_idx[i]].append(log_ratio[i].cpu())
            for i in range(len(log_ratio)):
                lst_adaptive2[lst_idx[i]].append(log_ratio[i].cpu() - log_ratio_prev[i].cpu())
            log_ratio_prev = log_ratio[:]
            return x_next, lst_idx, log_ratio_prev, per_sample_nfe

        if backsteps != 0.:
            bool_check = (lst_idx > min_backsteps) & (lst_idx <= max_backsteps) # img step이 min, max backstep 내에 있는지 확인
            bool_check_idx = torch.arange(len(lst_idx), device=bool_check.device)[bool_check] # img step이 min, max backstep 내에 있는 img index들
            if mode == 'debug':
                save_lst_idx = copy.deepcopy(lst_idx)
            if bool_check.sum() != 0:
                x_check = x_next[bool_check].clone().detach() # x_next (edm sampling 된 이미지) 가 check 대상인지 확인
                labels_ = labels[bool_check] if labels is not None else None # 이미지에 대한 class label
                log_ratio_prev_check = log_ratio_prev[bool_check] # log_ratio (algorithm에서 L_t)
                log_ratio = classifier_lib.get_grad_log_ratio(discriminator, vpsde, x_check, t_steps[lst_idx][bool_check], net.img_resolution, time_min, time_max, labels_, log_only=True).detach().cpu() # L_t-1
                bool_neg_log_ratio = log_ratio < adaptive2[lst_idx][bool_check] + torch.log(torch.rand_like(log_ratio) + 1e-7) + log_ratio_prev_check # log(L_t-1)<log(M_t-1) + log(u) + log(L_t) 면 Reject 
                bool_reject = torch.arange(len(bool_check), device=bool_check.device)[bool_check][bool_neg_log_ratio] # reject sample
                bool_accept = torch.arange(len(bool_check), device=bool_check.device)[bool_check][~bool_neg_log_ratio] # accept sample
                log_ratio_prev[bool_accept] = log_ratio[~bool_neg_log_ratio] # accetp log_ratio_prev 저장

                if bool_neg_log_ratio.sum() != 0: # reject할 게 있으면
                    labels_reject = labels[bool_reject] if labels is not None else None # reject image에 대한 class label
                    lst_idx[bool_reject] = lst_idx[bool_reject] - backsteps # backstep만큼 noise를 줄 예정

                    eps_rand = randn_like(x_check[bool_neg_log_ratio]) # eps ~ N(0, I)
                    x_back = x_next[bool_check][bool_neg_log_ratio] + (t_steps[lst_idx[bool_reject]] ** 2 - t_steps[lst_idx[bool_reject] + backsteps] ** 2).sqrt()[:, None, None, None] * eps_rand # t - 1 에서 t - 1 + backstep 만큼 noise 주기
                    log_ratio_back = classifier_lib.get_grad_log_ratio(discriminator, vpsde, x_back, t_steps[lst_idx][bool_check][bool_neg_log_ratio], net.img_resolution, time_min, time_max, labels_reject, log_only=True).detach().cpu() # L_t-1+backstep 계산
                    if litrs != 0: # MH sampling 진행한다면
                        denoised = net(x_back, t_steps[lst_idx][bool_check][bool_neg_log_ratio], labels_reject).to(torch.float64) # D(x_t-1+backstep , sigma)
                        per_sample_nfe[bool_reject] += 1 # Network 통과로 nfe += 1
                        d_cur = - (x_back - denoised) / (t_steps[lst_idx][bool_check][bool_neg_log_ratio][:, None, None, None] ** 2) # score 값 at x_t-1+backstep, EDM (3) 식
                        mu_cur = x_back + (t_steps[lst_idx[bool_reject]] ** 2 - t_steps[lst_idx[bool_reject] + backsteps] ** 2)[:, None, None, None] * d_cur # mean of marginal , NCSN++ (47) 식
                    log_ratio_prev[bool_reject] = log_ratio_back.clone().detach() # reject 대상 log_ratio_prev를 L_t-1+backstep으로 바꿈
                    x_check[bool_neg_log_ratio] = x_back.clone().detach() # check 대상을 x_t-1+backstep으로 바꿈

                    for _ in range(litrs): # MH sampling litrs만큼 진행
                        eps_rand = randn_like(x_check[bool_neg_log_ratio]) # eps ~ N(0,I)
                        x_back = x_next[bool_check][bool_neg_log_ratio] + (t_steps[lst_idx[bool_reject]] ** 2 - t_steps[lst_idx[bool_reject] + backsteps] ** 2).sqrt()[:, None, None, None] * eps_rand # backward 보냄 (위와 동일)
                        log_ratio_back = classifier_lib.get_grad_log_ratio(discriminator, vpsde, x_back, t_steps[lst_idx][bool_check][bool_neg_log_ratio], net.img_resolution, time_min, time_max, labels_reject, log_only=True).detach().cpu()
                        denoised = net(x_back, t_steps[lst_idx][bool_check][bool_neg_log_ratio], labels_reject).to(torch.float64) # 위와 동일
                        per_sample_nfe[bool_reject] += 1 # 위와 동일
                        d_back = - (x_back - denoised) / (t_steps[lst_idx][bool_check][bool_neg_log_ratio][:, None, None, None] ** 2) # 위와 동일
                        mu_back = x_back + (t_steps[lst_idx[bool_reject]] ** 2 - t_steps[lst_idx[bool_reject] + backsteps] ** 2)[:, None, None, None] * d_back # 위와 동일

                        log_kernel_ratio = (mu_cur * (2 * x_next[bool_check][bool_neg_log_ratio] - mu_cur)).sum(dim=(1,2,3)) - (mu_back * (2 * x_next[bool_check][bool_neg_log_ratio] - mu_back)).sum(dim=(1,2,3)) # score network 내 norm 부분 먼저 계산, - 지우고 분모에 있는 norm - 분자에 있는 norm 계산
                        log_kernel_ratio = (log_kernel_ratio / (2 * (t_steps[lst_idx[bool_reject]] ** 2 - t_steps[lst_idx[bool_reject] + backsteps] ** 2) * (t_steps[lst_idx[bool_reject] + backsteps] ** 2 / t_steps[lst_idx[bool_reject]] ** 2))).cpu() # 2sigma^2으로 나눠주는 부분, NCSN++ (47) 식
                        log_ratio_mph = log_ratio_back - log_ratio_prev[bool_check][bool_neg_log_ratio] + log_kernel_ratio # L_t(x_t(l+1)) / L(t(x_t(l))) * kernel_ratio

                        bool_mph_log_ratio = log_ratio_mph < torch.log(torch.rand_like(log_ratio_mph) + 1e-7) # acceptance prob. 보다 크면 reject
                        bool_mph_accept = torch.arange(len(bool_check), device=bool_check.device)[bool_check][bool_neg_log_ratio][~bool_mph_log_ratio] # accept 된 image num 확인
                        bool_mph_check_accept = torch.arange(len(x_check), device=bool_check.device)[bool_neg_log_ratio][~bool_mph_log_ratio] # 위와 동일? x_check이랑 bool_check 길이가 다른 경우가 있는지?
                        print(len(log_ratio_mph), len(bool_mph_accept))

                        x_check[bool_mph_check_accept] = x_back[~bool_mph_log_ratio] # check update
                        log_ratio_prev[bool_mph_accept] = log_ratio_back[~bool_mph_log_ratio] # log_ratio_prev update
                        mu_cur[~bool_mph_log_ratio] = mu_back[~bool_mph_log_ratio] # mu update

                    x_next[bool_reject] = x_check[bool_neg_log_ratio] # MH-sampling 완료
                    # log_ratio_prev_check[~bool_neg_log_ratio] = log_ratio[~bool_neg_log_ratio]

        return x_next, lst_idx, log_ratio_prev, per_sample_nfe

    def save_img(images, index, save_type="npz", batch_size=100, num_iter=-1):
        ## Save images.
        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        if save_type == "png":
            count = 0
            for image_np in images_np:
                image_path = os.path.join(outdir, f'{index*batch_size+count:06d}.png')
                count += 1
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)

        elif save_type == "npz":
            # r = np.random.randint(1000000)
            if num_iter != -1:
                file_name_npz = os.path.join(outdir, f"samples_{index}_{num_iter}.npz")
                file_name_png = os.path.join(outdir, f"sample_{index}_{num_iter}.png")
            else:
                file_name_npz = os.path.join(outdir, f"samples_{index}.npz")
                file_name_png = os.path.join(outdir, f"sample_{index}.png")
            with tf.io.gfile.GFile(file_name_npz, "wb") as fout:
                io_buffer = io.BytesIO()
                if class_labels == None:
                    np.savez_compressed(io_buffer, samples=images_np)
                else:
                    np.savez_compressed(io_buffer, samples=images_np, label=class_labels.cpu().numpy())
                fout.write(io_buffer.getvalue())

            nrow = int(np.sqrt(images_np.shape[0]))
            image_grid = make_grid(torch.tensor(images_np).permute(0, 3, 1, 2) / 255., nrow, padding=2)
            with tf.io.gfile.GFile(file_name_png, "wb") as fout:
                save_image(image_grid, fout)



    if dg_weight_2nd_order == 0.:
        dg_weight_2nd_order = dg_weight_1st_order
    print(f'dg_weight_1st_order: {dg_weight_1st_order}')
    print(f'dg_weight_2nd_order: {dg_weight_2nd_order}')

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    if mode == 'debug':
        import copy
        total_nfe = 0
        nfe_path = os.path.join(outdir, f'nfe_analysis.pickle')
        with open(nfe_path, 'rb') as f:
            dict_nfe = pickle.load(f)

    if adaptive_pickle == 'None':
        # Warmup
        lst_adaptive = [[] for i in range(len(t_steps))]
        lst_adaptive2 = [[] for i in range(len(t_steps))]
        x_next = latents.to(torch.float64) * t_steps[0]
        lst_idx = torch.zeros((latents.shape[0],), device=latents.device).long()
        log_ratio_prev = torch.zeros((latents.shape[0],))
        per_sample_nfe = torch.zeros((latents.shape[0],)).long()
        num_warm = 0
        while num_warm < iter_warmup:
            x_next, lst_idx, log_ratio_prev, per_sample_nfe = sampling_loop(x_next, lst_idx, log_ratio_prev, per_sample_nfe, class_labels, warmup=True)
            bool_fin = lst_idx == num_steps
            if bool_fin.sum() > 0:
                x_next[bool_fin] = torch.randn_like(x_next[bool_fin]).to(torch.float64) * t_steps[0]
                lst_idx[bool_fin] = torch.zeros_like(lst_idx[bool_fin]).long()
                if (class_labels is not None) & (class_idx is None):
                    class_labels[bool_fin] = torch.eye(net.label_dim, device=class_labels.device)[torch.randint(net.label_dim, size=[bool_fin.sum()], device=class_labels.device)]
                num_warm += 1
        lst_adaptive = [torch.stack(lst_adaptive[i]) for i in range(0, len(t_steps))]
        lst_adaptive2 = [torch.zeros(len(x_next)*iter_warmup)] + [torch.stack(lst_adaptive2[i]) for i in range(1, len(t_steps))]
        adaptive_path = os.path.join(outdir, f'adaptive.pickle')
        adaptive2_path = os.path.join(outdir, f'adaptive2.pickle')
        with open(adaptive_path, 'wb') as f:
            pickle.dump(lst_adaptive, f)
        with open(adaptive2_path, 'wb') as f:
            pickle.dump(lst_adaptive2, f)
    else:
        with open(adaptive_pickle, 'rb') as f:
            lst_adaptive = pickle.load(f)
        with open(adaptive_pickle2, 'rb') as f:
            lst_adaptive2 = pickle.load(f)
    adaptive = torch.zeros_like(t_steps).cpu()
    for k in range(len(t_steps)):
        adaptive[k] = max(0., torch.quantile(lst_adaptive[k], dg_weight_1st_order).item())
    print(adaptive)
    adaptive2 = torch.zeros_like(t_steps).cpu()
    for k in range(len(t_steps)):
        adaptive2[k] = max(0., torch.quantile(lst_adaptive2[k], dg_weight_2nd_order).item())
    print(adaptive2)

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    lst_idx = torch.zeros((latents.shape[0],), device=latents.device).long()
    log_ratio_prev = torch.zeros((latents.shape[0],))
    per_sample_nfe = torch.zeros((latents.shape[0],)).long()

    pbar = tqdm(desc='Number of re-init. samples')

    x_fin = torch.zeros_like(x_next)
    tot_per_sample_nfe = []
    total_samples = 0
    index = 0
    current_time = time.time()
    while total_samples <= num_samples:
        x_next, lst_idx, log_ratio_prev, per_sample_nfe = sampling_loop(x_next, lst_idx, log_ratio_prev, per_sample_nfe, class_labels)
        bool_fin = lst_idx == num_steps
        if bool_fin.sum() > 0:
            if (batch_size - total_samples % batch_size) <= bool_fin.sum():
                x_fin[total_samples % batch_size:] = x_next[bool_fin][:batch_size - total_samples % batch_size]
                if not do_seed:
                    r = np.random.randint(1000000)
                    save_img(x_fin, index=r)
                else:
                    save_img(x_fin, index=index)
                index += 1
                x_fin = torch.zeros_like(x_next)

                x_fin[:bool_fin.sum() - batch_size + total_samples % batch_size] = x_next[bool_fin][batch_size - total_samples % batch_size:]
                total_samples += bool_fin.sum()
            else:
                x_fin[total_samples % batch_size:total_samples % batch_size + bool_fin.sum()] = x_next[bool_fin]
                total_samples += bool_fin.sum()
            x_next[bool_fin] = torch.randn_like(x_next[bool_fin]).to(torch.float64) * t_steps[0]
            lst_idx[bool_fin] = torch.zeros_like(lst_idx[bool_fin]).long()
            log_ratio_prev[bool_fin] = torch.zeros_like(log_ratio_prev[bool_fin])

            tot_per_sample_nfe += per_sample_nfe[bool_fin].tolist()
            per_sample_nfe[bool_fin] = torch.zeros_like(per_sample_nfe[bool_fin]).long()

            if (class_labels is not None) & (class_idx is None):
                class_labels[bool_fin] = torch.eye(net.label_dim, device=class_labels.device)[torch.randint(net.label_dim, size=[bool_fin.sum()], device=class_labels.device)]

            if mode == 'debug':
                dict_nfe['total_nfe'] = total_nfe
                dict_nfe['total_samples'] = total_samples.item()
                dict_nfe['tot_per_sample_nfe'] = tot_per_sample_nfe
                with open(nfe_path, 'wb') as f:
                    pickle.dump(dict_nfe, f)
    print(time.time()-current_time)

    if mode == 'debug':
        dict_nfe['total_nfe'] = dict_nfe.get('total_nfe', 0) + total_nfe
        dict_nfe['total_samples'] = dict_nfe.get('total_samples', 0) + num_samples
        dict_nfe['tot_per_sample_nfe'] = tot_per_sample_nfe

        with open(nfe_path, 'wb') as f:
            pickle.dump(dict_nfe, f)
    # return x_next

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'batch_size',     help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=100, show_default=True)

@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)

@click.option('--solver',                  help='Ablate ODE solver', metavar='euler|heun',                          type=click.Choice(['euler', 'heun']))
@click.option('--disc', 'discretization',  help='Ablate time step discretization {t_i}', metavar='vp|ve|iddpm|edm', type=click.Choice(['vp', 've', 'iddpm', 'edm']))
@click.option('--schedule',                help='Ablate noise schedule sigma(t)', metavar='vp|ve|linear',           type=click.Choice(['vp', 've', 'linear']))
@click.option('--scaling',                 help='Ablate signal scaling s(t)', metavar='vp|none',                    type=click.Choice(['vp', 'none']))

#---------------------------------------------------------------------------- Options for Discriminator-Guidance
## Sampling configureation
@click.option('--do_seed',                 help='Applying manual seed or not', metavar='INT',                       type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--seed',                    help='Seed number',                 metavar='INT',                       type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--num_samples',             help='Num samples',                 metavar='INT',                       type=click.IntRange(min=1), default=50000, show_default=True)
@click.option('--save_type',               help='png or npz',                  metavar='png|npz',                   type=click.Choice(['png', 'npz']), default='npz')
@click.option('--device',                  help='Device', metavar='STR',                                            type=str, default='cuda:0')

## DG configuration
@click.option('--dg_weight_1st_order',     help='Weight of DG for 1st prediction',       metavar='FLOAT',           type=float, default=0., show_default=True)
@click.option('--dg_weight_2nd_order',     help='Weight of DG for 2nd prediction',       metavar='FLOAT',           type=click.FloatRange(min=0), default=0., show_default=True)
@click.option('--time_min',                help='Minimum time[0,1] to apply DG', metavar='FLOAT',                   type=click.FloatRange(min=0., max=1.), default=0.01, show_default=True)
@click.option('--time_max',                help='Maximum time[0,1] to apply DG', metavar='FLOAT',                   type=click.FloatRange(min=0., max=1.), default=1.0, show_default=True)
@click.option('--boosting',                help='If true, dg scale up low log ratio samples', metavar='INT',        type=click.IntRange(min=0), default=0, show_default=True)

## Discriminator checkpoint
@click.option('--pretrained_classifier_ckpt',help='Path of ADM classifier(latent extractor)',  metavar='STR',       type=str, default='checkpoints/ADM_classifier/32x32_classifier.pt', show_default=True)
@click.option('--discriminator_ckpt',      help='Path of discriminator',  metavar='STR',                            type=str, default='checkpoints/discriminator/cifar_uncond/discriminator_60.pt', show_default=True)

## Discriminator architecture
@click.option('--cond',                    help='Is it conditional discriminator?', metavar='INT',                  type=click.IntRange(min=0, max=1), default=0, show_default=True)
@click.option('--backsteps',               help='backsteps', metavar='INT',                                         type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--min_backsteps',           help='min_backsteps', metavar='INT',                                     type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--max_backsteps',           help='max_backsteps', metavar='INT',                                     type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--mode',                    help='Mode', metavar='STR',                                              type=str, default='default')
@click.option('--adaptive_pickle',         help='Path of adaptive',  metavar='STR',                                 type=str, default='None', show_default=True)
@click.option('--adaptive_pickle2',        help='Path of adaptive2',  metavar='STR',                                type=str, default='None', show_default=True)
@click.option('--iter_warmup',             help='iteration of warmup', metavar='INT',                               type=click.IntRange(min=0), default=10, show_default=True)
@click.option('--max_iter',                help='max_iter', metavar='INT',                                          type=click.IntRange(min=0), default=999999, show_default=True)
@click.option('--snr',                     help='SNR', metavar='FLOAT',                                             type=click.FloatRange(min=0.), default=0., show_default=True)
@click.option('--litrs',                   help='langevin_iter', metavar='INT',                                     type=click.IntRange(min=0), default=3, show_default=True)

def main(boosting, time_min, time_max, dg_weight_1st_order, dg_weight_2nd_order, cond, pretrained_classifier_ckpt, discriminator_ckpt, save_type, batch_size, do_seed, snr, litrs, seed, num_samples, network_pkl, outdir, class_idx, device, backsteps, min_backsteps, max_backsteps, mode, adaptive_pickle, adaptive_pickle2, iter_warmup, max_iter, **sampler_kwargs):
    ## Load pretrained score network.
    print(f'Loading network from "{network_pkl}"...')
    with open(network_pkl, 'rb') as f:
        net = pickle.load(f)['ema'].to(device)

    ## Load discriminator
    if 'ffhq' in network_pkl:
        depth = 4
    else:
        depth = 2
    discriminator = classifier_lib.get_discriminator(pretrained_classifier_ckpt, discriminator_ckpt,
                                                 net.label_dim and cond, net.img_resolution, device, depth=depth, enable_grad=True)
    print(discriminator)
    vpsde = classifier_lib.vpsde()

    ## Loop over batches.
    print(f'Generating {num_samples} images to "{outdir}"...')
    os.makedirs(outdir, exist_ok=True)

    if mode == 'debug':
        dict_nfe = {'dict_nfe': {}, 'total_nfe': 0, 'total_samples': 0}
        nfe_path = os.path.join(outdir, f'nfe_analysis.pickle')
        with open(nfe_path, 'wb') as f:
            pickle.dump(dict_nfe, f)

    ## Set seed
    if do_seed:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
    ## Pick latents and labels.
    latents = torch.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
    class_labels = None
    if net.label_dim:
        class_labels = torch.eye(net.label_dim, device=device)[torch.randint(net.label_dim, size=[batch_size], device=device)]
    if class_idx is not None:
        class_labels[:, :] = 0
        class_labels[:, class_idx] = 1

    ## Generate images.
    sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
    edm_sampler(boosting, time_min, time_max, vpsde, dg_weight_1st_order, dg_weight_2nd_order, discriminator,
                net, latents, class_labels, randn_like=torch.randn_like, backsteps=backsteps, min_backsteps=min_backsteps,
                max_backsteps=max_backsteps, mode=mode, outdir=outdir, adaptive_pickle=adaptive_pickle, adaptive_pickle2=adaptive_pickle2, class_idx=class_idx,
                batch_size=batch_size, num_samples=num_samples, iter_warmup=iter_warmup, max_iter=max_iter, do_seed=do_seed, snr=snr, litrs=litrs, **sampler_kwargs)

#----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
#----------------------------------------------------------------------------
