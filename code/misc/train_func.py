import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import h5py as hp
import time

from models import *
from utils import *
from losses import *
from neat_func import *
from tqdm import tqdm
from psf_torch import PsfGenerator3D
from scipy.ndimage import gaussian_filter
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR


def train(args, 
          im, 
          wf_init=None,
          display_figures=True, 
          save_files=False, 
          mode_num=3+4+5, 
          excluded_modes=[5,11,15],
          device_num=0,
          freeze_psf=False):
    
    if args.pretraining_lp_sigma is not None:
        y_lp = gaussian_filter(im.copy(), args.pretraining_lp_sigma)
        y = torch.from_numpy(y_lp).type(dtype).cuda(device_num).view(im.shape[0], im.shape[1], im.shape[2])


    INPUT_HEIGHT = y.shape[1]
    INPUT_WIDTH = y.shape[2]
    INPUT_DEPTH = y.shape[0]

    psf = PsfGenerator3D(psf_shape=(im.shape[0], im.shape[1], im.shape[2]), 
                         units=(args.psf_dz, args.psf_dy, args.psf_dx), 
                         na_exc=args.na_exc, 
                         lam_detection=args.excitation_wavelength, 
                         n=args.n_obj)

    coordinates = input_coord_2d(INPUT_WIDTH, INPUT_HEIGHT).cuda(device_num)

    if args.encoding_option == 'radial':
        coordinates = radial_encoding(coordinates, 
                                      args.radial_encoding_angle, 
                                      args.radial_encoding_depth).cuda(device_num).detach() 

    else:
        return
    
    net_obj = NF(D = args.nerf_num_layers,
                 W = args.nerf_num_filters,
                 skips = args.nerf_skips, 
                 in_channels = coordinates.shape[-1], 
                 out_channels = INPUT_DEPTH).cuda(device_num)

    
    """
    Learning step #1
    """
    print('Running - Learning step #1')
    if args.pretraining:
        t_start = time.time()

        optimizer = torch.optim.RAdam([{'params':net_obj.parameters(), 
                                        'lr':args.pretraining_lr}],
                                     betas=(args.opt_beta_1_obj, 
                                            args.opt_beta_2_obj), 
                                     eps=args.opt_eps)
        
        loss_list = np.empty(shape = (1 + args.pretraining_num_iter, ))
        loss_list[:] = np.NaN

        
        if args.lr_schedule == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, 
                                          args.pretraining_num_iter, 
                                          args.pretraining_last_epoch_lr)

        
        for step in tqdm(range(args.pretraining_num_iter)):
            out_x = net_obj(coordinates)

            if args.nerf_beta is None:
                out_x = args.nerf_max_val * nn.Sigmoid()(out_x)
            else:
                out_x = nn.Softplus(beta = args.nerf_beta)(out_x)

            out_x_m = out_x.view(im.shape[1], im.shape[2], im.shape[0]).permute(2, 0, 1)

            loss = hybrid_loss(out_x_m, y, 
                               args.pretraining_ssim_weight, 
                               args.pretraining_measurement_scalar, 
                               args.relative_mse_eps, 
                               mode=args.relative_error_mode)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.lr_schedule == 'cosine':
                scheduler.step()

            elif args.lr_schedule == 'exponential':
                optimizer.param_groups[0]['lr'] *= np.power(args.pretraining_last_epoch_lr/args.pretraining_lr, 1/args.pretraining_num_iter)

            
            loss_list[step] = loss.item()

        t_end = time.time()
        print('Learning step #1 - Elapsed time: ' + str(t_end - t_start) + ' seconds.')

        
        if display_figures:
            plt.figure(figsize = (40, 5))
            plt.subplot(1,7,1); plt.imshow(out_x_m.detach().cpu().numpy().max(0)); plt.colorbar(fraction=0.046)
            plt.subplot(1,7,2); plt.imshow(y.detach().cpu().numpy().max(0)); plt.colorbar(fraction=0.046)
            plt.subplot(1,7,3); plt.imshow(out_x_m.detach().cpu().numpy().max(1)); plt.colorbar(fraction=0.046)
            plt.subplot(1,7,4); plt.imshow(y.detach().cpu().numpy().max(1)); plt.colorbar(fraction=0.046)
            plt.subplot(1,7,5); plt.imshow(out_x_m.detach().cpu().numpy().max(2)); plt.colorbar(fraction=0.046)
            plt.subplot(1,7,6); plt.imshow(y.detach().cpu().numpy().max(2)); plt.colorbar(fraction=0.046)
            plt.subplot(1,7,7); plt.plot(loss_list)
            plt.show()
            
    else:
        net_obj.train()

    
    """
    Learning step #2
    """
    print('Running - Learning step #2')
    y_pre = out_x_m.detach().cpu().numpy()
    y = torch.from_numpy(im.copy()).type(dtype).cuda(device_num).view(im.shape[0], im.shape[1], im.shape[2])

    if wf_init is None:
        net_ker = optimal_kernel(max_val = args.kernel_max_val, 
                                 order_up_to = args.kernel_order_up_to) # 5e-2
    else:
        net_ker = optimal_kernel(init_value = wf_init)
        
    wf_c = 2. if args.normalized else 1.
    
    opt_config = [{'params':net_obj.parameters(), 
                   'lr':args.training_lr_obj, 
                   'betas':(args.opt_beta_1_obj, 
                            args.opt_beta_2_obj)},
                  {'params':net_ker.parameters(), 
                   'lr':args.training_lr_ker, 
                   'betas':(args.opt_beta_1_ker, 
                            args.opt_beta_2_ker)}]

    
    if args.include_lb:
        lb = learnable_background_v2(init_val = args.lb_init_val, 
                                     image_stack = y, 
                                     rank = args.lb_rank)
        
        opt_config += [{'params':lb.parameters(), 
                        'lr':args.training_lr_lb, 
                        'betas':(args.opt_beta_1_lb, 
                                 args.opt_beta_2_lb)}]

    if args.adaptive_reg:
        ar = adaptive_registration(y.shape[0], 
                                   include_scalar=args.include_scalar).cuda(0)
        
        opt_config += [{'params':ar.parameters(), 
                        'lr':args.training_lr_ar, 
                        'betas':(args.opt_beta_1_ar, 
                                 args.opt_beta_2_ar)}]
        
    if args.training_opt == 'Adam':
        optimizer = torch.optim.Adam(opt_config, 
                                     eps=args.opt_eps) # 07/26/23: 1e-15, 0.0
        
    elif args.training_opt == 'RAdam':
        optimizer = torch.optim.RAdam(opt_config, 
                                      eps=args.opt_eps) # 07/26/23: 1e-15, 0.0

    if args.lr_schedule == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, 
                                      args.training_num_iter, 
                                      args.training_last_epoch_lr)


    loss_list = np.empty(shape = (1 + args.training_num_iter, )); loss_list[:] = np.NaN
    wfe_list = np.empty(shape = (1 + args.training_num_iter, )); wfe_list[:] = np.NaN
    lr_obj_list = np.empty(shape = (1 + args.training_num_iter, )); lr_obj_list[:] = np.NaN
    lr_ker_list = np.empty(shape = (1 + args.training_num_iter, )); lr_ker_list[:] = np.NaN
    lr_lb_list = np.empty(shape = (1 + args.training_num_iter, )); lr_lb_list[:] = np.NaN

    
    if args.adaptive_reg:
        ar_list = np.empty(shape = (1 + args.training_num_iter, y.shape[0], 2, 3)); ar_list[:] = np.NaN

    
    t_start = time.time()

    for step in tqdm(range(args.training_num_iter)):
        if args.adaptive_reg:
            if args.include_scalar:
                y_ar, As, S = ar(y)
            else:
                y_ar, As = ar(y)

        
        out_x = net_obj(coordinates)

        
        if args.nerf_beta is None:
            out_x = args.nerf_max_val * nn.Sigmoid()(out_x)
            
        else:
            out_x = nn.Softplus(beta = args.nerf_beta)(out_x)
            out_x = torch.minimum(torch.full_like(out_x, args.nerf_max_val), out_x) # 30.0

        out_x_m = out_x.view(im.shape[1], im.shape[2], im.shape[0]).permute(2, 0, 1)

        
        if not freeze_psf:
            wf = net_ker.k
        else:
            wf = wf_ext

        
        out_k_m = psf.incoherent_psf(wf, normalized=args.normalized)
        out_k_m = out_k_m ** 2
        out_k_m /= torch.sum(out_k_m)

        k_vis = psf.masked_phase_array(wf, normalized=args.normalized)

        if args.padding_xy != 0 or args.padding_z != 0:
            out_x_m = torch.nn.functional.pad(out_x_m.unsqueeze(0).unsqueeze(0), 
                                              (args.padding_xy, args.padding_xy, args.padding_xy, args.padding_xy, args.padding_z, args.padding_z), 
                                              mode = args.padding_mode).squeeze(0).squeeze(0)
            
            out_k_m = torch.nn.functional.pad(out_k_m.unsqueeze(0).unsqueeze(0), 
                                              (args.padding_xy, args.padding_xy, args.padding_xy, args.padding_xy, args.padding_z, args.padding_z), 
                                              mode = args.padding_mode).squeeze(0).squeeze(0)       
            
        out_y = fft_convolve(out_x_m, out_k_m, mode='fftn')

        if args.padding_xy != 0 or args.padding_z != 0:
            out_x_m = out_x_m[args.padding_z:im.shape[0]+args.padding_z, args.padding_xy:im.shape[1]+args.padding_xy, args.padding_xy:im.shape[2]+args.padding_xy]
            out_k_m = out_k_m[args.padding_z:im.shape[0]+args.padding_z, args.padding_xy:im.shape[1]+args.padding_xy, args.padding_xy:im.shape[2]+args.padding_xy]
            out_y = out_y[args.padding_z:im.shape[0]+args.padding_z, args.padding_xy:im.shape[1]+args.padding_xy, args.padding_xy:im.shape[2]+args.padding_xy]
        

        if args.include_lb:
            out_y += torch.minimum(lb(), torch.full_like(lb(), args.lb_max))

        
        if not args.adaptive_reg:
            loss = hybrid_loss(out_y, y,
                               args.ssim_weight, 
                               1.0, 
                               args.relative_mse_eps, 
                               mode=args.relative_error_mode)
            
        else:
            loss = hybrid_loss(out_y, y_ar,
                           args.ssim_weight, 
                           1.0, 
                           args.relative_mse_eps, 
                           mode=args.relative_error_mode)

        
        for m in excluded_modes:
            loss += single_mode_control(wf, m-4, -0.0, 0.0)

        
        loss += args.nld_z * nld_1d(out_x_m, axis='z', 
                                    upper_bound = args.nld_z_ubd, 
                                    lower_bound = args.nld_z_lbd,
                                    negative_slope=args.nld_z_negative_slope, 
                                    normalize = args.tv_normalize)

        
        loss += args.hessian_v2 * hessian_loss_v2(out_x_m, args.psf_dx, args.psf_dz, 
                                                  args.hessian_v2_eps,
                                                  args.ab_eps,
                                                  args.ab_weight)

        
        if args.include_lb:
            loss += args.lb_reg * torch.square(lb()).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if step == 0 or step == 250-1 or step == 1000-1 or step == 2000-1 or step == 5000-1:           
            if display_figures:
                plt.figure(figsize = (40, 10))
                plt.subplot(2,7,1); plt.plot(wfe_list)
                plt.subplot(2,7,2); plt.yticks([0, 2, 4, 6, 8, 10], ['4', '6', '8', '10', '12', '14'])
                plt.barh(np.arange(0, mode_num)-0.2, width=wf_c*wf.detach().cpu().numpy(), color = 'r', linestyle = '--', height=0.4, alpha=0.25)
                plt.subplot(2,7,3); plt.imshow(np.fft.fftshift(k_vis.detach().cpu().numpy()))
                plt.subplot(2,7,4); plt.imshow(y.detach().cpu().numpy().max(0), cmap = args.cmap); plt.colorbar(fraction=0.046)
                plt.subplot(2,7,5); plt.imshow(out_x_m.detach().cpu().numpy().max(0), cmap = args.cmap); plt.colorbar(fraction=0.046)
                plt.subplot(2,7,6); plt.imshow(y.detach().cpu().numpy().max(1), cmap = args.cmap, aspect=args.psf_dz/args.psf_dx); plt.colorbar(fraction=0.046)
                plt.subplot(2,7,7); plt.imshow(out_x_m.detach().cpu().numpy().max(1), cmap = args.cmap, aspect=args.psf_dz/args.psf_dx); plt.colorbar(fraction=0.046)
                plt.subplot(2,7,9); plt.imshow(y.detach().cpu().numpy()[y.shape[0]//2], cmap = args.cmap); plt.colorbar(fraction=0.046)
                plt.subplot(2,7,12); plt.imshow(y.detach().cpu().numpy()[:, y.shape[1]//2], cmap = args.cmap); plt.colorbar(fraction=0.046)
                
                if args.include_lb:
                    plt.subplot(2,7,8); plt.imshow(lb().detach().cpu().numpy()[y.shape[0]//2], cmap = args.cmap); plt.colorbar(fraction=0.046)
                    plt.subplot(2,7,10); plt.imshow((y-lb()).detach().cpu().numpy()[y.shape[0]//2], cmap = args.cmap); plt.colorbar(fraction=0.046)
                    plt.subplot(2,7,11); plt.imshow(lb().detach().cpu().numpy()[:, y.shape[1]//2], cmap = args.cmap); plt.colorbar(fraction=0.046)
                    plt.subplot(2,7,13); plt.imshow((y-lb()).detach().cpu().numpy()[:, y.shape[1]//2], cmap = args.cmap); plt.colorbar(fraction=0.046)

                if args.adaptive_reg:
                    plt.subplot(2,7,14)
                    plt.imshow(y_ar.detach().cpu().numpy().max(1), cmap = args.cmap, aspect=args.psf_dz/args.psf_dx); plt.colorbar(fraction=0.046)
                plt.show()


        if args.lr_schedule == 'cosine':
            scheduler.step()
            
        elif args.lr_schedule == 'exponential':
            optimizer.param_groups[0]['lr'] *= np.power(args.training_last_epoch_lr/args.training_lr_obj, 1/args.training_num_iter)
            optimizer.param_groups[1]['lr'] *= np.power(args.training_last_epoch_lr/args.training_lr_ker, 1/args.training_num_iter)
            
            if args.include_lb:
                optimizer.param_groups[2]['lr'] *= np.power(args.training_last_epoch_lr/args.training_lr_lb, 1/args.training_num_iter)

            if args.adaptive_reg:
                optimizer.param_groups[-1]['lr'] *= np.power(args.training_last_epoch_lr/args.training_lr_ar, 1/args.training_num_iter)
                

        loss_list[step] = loss.item()
        wfe_list[step] = torch_to_np(args.excitation_wavelength * 1e3 * torch.sqrt(torch.sum(torch.square(wf_c*wf)))) # wave -> nm RMS
        lr_obj_list[step] = optimizer.param_groups[0]['lr']
        lr_ker_list[step] = optimizer.param_groups[1]['lr']
        

        if args.include_lb:
            lr_lb_list[step] = optimizer.param_groups[2]['lr']

        
        if args.adaptive_reg:
            ar_list[step] = As.detach().cpu().numpy()
            
    
    t_end = time.time()
    print('Learning step #2 - Elapsed time: ' + str(t_end - t_start) + ' seconds.')


    y = torch_to_np(y)
    out_x_m = torch_to_np(out_x_m)
    out_k_m = torch_to_np(out_k_m)
    out_y = torch_to_np(out_y)
    wf = wf_c * torch_to_np(wf) 

    
    if save_files:
        hf = hp.File(args.rec_save_path_prefix + 'rec.h5', 'w')
        hf.create_dataset('out_x_m', data=out_x_m)
        hf.create_dataset('out_k_m', data=out_k_m)
        hf.create_dataset('out_y', data=out_y)
        hf.create_dataset('wf', data=wf)
        hf.create_dataset('loss_list', data=loss_list)
        hf.create_dataset('y', data=y)
        hf.create_dataset('y_min', data=args.y_min)
        hf.create_dataset('y_max', data=args.y_max)
        hf.close()


        f = open(args.rec_save_path_prefix + 'hyperparameters_summary.txt', 'w')
        f.write(str(vars(args)))
        f.close()
        

    if display_figures:
        plt.figure(figsize = (40, 5))
        plt.subplot(1,8,1); plt.imshow(im.max(0), cmap = args.cmap)
        plt.subplot(1,8,2); plt.imshow(out_x_m.max(0), cmap = args.cmap)
        plt.subplot(1,8,3); plt.imshow(out_k_m[y.shape[0]//2-100:y.shape[0]//2+100, y.shape[1]//2, y.shape[2]//2-100:y.shape[2]//2+100], cmap = args.cmap)
        plt.subplot(1,8,5); plt.xlabel('Mode'); plt.xticks([0, 2, 4, 6, 8, 10], ['4', '6', '8', '10', '12', '14']); plt.ylabel('Coefficient (wave)'); plt.stem(wf)
        plt.subplot(1,8,6); plt.imshow(np.fft.fftshift(k_vis.detach().cpu().numpy()))
        plt.subplot(1,8,7); plt.semilogy(loss_list)
        plt.subplot(1,8,8); plt.plot(wfe_list)
        plt.savefig(args.rec_save_path_prefix + 'results_figure.pdf', bbox_inches = 'tight')
        plt.show()


    if args.adaptive_reg:
        if args.include_lb:
            return y, y_pre, out_x_m, out_k_m, out_y, wf, y_ar, np.fft.fftshift(k_vis.detach().cpu().numpy()), loss_list, torch.minimum(lb(), torch.full_like(lb(), args.lb_max)), As
        else:
            return y, y_pre, out_x_m, out_k_m, out_y, wf, y_ar, np.fft.fftshift(k_vis.detach().cpu().numpy()), loss_list, As
        
    else:
        if args.include_lb:
            return y, y_pre, out_x_m, out_k_m, out_y, wf, np.fft.fftshift(k_vis.detach().cpu().numpy()), loss_list, torch.minimum(lb(), torch.full_like(lb(), args.lb_max))
        else:
            return y, y_pre, out_x_m, out_k_m, out_y, wf, np.fft.fftshift(k_vis.detach().cpu().numpy()), loss_list