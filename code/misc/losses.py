import torch

dtype = torch.cuda.FloatTensor

def tv_1d(img, axis = 'z', normalize = False):
    if axis == 'z':
        if not normalize:
            _variance = torch.sum(torch.abs(img[0:-1, :, :] - img[1::, :, :]))
        else:
            _variance = torch.mean(torch.abs(img[0:-1, :, :] - img[1::, :, :]))
            
    elif axis == 'y':
        if not normalize:
            _variance = torch.sum(torch.abs(img[:, 0:-1, :] - img[:, 1::, :]))
        else:
            _variance = torch.mean(torch.abs(img[:, 0:-1, :] - img[:, 1::, :]))
            
    else:
        if not normalize:
            _variance = torch.sum(torch.abs(img[:, :, 0:-1] - img[:, :, 1::]))
        else:
            _variance = torch.mean(torch.abs(img[:, :, 0:-1] - img[:, :, 1::]))
            
    return _variance


def tv_1d_iso(img, axis = 'z', normalize = False):
    if axis == 'z':
        if not normalize:
            _variance = torch.sum(torch.square(img[0:-1, :, :] - img[1::, :, :]))
        else:
            _variance = torch.mean(torch.square(img[0:-1, :, :] - img[1::, :, :]))
            
    elif axis == 'y':
        if not normalize:
            _variance = torch.sum(torch.square(img[:, 0:-1, :] - img[:, 1::, :]))
        else:
            _variance = torch.mean(torch.square(img[:, 0:-1, :] - img[:, 1::, :]))
            
    else:
        if not normalize:
            _variance = torch.sum(torch.square(img[:, :, 0:-1] - img[:, :, 1::]))
        else:
            _variance = torch.mean(torch.square(img[:, :, 0:-1] - img[:, :, 1::]))
            
    return _variance


def tv_2d(img):
    # img: D x W x H
    h_variance = torch.sum(torch.abs(img - torch.roll(img, 1, dims = 1)))
    w_variance = torch.sum(torch.abs(img - torch.roll(img, 1, dims = 2)))
    
    return h_variance + w_variance


def tv_3d(img, weights):
    # d_variance = torch.sum(torch.abs(img - torch.roll(img, 1, dims = 0)))
    # h_variance = torch.sum(torch.abs(img - torch.roll(img, 1, dims = 1)))
    # w_variance = torch.sum(torch.abs(img - torch.roll(img, 1, dims = 2)))
    d_variance = torch.sum(torch.abs(img[0:-1, :, :] - img[1::, :, :]))
    h_variance = torch.sum(torch.abs(img[:, 0:-1, :] - img[:, 1::, :]))
    w_variance = torch.sum(torch.abs(img[:, :, 0:-1] - img[:, :, 1::]))
    
    return weights[0] * d_variance + weights[1] * h_variance + weights[2] * w_variance


def DnCNN_loss(model, inp, tar, rnd):
    with torch.no_grad():
        return ((model(inp[rnd, :, :].unsqueeze(1) * 2**8) - model(tar[rnd, :, :].unsqueeze(1) * 2**8))**2).mean()


def npcc_loss(y_pred, y_true):
    up = torch.mean((y_pred - torch.mean(y_pred)) * (y_true - torch.mean(y_true)))
    down = torch.std(y_pred) * torch.std(y_true)
    loss = 1.0 - up / down

    return loss.type(dtype)


def ssim_loss(img1, img2, return_cs = False):
    mu1 = torch.mean(img1)
    mu2 = torch.mean(img2)
        
    mu1_mu2 = mu1 * mu2

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    sigma1_sq = torch.mean(img1 * img1) - mu1_sq
    sigma2_sq = torch.mean(img2 * img2) - mu2_sq
    sigma12 = torch.mean(img1 * img2) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    loss = 1.0 - (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / torch.clamp((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2), min = 1e-8, max = 1e8))

    if return_cs:
        cs = (2 * sigma12 + C2) / torch.clamp(sigma1_sq + sigma2_sq + C2, min=1e-8, max=1e8)
        return loss.type(dtype), cs
    
    else:
        return loss.type(dtype)
    
    
def msssim_loss(img1, img2, levels=4):
    mcs = []
    weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = img2.new_tensor(weights)
        
    for i in range(levels):
        l, cs = ssim_loss(img1, img2, return_cs=True)

        if i < levels - 1:
            mcs.append(torch.relu(cs))
            img1 = torch.nn.AvgPool2d(kernel_size=2)(img1)
            img2 = torch.nn.AvgPool2d(kernel_size=2)(img2)

    l = torch.relu(l)  # (batch, channel)
    mcs_and_ssim = torch.stack(mcs + [l], dim=0)  # (level, batch, channel)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights.view(-1, 1, 1), dim=0)
    
    return ms_ssim_val.mean()


def multiscale_hybrid_loss(img1, 
                           img2, 
                           ssim_weight, 
                           pretraining_measurement_scalar, 
                           measurement_background_noise_level,
                           pooling_levels=4,
                           pooling_in_2d=True,
                           kernel_size=2,
                           level_weights=None):
    
    if level_weights is None:
        assert pooling_levels == 0 
        loss = hybrid_loss(img1, img2, ssim_weight, pretraining_measurement_scalar, measurement_background_noise_level)
    
    else:
        assert len(level_weights) == pooling_levels+1
        level_weights = torch.from_numpy(level_weights/level_weights.sum()).cuda(0)
        loss = level_weights[0] * hybrid_loss(img1, img2, ssim_weight, pretraining_measurement_scalar, measurement_background_noise_level)

    for level in range(pooling_levels):
        if pooling_in_2d:
            img1 = torch.nn.AvgPool2d(kernel_size=kernel_size)(img1)
            img2 = torch.nn.AvgPool2d(kernel_size=kernel_size)(img2)

        else:
            img1 = torch.nn.AvgPool3d(kernel_size=kernel_size)(img1.unsqueeze(0)).squeeze()
            img2 = torch.nn.AvgPool3d(kernel_size=kernel_size)(img2.unsqueeze(0)).squeeze()

        loss += level_weights[level+1] * hybrid_loss(img1, img2, ssim_weight, pretraining_measurement_scalar, measurement_background_noise_level)
        
    return loss.type(dtype)
        
        
# def hybrid_loss(img1, 
#                 img2, 
#                 ssim_weight, 
#                 pretraining_measurement_scalar, 
#                 measurement_background_noise_level):
    
#     loss = ssim_weight * ssim_loss(img1, pretraining_measurement_scalar * img2)
#     relative_l2_error = torch.square(img1 - pretraining_measurement_scalar * img2) / (torch.square(img1.detach()) + measurement_background_noise_level)
#     loss += (1.-ssim_weight) * relative_l2_error.mean()
    
#     return loss.type(dtype)


def nld_1d(im, axis, upper_bound, lower_bound, negative_slope=0.01, normalize=False):
    def reg_fun(diff, ubd, lbd):
        if lbd == 0:
            return torch.maximum(diff, torch.full_like(diff, ubd)) + negative_slope * torch.minimum(diff, torch.full_like(diff, ubd))
        else:
            return torch.maximum(diff, torch.full_like(diff, ubd)) + negative_slope * torch.maximum(torch.minimum(diff, torch.full_like(diff, ubd)), torch.full_like(diff, lbd)) + torch.minimum(diff, torch.full_like(diff, lbd))
    
    if axis == 'z':
        im_diff = torch.abs(torch.roll(im, 1, dims=0) - im)
    elif axis == 'y':
        im_diff = torch.abs(torch.roll(im, 1, dims=1) - im)
    else:
        im_diff = torch.abs(torch.roll(im, 1, dims=2) - im)
    
    if normalize:
        loss_l = reg_fun(im_diff, upper_bound, lower_bound).mean()
    else:
        loss_l = reg_fun(im_diff, upper_bound, lower_bound).sum()
    
    return loss_l


def nonlinear_diffusion_loss(img, upper_bound):
    # h_abs_grad = torch.abs(img[:, :, :-1] - img[:, :, 1:])
    # w_abs_grad = torch.abs(img[:, :-1, :] - img[:, 1:, :])
    d_abs_grad = torch.abs(img[:-1, :, :] - img[1:, :, :])
    
    def reg_fun(abs_grad, ubd):
        return torch.minimum(abs_grad, torch.full_like(abs_grad, ubd))
        
    # loss_h = torch.sum(reg_fun(h_abs_grad, upper_bound))
    # loss_w = torch.sum(reg_fun(w_abs_grad, upper_bound))
    
    # return (loss_h + loss_w)
    
    loss_d = torch.sum(reg_fun(d_abs_grad, upper_bound))
    
    return loss_d


def tv_range_loss(img, lower_bound):
    h_abs_grad = torch.abs(img[:, :, :-1] - img[:, :, 1:])
    w_abs_grad = torch.abs(img[:, :-1, :] - img[:, 1:, :])
    
    def reg_fun(abs_grad, lbd):
        return torch.maximum(abs_grad, torch.full_like(abs_grad, lbd))
        
    loss_h = torch.sum(reg_fun(h_abs_grad, lower_bound))
    loss_w = torch.sum(reg_fun(w_abs_grad, lower_bound))
    
    return (loss_h + loss_w)


def second_order_diff_loss(img, upper_bound):
    second_order_diff = torch.abs(img[:, :-1, :-1] - img[:, :-1, 1:] - img[:, 1:, :-1] + img[:, 1:, 1:])
    
    def reg_fun(diff, lbd):
        return torch.minimum(diff, torch.full_like(diff, lbd))
        
    loss = torch.sum(reg_fun(second_order_diff, upper_bound))
    
    return loss


def hybrid_loss(img1, 
                img2, 
                ssim_weight, 
                pretraining_measurement_scalar, 
                relative_mse_eps,
                mode='l2'):
    
    if ssim_weight == 0:
        if mode == 'l2':
            loss = torch.square((img1 - pretraining_measurement_scalar * img2) / (img1.detach() + relative_mse_eps))
        else:
            loss = torch.abs((img1 - pretraining_measurement_scalar * img2) / (img1.detach() + relative_mse_eps))
        loss = loss.mean()

    elif ssim_weight < 1: 
        loss = ssim_weight * ssim_loss(img1, pretraining_measurement_scalar * img2)
        if mode == 'l2':
            relative_error = torch.square((img1 - pretraining_measurement_scalar * img2) / (img1.detach() + relative_mse_eps))
        else:
            relative_error = torch.abs((img1 - pretraining_measurement_scalar * img2) / (img1.detach() + relative_mse_eps))
        loss += (1.-ssim_weight) * relative_error.mean()
        
    else:
        loss = ssim_weight * ssim_loss(img1, pretraining_measurement_scalar * img2)

    return loss.type(dtype)


def relative_l1_loss(img1, img2, pretraining_measurement_scalar, relative_mse_eps):
    return (torch.abs(img1 - img2) / (torch.abs(img1.detach()) + relative_mse_eps)).mean()


def relative_l2_loss(y_pred, y_true, val):
    return torch.mean((y_pred - y_true)**2 / (y_pred.detach()**2 + val))


def fourier_loss(F1, F2):
    projection = torch.abs(F1 * torch.conj(F2)) / torch.abs(F1) / torch.abs(F2)
    
    return 1.0 - projection.mean()


def hessian_loss_v2(im, dx, dz, tv_eps=1e-6, ab_eps=1e-6, ab_weight=1.):
    dxm = torch.roll(im, -1, dims=2); dxp = torch.roll(im, 1, dims=2)
    dym = torch.roll(im, -1, dims=1); dyp = torch.roll(im, 1, dims=1)
    dzm = torch.roll(im, -1, dims=0); dzp = torch.roll(im, 1, dims=0)
    dxdy = torch.roll(torch.roll(im, -1, dims=1), -1, dims=2)
    dydz = torch.roll(torch.roll(im, -1, dims=0), -1, dims=1)
    dzdx = torch.roll(torch.roll(im, -1, dims=2), -1, dims=0)
    
    l1 = (dxm - 2*im + dxp)
    l2 = (dym - 2*im + dyp)
    l3 = (dzm - 2*im + dzp) * ((dx / dz)**2)
    # l3 = dzm - 2*im + dzp
    l4 = (2**0.5) * (im - dxm - dym + dxdy)
    l5 = (2**0.5) * (im - dym - dzm + dydz) * (dx / dz)
    l6 = (2**0.5) * (im - dzm - dxm + dzdx) * (dx / dz)   
    # l5 = (2**0.5) * (im - dym - dzm + dydz)
    # l6 = (2**0.5) * (im - dzm - dxm + dzdx) 

    ab = torch.abs(im)
    tv = torch.abs(l1) + torch.abs(l2) + torch.abs(l3) + torch.abs(l4) + torch.abs(l5) + torch.abs(l6)
    
    return ab_weight * (ab / (ab.detach() + ab_eps)).mean() + (tv / (tv.detach() + tv_eps)).mean()


def exclusion_loss(im1, im2, level=3):    
    psi = []
    
    for ds in range(level):
        _im1 = nn.functional.interpolate(im1.unsqueeze(0).unsqueeze(0), scale_factor = 1 / (2**ds)).squeeze(0).squeeze(0)
        _im2 = nn.functional.interpolate(im2.unsqueeze(0).unsqueeze(0), scale_factor = 1 / (2**ds)).squeeze(0).squeeze(0)
        
        _grad1 = torch.sqrt(torch.square(torch.roll(_im1, 1, 0) - _im1) + torch.square(torch.roll(_im1, 1, 1) - _im1))
        _grad1 = torch.clamp(_grad1 / torch.sqrt(torch.sum(_grad1 ** 2)), min = 1e-3, max = 1e3)
        _grad2 = torch.sqrt(torch.square(torch.roll(_im2, 1, 0) - _im2) + torch.square(torch.roll(_im2, 1, 1) - _im2))
        _grad2 = torch.clamp(_grad2 / torch.sqrt(torch.sum(_grad2 ** 2)), min = 1e-3, max = 1e3)
       
        _psi = torch.tanh(_grad1) * torch.tanh(_grad2)
        _psi = torch.sqrt(torch.sum(_psi ** 2))

        psi.append(_psi)

    return torch.stack(psi).sum()