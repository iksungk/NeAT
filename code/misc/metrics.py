import torch
import torch.nn.functional as F
from math import exp
import numpy as np
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from scipy.stats import skew, kurtosis, pearsonr, wasserstein_distance
from scipy.signal.windows import tukey
from skimage.metrics import structural_similarity

    
def figure_of_merit(out_y, y, max_iter=100, tol=1e-3):
    r = out_y.std()/y.std() # SNR low, r low

    classif = GaussianMixture(n_components=2, max_iter=max_iter, tol=tol)
    classif.fit(out_y.reshape((out_y.size, 1)))
    threshold = np.mean(classif.means_)
    out_y_s = np.ones_like(out_y)
    out_y_s[out_y < threshold] = 0

    dd = np.mean(out_y_s, 0).mean() * np.mean(out_y_s, 1).mean() * np.mean(out_y_s, 2).mean() # normalization factor for density

    lx = np.roll(out_y, 1, axis=0) - out_y
    ly = np.roll(out_y, 1, axis=1) - out_y
    lz = (np.roll(out_y, 1, axis=2) - out_y) * (args.psf_dx/args.psf_dz)
    grad = np.sqrt(lx**2 + ly**2 + lz**2)
    
    return grad.std()/out_y.mean() * r * np.power(dd, 1/3)
    

def interpolated_intercepts(x, y1, y2):
    def intercept(point1, point2, point3, point4):
        def line(p1, p2):
            A = (p1[1] - p2[1])
            B = (p2[0] - p1[0])
            C = (p1[0]*p2[1] - p2[0]*p1[1])
            return A, B, -C

        def intersection(L1, L2):
            D  = L1[0] * L2[1] - L1[1] * L2[0]
            Dx = L1[2] * L2[1] - L1[1] * L2[2]
            Dy = L1[0] * L2[2] - L1[2] * L2[0]

            x = Dx / D
            y = Dy / D
            return x,y

        L1 = line([point1[0],point1[1]], [point2[0],point2[1]])
        L2 = line([point3[0],point3[1]], [point4[0],point4[1]])

        R = intersection(L1, L2)

        return R

    idxs = np.argwhere(np.diff(np.sign(y1 - y2)) != 0)

    xcs = []
    ycs = []

    for idx in idxs:
        xc, yc = intercept((x[idx], y1[idx]),((x[idx+1], y1[idx+1])), ((x[idx], y2[idx])), ((x[idx+1], y2[idx+1])))
        xcs.append(xc)
        ycs.append(yc)
    return np.array(xcs), np.array(ycs)

    
def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int32)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    rad_prof = tbin / nr
    
    return rad_prof 


def skewness_fourier_domain(args, input_stack, input_stack_dn, alpha, plot_figures=False):
    win = tukey(args.dims[2]*2, alpha = alpha)[:, np.newaxis]
    win = np.einsum('ib,bj->ij', win, tukey(args.dims[1]*2, alpha = alpha)[np.newaxis, :])
    win = np.einsum('bjk,ib->ijk', win[np.newaxis, :, :], tukey(args.dims[0]*2, alpha = alpha)[:, np.newaxis])

    y_fft = np.abs(np.fft.fftshift(np.fft.fftn(input_stack * win)))**2
    y_fft_cnt = np.log10(y_fft[args.dims[0]])
    rad_xy = radial_profile(y_fft_cnt, [args.dims[1], args.dims[2]])[:-1]
    
    y_fft_dn = np.abs(np.fft.fftshift(np.fft.fftn(input_stack_dn * win)))**2
    y_fft_cnt_dn = np.log10(y_fft_dn[args.dims[0]])
    rad_xy_dn = radial_profile(y_fft_cnt_dn, [args.dims[1], args.dims[2]])[:-1]
    
    p_xy = np.max([pearsonr(y_fft_cnt.flatten(), y_fft_cnt_dn.flatten())[0], 
                   structural_similarity(y_fft_cnt, y_fft_cnt_dn)])
    # p_xy = np.mean(np.square(y_fft_cnt - y_fft_cnt_dn))
    # p_xy = structural_similarity(y_fft_cnt, y_fft_cnt_dn)
    
    # idx = np.argwhere(np.diff(np.sign(rad.max()/2 - rad))).flatten()
    # dist = idx[-1] - idx[0]
    # sp_freq = dist/args.dims[1]*0.5
    # # print(sp_freq)
    
    if plot_figures:
        plt.plot(rad_xy)
        plt.plot(rad_xy_dn)
        plt.axhline(rad_xy.max()/2)
        plt.show()
        
    y_fft_cnt = np.log10(y_fft[:, args.dims[1], args.dims[2]-args.dims[0]:args.dims[2]+args.dims[0]])
    rad_yz = radial_profile(y_fft_cnt, [args.dims[0], args.dims[0]])[:-1]
    
    y_fft_cnt_dn = np.log10(y_fft_dn[:, args.dims[1], args.dims[2]-args.dims[0]:args.dims[2]+args.dims[0]])
    rad_yz_dn = radial_profile(y_fft_cnt_dn, [args.dims[0], args.dims[0]])[:-1]
    
    p_yz = np.max([pearsonr(y_fft_cnt.flatten(), y_fft_cnt_dn.flatten())[0], 
                   structural_similarity(y_fft_cnt, y_fft_cnt_dn)])
    # p_yz = np.mean(np.square(y_fft_cnt - y_fft_cnt_dn))
    # p_yz = structural_similarity(y_fft_cnt, y_fft_cnt_dn)
    
    # idx = np.argwhere(np.diff(np.sign(rad.max()/2 - rad))).flatten()
    # dist = idx[-1] - idx[0]
    # sp_freq = dist/args.dims[1]*0.5
    # # print(sp_freq)
    
    if plot_figures:
        plt.plot(rad_yz)
        plt.plot(rad_yz_dn)
        plt.axhline(rad.max()/2)
        plt.show()
        
    y_fft_cnt = np.log10(y_fft[:, args.dims[1]-args.dims[0]:args.dims[1]+args.dims[0], args.dims[2]])
    rad_zx = radial_profile(y_fft_cnt, [args.dims[0], args.dims[0]])[:-1]
    
    y_fft_cnt_dn = np.log10(y_fft[:, args.dims[1]-args.dims[0]:args.dims[1]+args.dims[0], args.dims[2]])
    rad_zx_dn = radial_profile(y_fft_cnt_dn, [args.dims[0], args.dims[0]])[:-1]
    
    p_zx = np.max([pearsonr(y_fft_cnt.flatten(), y_fft_cnt_dn.flatten())[0], 
                   structural_similarity(y_fft_cnt, y_fft_cnt_dn)])
    # p_zx = np.mean(np.square(y_fft_cnt - y_fft_cnt_dn))
    # p_zx = structural_similarity(y_fft_cnt, y_fft_cnt_dn)
    
    # idx = np.argwhere(np.diff(np.sign(rad.max()/2 - rad))).flatten()
    # dist = idx[-1] - idx[0]
    # sp_freq = dist/args.dims[1]*0.5
    # # print(sp_freq)
    
    if plot_figures:
        plt.plot(rad_zx)
        plt.plot(rad_zx_dn)
        plt.axhline(rad_zx.max()/2)
        plt.show()

    return skew(rad_xy), skew(rad_yz), skew(rad_zx), skew(rad_xy_dn), skew(rad_yz_dn), skew(rad_zx_dn), p_xy, p_yz, p_zx


def kurtosis_fourier_domain(args, input_stack, input_stack_dn, alpha, plot_figures=False, fisher=False, return_fom=False):
    win = tukey(args.dims[2]*2, alpha = alpha)[:, np.newaxis]
    win = np.einsum('ib,bj->ij', win, tukey(args.dims[1]*2, alpha = alpha)[np.newaxis, :])
    win = np.einsum('bjk,ib->ijk', win[np.newaxis, :, :], tukey(args.dims[0]*2, alpha = alpha)[:, np.newaxis])

    y_fft = np.abs(np.fft.fftshift(np.fft.fftn(input_stack * win)))**2
    y_fft_cnt = np.log10(y_fft[args.dims[0]])
    rad = radial_profile(y_fft_cnt, [args.dims[1], args.dims[2]])
    rad_xy = np.concatenate((np.flip(rad), rad[1:]), axis=0)[1:-1]
    
    y_fft_dn = np.abs(np.fft.fftshift(np.fft.fftn(input_stack_dn * win)))**2
    y_fft_cnt_dn = np.log10(y_fft_dn[args.dims[0]])
    rad_dn = radial_profile(y_fft_cnt_dn, [args.dims[1], args.dims[2]])
    rad_xy_dn = np.concatenate((np.flip(rad_dn), rad_dn[1:]), axis=0)[1:-1]
    
    
#     if metric_option == 'pcc_ssim':
#         p_xy = np.max([pearsonr(y_fft_cnt.flatten(), y_fft_cnt_dn.flatten())[0], 
#                        structural_similarity(y_fft_cnt, y_fft_cnt_dn)])
#     elif metric_option == 'pcc':
#         p_xy = pearsonr(y_fft_cnt.flatten(), y_fft_cnt_dn.flatten())
    
#     elif metric_option == 'ssim':
#         p_xy = structural_similarity(y_fft_cnt, y_fft_cnt_dn)
        
#     elif metric_option == 'emd':
#         p_xy = wasserstein_distance(rad_xy_dn, rad_xy)
    
    
    xcs, ycs = interpolated_intercepts(np.arange((-rad_xy.size+1)/2, (rad_xy.size+1)/2, 1), rad_xy, np.full_like(rad_xy, rad_xy.max()/2))

    sp_freq_xy = (xcs[-1] - xcs[0])/100*0.5
    
    xcs, ycs = interpolated_intercepts(np.arange((-rad_xy_dn.size+1)/2, (rad_xy_dn.size+1)/2, 1), rad_xy_dn, np.full_like(rad_xy_dn, rad_xy_dn.max()/2))

    sp_freq_xy_dn = (xcs[-1] - xcs[0])/100*0.5
    
    
    if plot_figures:
        plt.plot(rad_xy)
        plt.plot(rad_xy_dn)
        plt.axhline(rad_xy.max()/2)
        plt.show()
        
    y_fft_cnt = np.log10(y_fft[:, args.dims[1], args.dims[2]-args.dims[0]:args.dims[2]+args.dims[0]])
    rad = radial_profile(y_fft_cnt, [args.dims[0], args.dims[0]])
    rad_yz = np.concatenate((np.flip(rad), rad[1:]), axis=0)[1:-1]
    
    y_fft_cnt_dn = np.log10(y_fft_dn[:, args.dims[1], args.dims[2]-args.dims[0]:args.dims[2]+args.dims[0]])
    rad_dn = radial_profile(y_fft_cnt_dn, [args.dims[0], args.dims[0]])
    rad_yz_dn = np.concatenate((np.flip(rad_dn), rad_dn[1:]), axis=0)[1:-1]
    
    # p_yz = np.mean(np.square(rad_yz - rad_yz_dn))
    
#     if metric_option == 'pcc_ssim':
#         p_yz = np.max([pearsonr(y_fft_cnt.flatten(), y_fft_cnt_dn.flatten())[0], 
#                        structural_similarity(y_fft_cnt, y_fft_cnt_dn)])
#     elif metric_option == 'pcc':
#         p_yz = pearsonr(y_fft_cnt.flatten(), y_fft_cnt_dn.flatten())
    
#     elif metric_option == 'ssim':
#         p_yz = structural_similarity(y_fft_cnt, y_fft_cnt_dn)
        
#     elif metric_option == 'emd':
#         p_yz = wasserstein_distance(rad_yz_dn, rad_yz)
        
    xcs, ycs = interpolated_intercepts(np.arange((-rad_yz.size+1)/2, (rad_yz.size+1)/2, 1), rad_yz, np.full_like(rad_yz, rad_yz.max()/2))

    sp_freq_yz = (xcs[-1] - xcs[0])/100*0.5
    
    xcs, ycs = interpolated_intercepts(np.arange((-rad_yz_dn.size+1)/2, (rad_yz_dn.size+1)/2, 1), rad_yz_dn, np.full_like(rad_yz_dn, rad_yz_dn.max()/2))

    sp_freq_yz_dn = (xcs[-1] - xcs[0])/100*0.5
    
    
    if plot_figures:
        plt.plot(rad_yz)
        plt.plot(rad_yz_dn)
        plt.axhline(rad_yz.max()/2)
        plt.show()
        
    y_fft_cnt = np.log10(y_fft[:, args.dims[1]-args.dims[0]:args.dims[1]+args.dims[0], args.dims[2]])
    rad = radial_profile(y_fft_cnt, [args.dims[0], args.dims[0]])
    rad_zx = np.concatenate((np.flip(rad), rad[1:]), axis=0)[1:-1]
    
    y_fft_cnt_dn = np.log10(y_fft_dn[:, args.dims[1]-args.dims[0]:args.dims[1]+args.dims[0], args.dims[2]])
    rad_dn = radial_profile(y_fft_cnt_dn, [args.dims[0], args.dims[0]])
    rad_zx_dn = np.concatenate((np.flip(rad_dn), rad_dn[1:]), axis=0)[1:-1]
    
    # p_zx = np.mean(np.square(rad_zx - rad_zx_dn))
    
#     if metric_option == 'pcc_ssim':
#         p_zx = np.max([pearsonr(y_fft_cnt.flatten(), y_fft_cnt_dn.flatten())[0], 
#                        structural_similarity(y_fft_cnt, y_fft_cnt_dn)])
#     elif metric_option == 'pcc':
#         p_zx = pearsonr(y_fft_cnt.flatten(), y_fft_cnt_dn.flatten())
    
#     elif metric_option == 'ssim':
#         p_zx = structural_similarity(y_fft_cnt, y_fft_cnt_dn)
        
#     elif metric_option == 'emd':
#         p_zx = wasserstein_distance(rad_zx_dn, rad_zx)
    
    xcs, ycs = interpolated_intercepts(np.arange((-rad_zx.size+1)/2, (rad_zx.size+1)/2, 1), rad_zx, np.full_like(rad_zx, rad_zx.max()/2))

    sp_freq_zx = (xcs[-1] - xcs[0])/100*0.5
    
    xcs, ycs = interpolated_intercepts(np.arange((-rad_zx_dn.size+1)/2, (rad_zx_dn.size+1)/2, 1), rad_zx_dn, np.full_like(rad_zx_dn, rad_zx_dn.max()/2))

    sp_freq_zx_dn = (xcs[-1] - xcs[0])/100*0.5
    

    if plot_figures:
        plt.plot(rad_zx)
        plt.plot(rad_zx_dn)
        plt.axhline(rad_zx.max()/2)
        plt.show()

        
    if return_fom:
        ns = (np.quantile(rad_xy, 0.1) + np.quantile(rad_yz, 0.1) + np.quantile(rad_zx, 0.1))/3
        fom = np.power((kurtosis(rad_xy, fisher=fisher) + kurtosis(rad_yz, fisher=fisher) + kurtosis(rad_zx, fisher=fisher))/3, 1/3) + 3*np.power(ns, -1/5)
        
        grad_z = np.roll(input_stack_dn, 1, axis=0) - input_stack_dn
        grad_y = np.roll(input_stack_dn, 1, axis=1) - input_stack_dn
        grad_x = np.roll(input_stack_dn, 1, axis=2) - input_stack_dn
        grad = np.sqrt(grad_z**2 + grad_y**2 + grad_x**2)

        fom += 0.2*np.power(grad.mean() / input_stack_dn.mean(), -1/6)
        
        # return np.power((kurtosis(rad_xy, fisher=fisher) + kurtosis(rad_yz, fisher=fisher) + kurtosis(rad_zx, fisher=fisher))/4, 1/3) + np.power(ns, -0.5)
        # return 10 * (np.power((kurtosis(rad_xy, fisher=fisher) + kurtosis(rad_yz, fisher=fisher) + kurtosis(rad_zx, fisher=fisher))/3, 1/3) + 3*np.power(ns, -1/4) - 4)
        return fom
    
    else:
        return [kurtosis(rad_xy, fisher=fisher), kurtosis(rad_yz, fisher=fisher), kurtosis(rad_zx, fisher=fisher)], [kurtosis(rad_xy_dn, fisher=fisher), kurtosis(rad_yz_dn, fisher=fisher), kurtosis(rad_zx_dn, fisher=fisher)], [sp_freq_xy, sp_freq_xy_dn, sp_freq_yz, sp_freq_yz_dn, sp_freq_zx, sp_freq_zx_dn], [rad_xy, rad_yz, rad_zx]



def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def create_3d_window(window_size, channel = 1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(-1)
    _3D_window = (_2D_window * _1D_window.t().unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    window = _3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, depth, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, depth, height, width)
        window = create_3d_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv3d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv3d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = v1 / v2  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        cs = cs.mean()
        ret = ssim_map.mean()
    else:
        cs = cs.mean(1).mean(1).mean(1)
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=None):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    ssims = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)

        # Relu normalize (not compliant with original definition)
        if normalize == "relu":
            ssims.append(torch.relu(sim))
            mcs.append(torch.relu(cs))
        else:
            ssims.append(sim)
            mcs.append(cs)

        img1 = F.avg_pool3d(img1, (2, 2, 2))
        img2 = F.avg_pool3d(img2, (2, 2, 2))

    ssims = torch.stack(ssims)
    mcs = torch.stack(mcs)

    # Simple normalize (not compliant with original definition)
    # TODO: remove support for normalize == True (kept for backward support)
    if normalize == "simple" or normalize == True:
        ssims = (ssims + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = ssims ** weights

    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1]) * pow2[-1]
    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)

    
