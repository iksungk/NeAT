import torch
import torch.nn.functional as F
import numpy as np
import scipy.ndimage as ndimage
import pandas
import math
import scipy

from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import jensenshannon
from sklearn.mixture import GaussianMixture
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift
from scipy.stats import pearsonr
from torch.fft import fftn, ifftn, fftshift, ifftshift, rfftn, irfftn, rfft
from skimage.restoration import denoise_nl_means


def next_power_of_2(x):
    return 1 if x == 0 else 2**math.ceil(math.log2(x))


# Converting torch tensors to numpy arrays.
def torch_to_np(x):
    return x.detach().cpu().numpy()


# Custom defined colormap.
def custom_div_cmap(numcolors=11, name='custom_div_cmap', mincol='black', midcol = 'k', maxcol='green'):
    from matplotlib.colors import LinearSegmentedColormap 
    
    cmap = LinearSegmentedColormap.from_list(name=name, 
                                             colors =[mincol, midcol, maxcol],
                                             N=numcolors)
    return cmap


# Imposing a constraint on a Zernike mode to keep its coefficient close to zero.
def single_mode_control(wf, num, vmin, vmax):
    return F.relu(wf[num] - vmax) + F.relu(-wf[num] + vmin)


# 3D Hann window.
def apply_window(im):
    im *= torch.hann_window(im.shape[0]).view(im.shape[0], 1, 1).type(dtype).cuda(0)
    im *= torch.hann_window(im.shape[1]).view(1, im.shape[1], 1).type(dtype).cuda(0)
    im *= torch.hann_window(im.shape[2]).view(1, 1, im.shape[2]).type(dtype).cuda(0)
    
    return im
    

# Convolution with FFT.
def fft_convolve(im1, im2, mode='fftn'):
    if mode == 'fftn':
        _im1 = fftn(im1)
        _im2 = fftn(im2)
        _im = _im1 * _im2
        
        return fftshift(torch.real(ifftn(_im)))
    
    elif mode == 'rfftn':
        _im1 = rfftn(im1)
        _im2 = rfftn(im2)
        _im = _im1 * _im2

        return fftshift(torch.real(irfftn(_im)))
    

def apply_poisson_noise(img, max_flux):
    img = torch.poisson(img * max_flux)
    img = img / max_flux
    
    return img


def _matrix_pow(matrix: torch.Tensor, p: float) -> torch.Tensor:
    vals, vecs = torch.linalg.eig(matrix)
    vals = torch.real(vals)
    vecs = torch.real(vecs)
    vals_pow = vals.pow(p)
    matrix_pow = torch.matmul(vecs, torch.matmul(torch.diag(vals_pow), torch.inverse(vecs)))
    return matrix_pow


def param_to_str(num):
    a = int(np.ceil(np.abs(np.log10(num))))
    b = num * np.power(10, a)
    
    if b.is_integer():
        b = int(b)
    else:
        b = np.round(b, 2)
        
    return str(b) + 'em' + str(a)


def tukeywin(L, r = 0.03):
    win = torch.ones(L, )

    win[0 : int(r/2 * L)] = 0.5 * (1 + np.cos(2 * np.pi / r * (torch.arange(0, int(r/2 * L))/(L-1) - r/2)))
    win[L - int(r/2 * L) : L] = 0.5 * (1 + np.cos(2 * np.pi / r * (torch.arange(L - int(r/2 * L), L)/(L-1) - 1 + r/2)))
    
    return win.cuda(0).type(dtype)


def compute_pcc(img1, img2, dim1, dim2, correction = True):
    mip_ref = np.max(img1, 0)
    mip_exp = np.max(img2, 0)
    
    pcc = pearsonr(np.abs(mip_ref).flatten(), np.abs(mip_exp).flatten())[0]
    
    if correction:
        shift, error, diffphase = phase_cross_correlation(mip_ref, mip_exp, upsample_factor=100, overlap_ratio = 0.9, normalization = None)
        mip_exp = fourier_shift(np.fft.fftn(mip_exp), shift)
        mip_exp = np.fft.ifftn(mip_exp).real

    pcc_shift = pearsonr(np.abs(mip_ref[dim1:dim2, dim1:dim2]).flatten(), 
                         np.abs(mip_exp[dim1:dim2, dim1:dim2]).flatten())[0]

    return np.maximum(pcc, pcc_shift)


def compute_ssim(img1, img2, dim1, dim2, correction = True):
    mip_ref = np.max(img1, 0)
    mip_exp = np.max(img2, 0)
    
    val = ssim(mip_ref, mip_exp, data_range = 1)
    
    if correction:
        shift, error, diffphase = phase_cross_correlation(mip_ref, mip_exp, upsample_factor=100, overlap_ratio = 0.9, normalization = None)
        mip_exp = fourier_shift(np.fft.fftn(mip_exp), shift)
        mip_exp = np.fft.ifftn(mip_exp).real

    val_shift = ssim(mip_ref[dim1:dim2, dim1:dim2], 
                     mip_exp[dim1:dim2, dim1:dim2], data_range=mip_ref.max() - mip_ref.min())

    return np.maximum(val, val_shift)
    

def signal_to_background_ratio_gaussian_mixture_nlm_v2(ref, tar, l=400, n_lp=40, 
                                                       sigma_est=45, patch_kw = None, return_imgs=False):
    # patch_kw = dict(patch_size=11,      
    #                 patch_distance=12, 
    #                 channel_axis=-1)

    im = np.max(ref, 0)

    im_lp = ndimage.gaussian_filter(im, sigma=l/(4.*n_lp))
    im_corr = im - im_lp
    im_corr = denoise_nl_means(np.expand_dims(im_corr, -1), h=0.6 * sigma_est, sigma=sigma_est,
                               fast_mode=True, **patch_kw)

    classif = GaussianMixture(n_components=2, tol=1e-4, reg_covar=1e-6, max_iter=500, n_init=5)
    classif.fit(im_corr.reshape((im_corr.size, 1)))
    threshold = np.mean(classif.means_)

    im_b = np.copy(im)
    im_b[im_corr > threshold] = 0

    im_aber = np.max(tar, 0)
    sbr = np.median(im_aber[im_b == 0]) / np.median(im_aber[im_b > 0])
    
    im_aber_b = np.copy(im_aber)
    im_aber_b[im_b == 0] = 0
    im_aber_s = im_aber - im_aber_b
    
    if return_imgs:
        # return sbr, im_b, im_s, im_corr, im_lp
        return sbr, im_aber_b, im_aber_s, im_b, im_corr

    else:
        return sbr
 

def bck_tv_std_compute(args, im, idx, include_im_sq=False):
    dx = args.psf_dx
    dz = args.psf_dz
    
    dxm = np.roll(im, -1, axis=2); dxp = np.roll(im, 1, axis=2)
    dym = np.roll(im, -1, axis=1); dyp = np.roll(im, 1, axis=1)
    dzm = np.roll(im, -1, axis=0); dzp = np.roll(im, 1, axis=0)
    dxdy = np.roll(np.roll(im, -1, axis=1), -1, axis=2)
    dydz = np.roll(np.roll(im, -1, axis=0), -1, axis=1)
    dzdx = np.roll(np.roll(im, -1, axis=2), -1, axis=0)
    
    l1 = (dxm - 2*im + dxp)
    l2 = (dym - 2*im + dyp)
    l3 = (dzm - 2*im + dzp) * ((dx / dz)**2)
    l4 = (2**0.5) * (im - dxm - dym + dxdy)
    l5 = (2**0.5) * (im - dym - dzm + dydz) * (dx / dz)
    l6 = (2**0.5) * (im - dzm - dxm + dzdx) * (dx / dz)

    if include_im_sq:
        return np.min([(im**2)[idx].std(),
                       (l1**2)[idx].std(), 
                       (l2**2)[idx].std(), 
                       (l3**2)[idx].std(), 
                       (l4**2)[idx].std(), 
                       (l5**2)[idx].std(), 
                       (l6**2)[idx].std()])
        
    else:
        return np.min([np.abs(l1)[idx].std(), 
                       np.abs(l2)[idx].std(), 
                       np.abs(l3)[idx].std(), 
                       np.abs(l4)[idx].std(), 
                       np.abs(l5)[idx].std(), 
                       np.abs(l6)[idx].std()])


def estimate_structure_max_value(args, d, h, w, feature_sz):
    pt = np.zeros((d, h, w))
    pt[d//2, h//2, w//2] = 1
    
    sz1 = feature_sz/2
    sz2 = args.excitation_wavelength / args.na_exc * 0.4 / args.psf_dx / 2
    
    im = scipy.ndimage.gaussian_filter(pt, sz1); im /= im.max()
    k = scipy.ndimage.gaussian_filter(pt, sz2); k /= k.sum()
    im_k = np.real(fft_convolve(torch.from_numpy(im), torch.from_numpy(k)))

    return 1/im_k.max()
    

def wf_convention_from_ml_to_dm_thorscope(wf_ml):
    assert wf_ml.size == 12
    
    wf_dm = np.zeros_like(wf_ml)
    
    wf_dm[0] = -wf_ml[0]
    wf_dm[1] = wf_ml[1]
    wf_dm[2] = -wf_ml[2]
    
    wf_dm[3] = wf_ml[3]
    wf_dm[4] = wf_ml[4]
    wf_dm[5] = wf_ml[5]
    wf_dm[6] = wf_ml[6]
    
    wf_dm[7] = -wf_ml[7]   
    wf_dm[8] = -wf_ml[8]
    wf_dm[9] = -wf_ml[9]
    wf_dm[10] = -wf_ml[10]
    wf_dm[11] = -wf_ml[11]
    
    return wf_dm


def estimated_to_given_conversion_ml_to_dmd(ml_wf, order_up_to = 5):
    dmd_wf = np.zeros(ml_wf.shape)

    dmd_wf[0] = -ml_wf[0]
    dmd_wf[1] = ml_wf[1]
    dmd_wf[2] = -ml_wf[2]

    dmd_wf[3] = -ml_wf[6]
    dmd_wf[4] = ml_wf[5]
    dmd_wf[5] = -ml_wf[4]
    dmd_wf[6] = ml_wf[3]

    dmd_wf[7] = ml_wf[7]
    dmd_wf[8] = -ml_wf[8]
    dmd_wf[9] = ml_wf[9]
    dmd_wf[10] = -ml_wf[10]
    dmd_wf[11] = ml_wf[11]
    
    if order_up_to > 4:
        dmd_wf[12] = ml_wf[17]
        dmd_wf[13] = -ml_wf[16]
        dmd_wf[14] = ml_wf[15]
        dmd_wf[15] = -ml_wf[14]
        dmd_wf[16] = ml_wf[13]
        dmd_wf[17] = -ml_wf[12]
        
    if order_up_to > 5:
        dmd_wf[18] = -ml_wf[18]
        dmd_wf[19] = ml_wf[19]
        dmd_wf[20] = -ml_wf[20]
        dmd_wf[21] = ml_wf[21]
        dmd_wf[22] = -ml_wf[22]
        dmd_wf[23] = ml_wf[23]
        dmd_wf[24] = -ml_wf[24]
    
    if order_up_to > 6:
        dmd_wf[25] = -ml_wf[32]
        dmd_wf[26] = ml_wf[31]
        dmd_wf[27] = -ml_wf[30]
        dmd_wf[28] = ml_wf[29]
        dmd_wf[29] = -ml_wf[28]
        dmd_wf[30] = ml_wf[27]
        dmd_wf[31] = -ml_wf[26]
        dmd_wf[32] = ml_wf[25]
        
    if order_up_to > 7:
        dmd_wf[33] = ml_wf[33]
        dmd_wf[34] = -ml_wf[34]
        dmd_wf[35] = ml_wf[35]
        dmd_wf[36] = -ml_wf[36]
        dmd_wf[37] = ml_wf[37]
        dmd_wf[38] = -ml_wf[38]
        dmd_wf[39] = ml_wf[39]
        dmd_wf[40] = -ml_wf[40]
        dmd_wf[41] = ml_wf[41]
        
    if order_up_to > 8:
        dmd_wf[42] = ml_wf[51]
        dmd_wf[43] = -ml_wf[50]
        dmd_wf[44] = ml_wf[49]
        dmd_wf[45] = -ml_wf[48]
        dmd_wf[46] = ml_wf[47]
        dmd_wf[47] = -ml_wf[46]
        dmd_wf[48] = ml_wf[45]
        dmd_wf[49] = -ml_wf[44]
        dmd_wf[50] = ml_wf[43]
        dmd_wf[51] = -ml_wf[42]
        
    return dmd_wf


def estimated_to_given_conversion_dmd_to_ml(dmd_wf, order_up_to = 4):
    ml_wf = np.zeros(dmd_wf.shape)

    ml_wf[0] = -dmd_wf[0]
    ml_wf[1] = dmd_wf[1]
    ml_wf[2] = -dmd_wf[2]

    ml_wf[3] = dmd_wf[6]
    ml_wf[4] = -dmd_wf[5]
    ml_wf[5] = dmd_wf[4]
    ml_wf[6] = -dmd_wf[3]

    ml_wf[7] = dmd_wf[7]
    ml_wf[8] = -dmd_wf[8]
    ml_wf[9] = dmd_wf[9]
    ml_wf[10] = -dmd_wf[10]
    ml_wf[11] = dmd_wf[11]
    
    if order_up_to > 4:
        ml_wf[12] = -dmd_wf[17]
        ml_wf[13] = dmd_wf[16]
        ml_wf[14] = -dmd_wf[15]
        ml_wf[15] = dmd_wf[14]
        ml_wf[16] = -dmd_wf[13]
        ml_wf[17] = dmd_wf[12]
    
    return ml_wf


def find_translation_and_fix(ref, moving, mode='mean'):
    if mode == 'max':
        shift_xy = phase_cross_correlation(ref.max(0), moving.max(0), upsample_factor=1000, normalization=None); 
    elif mode == 'mean':
        shift_xy = phase_cross_correlation(ref.mean(0), moving.mean(0), upsample_factor=1000, normalization=None); 
    # print(shift_xy[0])
    moving_mv = fourier_shift(np.fft.fftn(moving), (0, shift_xy[0][0], shift_xy[0][1]))
    moving_mv = np.fft.ifftn(moving_mv).real

    if mode == 'max':
        shift_xz = phase_cross_correlation(ref.max(1), moving_mv.max(1), upsample_factor=1000, normalization=None); 
    elif mode == 'mean':
        shift_xz = phase_cross_correlation(ref.mean(1), moving_mv.mean(1), upsample_factor=1000, normalization=None); 
    # print(shift_xz[0])
    moving_mv = fourier_shift(np.fft.fftn(moving_mv), (shift_xz[0][0], 0, shift_xz[0][1]))
    moving_mv = np.fft.ifftn(moving_mv).real

    if mode == 'max':
        shift_yz = phase_cross_correlation(ref.max(2), moving_mv.max(2), upsample_factor=1000, normalization=None); 
    elif mode == 'mean':
        shift_yz = phase_cross_correlation(ref.mean(2), moving_mv.mean(2), upsample_factor=1000, normalization=None);         
    # print(shift_yz[0])
    moving_mv = fourier_shift(np.fft.fftn(moving_mv), (shift_yz[0][0], shift_yz[0][1], 0))
    moving_mv = np.fft.ifftn(moving_mv).real
    
    return moving_mv, shift_xy, shift_xz, shift_yz


cm_data = [[0.2422, 0.1504, 0.6603],
           [0.2444, 0.1534, 0.6728],
           [0.2464, 0.1569, 0.6847],
           [0.2484, 0.1607, 0.6961],
           [0.2503, 0.1648, 0.7071],
           [0.2522, 0.1689, 0.7179],
           [0.254, 0.1732, 0.7286],
           [0.2558, 0.1773, 0.7393],
           [0.2576, 0.1814, 0.7501],
           [0.2594, 0.1854, 0.761],
           [0.2611, 0.1893, 0.7719],
           [0.2628, 0.1932, 0.7828],
           [0.2645, 0.1972, 0.7937],
           [0.2661, 0.2011, 0.8043],
           [0.2676, 0.2052, 0.8148],
           [0.2691, 0.2094, 0.8249],
           [0.2704, 0.2138, 0.8346],
           [0.2717, 0.2184, 0.8439],
           [0.2729, 0.2231, 0.8528],
           [0.274, 0.228, 0.8612],
           [0.2749, 0.233, 0.8692],
           [0.2758, 0.2382, 0.8767],
           [0.2766, 0.2435, 0.884],
           [0.2774, 0.2489, 0.8908],
           [0.2781, 0.2543, 0.8973],
           [0.2788, 0.2598, 0.9035],
           [0.2794, 0.2653, 0.9094],
           [0.2798, 0.2708, 0.915],
           [0.2802, 0.2764, 0.9204],
           [0.2806, 0.2819, 0.9255],
           [0.2809, 0.2875, 0.9305],
           [0.2811, 0.293, 0.9352],
           [0.2813, 0.2985, 0.9397],
           [0.2814, 0.304, 0.9441],
           [0.2814, 0.3095, 0.9483],
           [0.2813, 0.315, 0.9524],
           [0.2811, 0.3204, 0.9563],
           [0.2809, 0.3259, 0.96],
           [0.2807, 0.3313, 0.9636],
           [0.2803, 0.3367, 0.967],
           [0.2798, 0.3421, 0.9702],
           [0.2791, 0.3475, 0.9733],
           [0.2784, 0.3529, 0.9763],
           [0.2776, 0.3583, 0.9791],
           [0.2766, 0.3638, 0.9817],
           [0.2754, 0.3693, 0.984],
           [0.2741, 0.3748, 0.9862],
           [0.2726, 0.3804, 0.9881],
           [0.271, 0.386, 0.9898],
           [0.2691, 0.3916, 0.9912],
           [0.267, 0.3973, 0.9924],
           [0.2647, 0.403, 0.9935],
           [0.2621, 0.4088, 0.9946],
           [0.2591, 0.4145, 0.9955],
           [0.2556, 0.4203, 0.9965],
           [0.2517, 0.4261, 0.9974],
           [0.2473, 0.4319, 0.9983],
           [0.2424, 0.4378, 0.9991],
           [0.2369, 0.4437, 0.9996],
           [0.2311, 0.4497, 0.9995],
           [0.225, 0.4559, 0.9985],
           [0.2189, 0.462, 0.9968],
           [0.2128, 0.4682, 0.9948],
           [0.2066, 0.4743, 0.9926],
           [0.2006, 0.4803, 0.9906],
           [0.195, 0.4861, 0.9887],
           [0.1903, 0.4919, 0.9867],
           [0.1869, 0.4975, 0.9844],
           [0.1847, 0.503, 0.9819],
           [0.1831, 0.5084, 0.9793],
           [0.1818, 0.5138, 0.9766],
           [0.1806, 0.5191, 0.9738],
           [0.1795, 0.5244, 0.9709],
           [0.1785, 0.5296, 0.9677],
           [0.1778, 0.5349, 0.9641],
           [0.1773, 0.5401, 0.9602],
           [0.1768, 0.5452, 0.956],
           [0.1764, 0.5504, 0.9516],
           [0.1755, 0.5554, 0.9473],
           [0.174, 0.5605, 0.9432],
           [0.1716, 0.5655, 0.9393],
           [0.1686, 0.5705, 0.9357],
           [0.1649, 0.5755, 0.9323],
           [0.161, 0.5805, 0.9289],
           [0.1573, 0.5854, 0.9254],
           [0.154, 0.5902, 0.9218],
           [0.1513, 0.595, 0.9182],
           [0.1492, 0.5997, 0.9147],
           [0.1475, 0.6043, 0.9113],
           [0.1461, 0.6089, 0.908],
           [0.1446, 0.6135, 0.905],
           [0.1429, 0.618, 0.9022],
           [0.1408, 0.6226, 0.8998],
           [0.1383, 0.6272, 0.8975],
           [0.1354, 0.6317, 0.8953],
           [0.1321, 0.6363, 0.8932],
           [0.1288, 0.6408, 0.891],
           [0.1253, 0.6453, 0.8887],
           [0.1219, 0.6497, 0.8862],
           [0.1185, 0.6541, 0.8834],
           [0.1152, 0.6584, 0.8804],
           [0.1119, 0.6627, 0.877],
           [0.1085, 0.6669, 0.8734],
           [0.1048, 0.671, 0.8695],
           [0.1009, 0.675, 0.8653],
           [0.0964, 0.6789, 0.8609],
           [0.0914, 0.6828, 0.8562],
           [0.0855, 0.6865, 0.8513],
           [0.0789, 0.6902, 0.8462],
           [0.0713, 0.6938, 0.8409],
           [0.0628, 0.6972, 0.8355],
           [0.0535, 0.7006, 0.8299],
           [0.0433, 0.7039, 0.8242],
           [0.0328, 0.7071, 0.8183],
           [0.0234, 0.7103, 0.8124],
           [0.0155, 0.7133, 0.8064],
           [0.0091, 0.7163, 0.8003],
           [0.0046, 0.7192, 0.7941],
           [0.0019, 0.722, 0.7878],
           [0.0009, 0.7248, 0.7815],
           [0.0018, 0.7275, 0.7752],
           [0.0046, 0.7301, 0.7688],
           [0.0094, 0.7327, 0.7623],
           [0.0162, 0.7352, 0.7558],
           [0.0253, 0.7376, 0.7492],
           [0.0369, 0.74, 0.7426],
           [0.0504, 0.7423, 0.7359],
           [0.0638, 0.7446, 0.7292],
           [0.077, 0.7468, 0.7224],
           [0.0899, 0.7489, 0.7156],
           [0.1023, 0.751, 0.7088],
           [0.1141, 0.7531, 0.7019],
           [0.1252, 0.7552, 0.695],
           [0.1354, 0.7572, 0.6881],
           [0.1448, 0.7593, 0.6812],
           [0.1532, 0.7614, 0.6741],
           [0.1609, 0.7635, 0.6671],
           [0.1678, 0.7656, 0.6599],
           [0.1741, 0.7678, 0.6527],
           [0.1799, 0.7699, 0.6454],
           [0.1853, 0.7721, 0.6379],
           [0.1905, 0.7743, 0.6303],
           [0.1954, 0.7765, 0.6225],
           [0.2003, 0.7787, 0.6146],
           [0.2061, 0.7808, 0.6065],
           [0.2118, 0.7828, 0.5983],
           [0.2178, 0.7849, 0.5899],
           [0.2244, 0.7869, 0.5813],
           [0.2318, 0.7887, 0.5725],
           [0.2401, 0.7905, 0.5636],
           [0.2491, 0.7922, 0.5546],
           [0.2589, 0.7937, 0.5454],
           [0.2695, 0.7951, 0.536],
           [0.2809, 0.7964, 0.5266],
           [0.2929, 0.7975, 0.517],
           [0.3052, 0.7985, 0.5074],
           [0.3176, 0.7994, 0.4975],
           [0.3301, 0.8002, 0.4876],
           [0.3424, 0.8009, 0.4774],
           [0.3548, 0.8016, 0.4669],
           [0.3671, 0.8021, 0.4563],
           [0.3795, 0.8026, 0.4454],
           [0.3921, 0.8029, 0.4344],
           [0.405, 0.8031, 0.4233],
           [0.4184, 0.803, 0.4122],
           [0.4322, 0.8028, 0.4013],
           [0.4463, 0.8024, 0.3904],
           [0.4608, 0.8018, 0.3797],
           [0.4753, 0.8011, 0.3691],
           [0.4899, 0.8002, 0.3586],
           [0.5044, 0.7993, 0.348],
           [0.5187, 0.7982, 0.3374],
           [0.5329, 0.797, 0.3267],
           [0.547, 0.7957, 0.3159],
           [0.5609, 0.7943, 0.305],
           [0.5748, 0.7929, 0.2941],
           [0.5886, 0.7913, 0.2833],
           [0.6024, 0.7896, 0.2726],
           [0.6161, 0.7878, 0.2622],
           [0.6297, 0.7859, 0.2521],
           [0.6433, 0.7839, 0.2423],
           [0.6567, 0.7818, 0.2329],
           [0.6701, 0.7796, 0.2239],
           [0.6833, 0.7773, 0.2155],
           [0.6963, 0.775, 0.2075],
           [0.7091, 0.7727, 0.1998],
           [0.7218, 0.7703, 0.1924],
           [0.7344, 0.7679, 0.1852],
           [0.7468, 0.7654, 0.1782],
           [0.759, 0.7629, 0.1717],
           [0.771, 0.7604, 0.1658],
           [0.7829, 0.7579, 0.1608],
           [0.7945, 0.7554, 0.157],
           [0.806, 0.7529, 0.1546],
           [0.8172, 0.7505, 0.1535],
           [0.8281, 0.7481, 0.1536],
           [0.8389, 0.7457, 0.1546],
           [0.8495, 0.7435, 0.1564],
           [0.86, 0.7413, 0.1587],
           [0.8703, 0.7392, 0.1615],
           [0.8804, 0.7372, 0.165],
           [0.8903, 0.7353, 0.1695],
           [0.9, 0.7336, 0.1749],
           [0.9093, 0.7321, 0.1815],
           [0.9184, 0.7308, 0.189],
           [0.9272, 0.7298, 0.1973],
           [0.9357, 0.729, 0.2061],
           [0.944, 0.7285, 0.2151],
           [0.9523, 0.7284, 0.2237],
           [0.9606, 0.7285, 0.2312],
           [0.9689, 0.7292, 0.2373],
           [0.977, 0.7304, 0.2418],
           [0.9842, 0.733, 0.2446],
           [0.99, 0.7365, 0.2429],
           [0.9946, 0.7407, 0.2394],
           [0.9966, 0.7458, 0.2351],
           [0.9971, 0.7513, 0.2309],
           [0.9972, 0.7569, 0.2267],
           [0.9971, 0.7626, 0.2224],
           [0.9969, 0.7683, 0.2181],
           [0.9966, 0.774, 0.2138],
           [0.9962, 0.7798, 0.2095],
           [0.9957, 0.7856, 0.2053],
           [0.9949, 0.7915, 0.2012],
           [0.9938, 0.7974, 0.1974],
           [0.9923, 0.8034, 0.1939],
           [0.9906, 0.8095, 0.1906],
           [0.9885, 0.8156, 0.1875],
           [0.9861, 0.8218, 0.1846],
           [0.9835, 0.828, 0.1817],
           [0.9807, 0.8342, 0.1787],
           [0.9778, 0.8404, 0.1757],
           [0.9748, 0.8467, 0.1726],
           [0.972, 0.8529, 0.1695],
           [0.9694, 0.8591, 0.1665],
           [0.9671, 0.8654, 0.1636],
           [0.9651, 0.8716, 0.1608],
           [0.9634, 0.8778, 0.1582],
           [0.9619, 0.884, 0.1557],
           [0.9608, 0.8902, 0.1532],
           [0.9601, 0.8963, 0.1507],
           [0.9596, 0.9023, 0.148],
           [0.9595, 0.9084, 0.145],
           [0.9597, 0.9143, 0.1418],
           [0.9601, 0.9203, 0.1382],
           [0.9608, 0.9262, 0.1344],
           [0.9618, 0.932, 0.1304],
           [0.9629, 0.9379, 0.1261],
           [0.9642, 0.9437, 0.1216],
           [0.9657, 0.9494, 0.1168],
           [0.9674, 0.9552, 0.1116],
           [0.9692, 0.9609, 0.1061],
           [0.9711, 0.9667, 0.1001],
           [0.973, 0.9724, 0.0938],
           [0.9749, 0.9782, 0.0872],
           [0.9769, 0.9839, 0.0805]]

parula_map = LinearSegmentedColormap.from_list('parula', cm_data)