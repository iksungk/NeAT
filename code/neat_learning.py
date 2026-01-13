import os 
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import sys
sys.path.append('./')
sys.path.append('./misc/')

import math
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.io as sio
import time
import h5py as hp
import mrcfile
import tifffile
import argparse
import scipy
import pandas
import cv2
import climage 

from models import *
from utils import *
from losses import *
from metrics import *
from neat_func import *
from train_func import *

from termplot import Plot
from tqdm import tqdm
from PIL import Image
from psf_torch import PsfGenerator3D
from scipy.stats import pearsonr
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
from sklearn.mixture import GaussianMixture
from skimage.transform import rescale, resize
from skimage.metrics import structural_similarity
from matplotlib.colors import LinearSegmentedColormap
from torch.fft import fftn, ifftn, fftshift, ifftshift, rfftn, irfftn, rfft
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR


dtype = torch.cuda.FloatTensor
torch.backends.cudnn.benchmark = True

gfp = custom_div_cmap(2**16-1, mincol='#000000', midcol = '#00FF00', maxcol='#FFFFFF')

df = pandas.read_csv('./misc/green_hot_lut.csv')
cm_data = np.divide(df.values, np.full_like(df.values, 255.))
green_hot = LinearSegmentedColormap.from_list('green_hot', cm_data)


parser = argparse.ArgumentParser(description="Hyperparameters - tentative")

parser.add_argument('--dataset', '-d', type=str, default='')
parser.add_argument('--sample_motion', '-sm', type=bool, default=False)
parser.add_argument('--feature_nominal_diameter', '-fnd', type=int, default=4)

parser.add_argument('--filepath_ref', type=str, default='../data/')
parser.add_argument('--filepath', type=str, default='./data/')
parser.add_argument('--rec_save_path_prefix', type=str, default='../results/')

parser.add_argument('--suffix', type=str, default='NeAT')
parser.add_argument('--suffix_rec', type=str, default='')

parser.add_argument('--cnts', type=int, nargs='+', default=[50, 200, 200])
parser.add_argument('--dims', type=int, nargs='+', default=[50, 200, 200])
parser.add_argument('--sampling_z', type=int, default=1) # 1, 2
parser.add_argument('--sampling_xy', type=int, default=1) # 1, 2
parser.add_argument('--normalized', type=bool, default=True)
parser.add_argument('--psf_dz', type=float, default=0.2) # 0.2
parser.add_argument('--psf_dy', type=float, default=0.125) 
parser.add_argument('--psf_dx', type=float, default=0.125)
parser.add_argument('--na_exc', type=float, default=1.1)
parser.add_argument('--excitation_wavelength', type=float, default=0.92)
parser.add_argument('--n_obj', type=float, default=1.333)

parser.add_argument('--padding_xy', type=int, default=0)
parser.add_argument('--padding_z', type=int, default=0)
parser.add_argument('--padding_mode', type=str, default='replicate')

parser.add_argument('--encoding_option', type=str, default='radial')
parser.add_argument('--radial_encoding_angle', type=float, default=3)
parser.add_argument('--radial_encoding_depth', type=int, default=6)

parser.add_argument('--nerf_num_layers', type=int, default=6)
parser.add_argument('--nerf_num_filters', type=int, default=64)
parser.add_argument('--nerf_skips', type=list, default=[2,4,6])
parser.add_argument('--nerf_beta', type=float, default=1.0) 
parser.add_argument('--nerf_max_val', type=float, default=10.0)

parser.add_argument('--pretraining', type=bool, default=True, help='')
parser.add_argument('--pretraining_num_iter', type=int, default=1000, help='')
parser.add_argument('--pretraining_lr', type=float, default=1e-2, help='')
parser.add_argument('--pretraining_last_epoch_lr', type=float, default=2.5e-3, help='')
parser.add_argument('--pretraining_measurement_scalar', type=float, default=3.5, help='')
parser.add_argument('--pretraining_lp_sigma', type=float, default=0.0)

parser.add_argument('--training_num_iter', type=int, default=500) 
parser.add_argument('--training_lr_obj', type=float, default=5e-3)
parser.add_argument('--training_lr_ker', type=float, default=1e-2)
parser.add_argument('--training_last_epoch_lr', type=float, default=2.5e-3)
parser.add_argument('--kernel_max_val', type=float, default=1e-2)
parser.add_argument('--kernel_order_up_to', type=int, default=4)
parser.add_argument('--wf_error_criterion', type=float, default=2.0)

parser.add_argument('--opt_beta_1_obj', type=float, default=0.9)
parser.add_argument('--opt_beta_2_obj', type=float, default=0.999)
parser.add_argument('--opt_beta_1_ker', type=float, default=0.9)
parser.add_argument('--opt_beta_2_ker', type=float, default=0.999)
parser.add_argument('--opt_beta_1_lb', type=float, default=0.9)
parser.add_argument('--opt_beta_2_lb', type=float, default=0.999)
parser.add_argument('--opt_eps', type=float, default=1e-8)

parser.add_argument('--background_value', type=float, default=0.0)
parser.add_argument('--include_lb', type=bool, default=False)
parser.add_argument('--training_lr_lb', type=float, default=1e-3)
parser.add_argument('--lb_init_val', type=float, default=1e-2)
parser.add_argument('--lb_rank', type=int, default=5)
parser.add_argument('--lb_max', type=float, default=0.3)
parser.add_argument('--lb_reg', type=float, default=1e-3)

parser.add_argument('--ssim_weight', type=float, default=1.0)
parser.add_argument('--relative_error_mode', type=str, default='l2')
parser.add_argument('--relative_mse_eps', type=float, default=1e-2)
parser.add_argument('--hessian', type=float, default=5e-3)
parser.add_argument('--hessian_eps', type=float, default=2e-4)
parser.add_argument('--tv_z', type=float, default=1e-9)
parser.add_argument('--tv_xy', type=float, default=1e-9)
parser.add_argument('--tv_normalize', type=bool, default=False)

parser.add_argument('--nld_z', type=float, default=1e-6)
parser.add_argument('--nld_z_lbd', type=float, default=200.0)
parser.add_argument('--nld_z_negative_slope', type=float, default=1e-1)

parser.add_argument('--savgol_filter_z', type=bool, default=False) 

parser.add_argument('--lr_schedule', type=str, default='cosine')

args = parser.parse_args()


"""
Loading physical parameters and data
"""

args.hessian_version = 'v2'
include_im_sq = False if args.hessian_version == 'v2' else True

assert args.excitation_wavelength == 0.92 or args.excitation_wavelength == 1.
color = 'green' if args.excitation_wavelength == 0.92 else 'red'
args.cmap = green_hot if color == 'green' else 'hot'

print(args.dataset)
if not os.path.exists(args.rec_save_path_prefix + args.dataset):
    os.makedirs(args.rec_save_path_prefix + args.dataset)
  
args.rec_save_path_prefix = args.rec_save_path_prefix + args.dataset + '/'
im_stack = tifffile.imread('../data/' + args.dataset + '/AVG_Tifs.tif')

im_stack = np.flip(im_stack, 2)
im_stack = np.flip(im_stack, 0)

y_ = im_stack[args.cnts[0]-args.dims[0] : args.cnts[0]+args.dims[0] : args.sampling_z,
              args.cnts[1]-args.dims[1] : args.cnts[1]+args.dims[1] : args.sampling_xy,
              args.cnts[2]-args.dims[2] : args.cnts[2]+args.dims[2] : args.sampling_xy].astype(np.float32)

locs = str(args.cnts[0]-args.dims[0]) + '_' + str(args.cnts[0]+args.dims[0]) + '_'
locs += str(args.cnts[1]-args.dims[1]) + '_' + str(args.cnts[1]+args.dims[1]) + '_'
locs += str(args.cnts[2]-args.dims[2]) + '_' + str(args.cnts[2]+args.dims[2])

y_max = np.max(y_); args.y_max = y_max
y_min = np.min(y_); args.y_min = y_min

classif = GaussianMixture(n_components=2, tol = 1e-3, max_iter = 100)
classif.fit(y_.reshape((y_.size, 1)))
threshold = np.mean(np.sort(classif.means_, axis=0)[0:2])
idx_bck = y_ < threshold

y_ = (y_ - y_min) / (y_max - y_min)

y_dt = y_ - ndimage.uniform_filter(y_, size=3) # detrend with uniform filter
bck_std = y_dt[idx_bck].std()

bck_ab_std = np.abs(y_dt)[idx_bck].std()
bck_tv_std = bck_tv_std_compute(args, y_, idx_bck, include_im_sq=include_im_sq)


"""
Learning process
"""
# Options (variable)
have_sample_motion = args.sample_motion # adaptive reg
feature_nominal_diameter = args.feature_nominal_diameter # px


# -----------------------------------------
# Options (fixed)
capture_stronger_aberration = True
stronger_piecewise_smoothness = True
expect_power_attenuation_over_depth = True
# -----------------------------------------


# -----------------------------------------
# Hyperparameters (fixed)
args.normalized = True

args.lr_schedule = 'exponential'
args.pretraining_lr = 1e-2 
args.pretraining_last_epoch_lr = 1e-3 
args.training_lr_obj = 4e-3 
args.training_lr_ker = 4e-3
args.training_lr_lb = 4e-3 
args.training_last_epoch_lr = 1e-6
args.training_opt = 'RAdam'
args.lb_rank = 5 
args.lb_version = 'v2'
args.lb_reg = 0.0
args.lb_init_val = (0.1 * bck_std) ** (1/3) 
args.include_lb = expect_power_attenuation_over_depth # True if (thick) sample with beam absorption (esp. in vivo); noisy background (low SNR)
# lb for background signal fluctuations along axes / power decrease along the depth axis 

args.kernel_max_val = 1e-2
args.kernel_order_up_to = 4
args.encoding_option = 'radial'
args.radial_encoding_angle = 15 
args.radial_encoding_depth = int(math.log2(next_power_of_2(y_.shape[1] / feature_nominal_diameter) - 1))

args.pretraining_lp_sigma = 10./(args.psf_dz / 0.2) if capture_stronger_aberration else 5./(args.psf_dz / 0.2) 
args.pretraining_measurement_scalar = 1. * np.maximum(1., estimate_structure_max_value(args, y_.shape[0], y_.shape[1], y_.shape[2], feature_nominal_diameter))
args.pretraining_num_iter = 5000 
args.training_num_iter = 5000 

args.relative_error_mode = 'l2'

args.pretraining_ssim_weight = 1.
args.ssim_weight = np.maximum(0.25, np.minimum(1., -75 * (bck_std - 0.03) + 0.25)) # 0.25 if bck_std >= 0.03 else 1.0
args.relative_mse_eps = np.maximum(2e-3, bck_std) # 2e-3 lower limit for numerical stability

args.opt_beta_1_obj = 0.9
args.opt_beta_2_obj = 0.999
args.opt_beta_1_ker = 0.9
args.opt_beta_2_ker = 0.999
args.opt_beta_1_lb = 0.9
args.opt_beta_2_lb = 0.999

args.padding_xy = 50 # 50
args.padding_z = 25 # 25
args.padding_mode = 'replicate'

args.nerf_max_val = 10.
args.nerf_num_layers = 4
args.nerf_skips = [2,4,6]
args.nerf_beta = 1.0
args.nerf_num_filters = 64

args.tv_normalize = True
args.nld_z = 1e-6 * np.prod(y_.shape)
args.nld_z_ubd = 2.0 * args.pretraining_measurement_scalar 
args.nld_z_lbd = 5e-3 * args.pretraining_measurement_scalar if not args.include_lb else 0.0 
args.nld_z_negative_slope = 1e-1 if stronger_piecewise_smoothness else 2e-2 


args.hessian_v2 = 5e-3
args.hessian_v2_eps = np.maximum(2e-3, bck_tv_std) # 2e-3, lower limit for numerical stability
args.ab_eps = np.maximum(2e-3, bck_ab_std) # 2e-3, lower limit for numerical stability
args.ab_weight = 2.


args.adaptive_reg = have_sample_motion # for motion correction
args.include_scalar = False
args.training_lr_ar = 7e-2
args.opt_beta_1_ar = 0.9
args.opt_beta_2_ar = 0.999
# -----------------------------------------

print("Name of reconstruction file: " + args.rec_save_path_prefix + "rec.h5")
print("Not considering background..") if not expect_power_attenuation_over_depth else print("Considering background..")
print("Input: in vivo with motion artifacts.") if have_sample_motion else print("Input: no motion artifacts.")


results = train(args, 
                y_, 
                wf_init=None,
                display_figures=True, 
                save_files=True, 
                device_num=0,
                freeze_psf=False)

print("Results saved as rec.h5.")

print("Estimated Zernike coefficients (ANSI) (unit: waves): ")
print(np.round(results[5][:12], 3))


def _generate_pattern(_wf_given, diff_limit = False):
    N = int(1 / (2. * args.na_exc / args.excitation_wavelength / 1024) / args.psf_dx)

    psf = PsfGenerator3D(psf_shape=(1, N, N), 
                         units=(args.psf_dz, args.psf_dx, args.psf_dx), # um
                         na_exc=args.na_exc, 
                         lam_detection=args.excitation_wavelength, 
                         n=args.n_obj) 

    if diff_limit:
        kmask = torch.fft.fftshift(psf.kmask2).cuda(0)

    else:
        kmask = torch.ones_like(psf.kmask2).cuda(0)
    
    k_vis = torch.zeros_like(psf.krho)
    
    for j in range(len(_wf_given)):
        k_vis += 2 * torch.pi * _wf_given[j] * psf.zernike_polynomial(j+3, args.normalized) / 2
   
    k_vis = torch.fft.fftshift(k_vis)
    k_vis = torch.fliplr(k_vis)
    k_vis = torch.flipud(k_vis)
    A = get_affine_mat(k_vis.shape[0], k_vis.shape[1], 0., 0., -2.75, 1., 1.)
    k_vis = affine_img(k_vis.unsqueeze(0).unsqueeze(0), A)[0, 0].cuda(0).detach()
    
    k_vis = (k_vis - (-np.pi)) / (2 * np.pi) * (255 - 0)
    k_vis = torch.remainder(k_vis * kmask, 255)
    k_vis[~kmask] = 255//2
    k_vis = k_vis.detach().cpu().numpy()[N//2-512:N//2+512, N//2-512:N//2+512]

    return k_vis


print("Generating estimated aberration map (unit: waves)...")
k_vis_tmp = _generate_pattern(estimated_to_given_conversion_ml_to_dmd(results[5], order_up_to=4), diff_limit=True)
cv2.imwrite(args.rec_save_path_prefix + 'est_aber_map.bmp', np.uint8(k_vis_tmp)) 
print("Estimated aberration map saved as est_aber_map.bmp.")


print("Generating SLM pattern (8-bit grayscale, corrective and wrapped phase map)...")
k_vis_tmp = _generate_pattern(results[5])
cv2.imwrite(args.rec_save_path_prefix + 'slm_pattern.bmp', np.uint8(k_vis_tmp)) 
print("SLM pattern saved as slm_pattern.bmp.")