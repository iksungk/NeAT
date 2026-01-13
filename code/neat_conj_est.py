import os 
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import sys
sys.path.append('./')
sys.path.append('./misc/')

import math
import torch
import torchvision
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import h5py as hp

from neat_func import *
from utils import *
from tqdm import tqdm
from psf_torch import PsfGenerator3D


parser = argparse.ArgumentParser(description="Hyperparameters")

parser.add_argument('--rec_save_path_prefix', type=str, default='../results/')
parser.add_argument('--excitation_wavelength', type=float, default=0.92)
parser.add_argument('--na_exc', type=float, default=1.1)
parser.add_argument('--psf_dz', type=float, default=0.2) # 0.2
parser.add_argument('--psf_dy', type=float, default=0.125) 
parser.add_argument('--psf_dx', type=float, default=0.125)
parser.add_argument('--n_obj', type=float, default=1.333)
parser.add_argument('--normalized', type=bool, default=True)

args = parser.parse_args()


# Data loading
print('Load aberrations from calibration stacks.')
hf = hp.File('../results/commercial_beads_zeros/rec.h5', 'r')
y_ = hf['y'][()]
wf_zeros = hf['wf'][()]

hf = hp.File('../results/commercial_beads_mode4/rec.h5', 'r')
wf_mode4 = hf['wf'][()]

hf = hp.File('../results/commercial_beads_mode6/rec.h5', 'r')
wf_mode6 = hf['wf'][()]

hf = hp.File('../results/commercial_beads_mode8/rec.h5', 'r')
wf_mode8 = hf['wf'][()]

hf = hp.File('../results/commercial_beads_mode9/rec.h5', 'r')
wf_mode9 = hf['wf'][()]

hf = hp.File('../results/commercial_beads_mode13/rec.h5', 'r')
wf_mode13 = hf['wf'][()]

hf.close()


# Gradient descent
print('Gradient descent..')
color = 'green' if args.excitation_wavelength == 0.92 else 'red'

N = int(1 / (2. * args.na_exc / args.excitation_wavelength / 1024) / args.psf_dx)

psf = PsfGenerator3D(psf_shape=(1, N, N), 
                     units=(args.psf_dz, args.psf_dx, args.psf_dx), # um
                     na_exc=args.na_exc, 
                     lam_detection=args.excitation_wavelength, 
                     n=args.n_obj) 

kmask = torch.ones_like(psf.kmask2).cuda(0)

kmask_sm = kmask[N//2-512:N//2+512, N//2-512:N//2+512]

M = 5
_given = torch.zeros((M, N, N)).cuda(0)

for num in range(0, M):
    given = torch.zeros_like(psf.krho)
    
    wf_given = torch.zeros((12, ))

    if num == 0:
        wf_given[5] = 0.4
    elif num == 1:
        wf_given[4] = 0.4
    elif num == 2:
        wf_given[2] = 0.4
    elif num == 3:
        wf_given[0] = 0.4
    elif num == 4:
        wf_given[9] = 0.4
    elif num == 5:
        wf_given = torch.zeros((12, ))

    wf_given = np.copy(wf_given)
    
    for j in range(len(wf_given)):
        given += 2 * torch.pi * wf_given[j] * psf.zernike_polynomial(j+3, args.normalized) / 2
    
    given = torch.fft.fftshift(given)
    given = torch.fliplr(given)
    given = torch.flipud(given)

    A = get_affine_mat(given.shape[0], given.shape[1], 0., 0., 0., 1., 1.)
    given = affine_img(given.unsqueeze(0).unsqueeze(0), A)[0, 0].cuda(0).detach()
    
    _given[num] = given


_hat_mask = torch.zeros((M, 1024, 1024)).cuda(0)

for num in range(0, M):
    hat = torch.zeros_like(psf.krho)
    
    # (-1) in SLM calibration -- hat and given should have a difference in a multiplication factor of (-1).
    if num == 0:
        wf_hat = wf_zeros - wf_mode9

    elif num == 1:
        wf_hat = wf_zeros - wf_mode8

    elif num == 2:
        wf_hat = wf_zeros - wf_mode6

    elif num == 3:
        wf_hat = wf_zeros - wf_mode4

    elif num == 4:
        wf_hat = wf_zeros - wf_mode13

    else:
        wf_hat = torch.zeros((12, )) - wf_zeros

    wf_hat = np.copy(wf_hat)
    
    for j in range(len(wf_hat)):
        hat += 2 * np.pi * wf_hat[j] * psf.zernike_polynomial(j+3, args.normalized) / 2
    
    hat = torch.fft.fftshift(hat)
    hat = torch.fliplr(hat)
    hat = torch.flipud(hat)
    hat = affine_img(hat.unsqueeze(0).unsqueeze(0), A)[0, 0].cuda(0).detach()
    hat_mask = (hat * kmask)[N//2-512:N//2+512, N//2-512:N//2+512].cuda(0)

    _hat_mask[num] = hat_mask


ap = linear_operator().cuda(0)
s = scaling_coefficient().cuda(0)
N_grad = 2000

opt_config = [{'params':ap.parameters(), 'lr':1e-3,
               'betas':(0.9, 0.999)}, 
              {'params':s.parameters(), 'lr':1e-3,
               'betas':(0.9, 0.999)}]
optimizer = torch.optim.Adam(opt_config, eps=1e-8)

loss_list = np.empty(shape = (1 + N_grad, )); loss_list[:] = np.NaN
s_list = np.empty(shape = (1 + N_grad, )); s_list[:] = np.NaN
ap_list = np.empty(shape = (1 + N_grad, 2, 3)); ap_list[:] = np.NaN

_H_given_mask = torch.zeros((M, 1024, 1024)).cuda(0)

for grad_it in tqdm(range(N_grad)):
    loss = torch.zeros((M, )).cuda(0)
    
    for num in range(0, M):
        H_given = _given[num].cuda(0)
        H_given = affine_img(H_given.unsqueeze(0).unsqueeze(0), ap())[0, 0] 

        H_given_mask = (H_given * kmask)[N//2-512:N//2+512, N//2-512:N//2+512]
        _kmask = kmask[N//2-512:N//2+512, N//2-512:N//2+512]
        
        _H_given_mask[num] = H_given_mask.detach()
        
        loss[num] =  torch.abs(H_given_mask[_kmask] - _hat_mask[num][_kmask].detach()).mean()
        
    loss = loss.mean()
    loss_list[grad_it] = loss.item()
    s_list[grad_it] = s().detach().cpu().numpy()
    ap_list[grad_it] = ap().detach().cpu().numpy()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if grad_it % 1000 == 999:
        plt.figure(figsize = (15, 5))
        plt.subplot(1,3,1); plt.plot(loss_list)
        plt.subplot(1,3,2)
        for ii in range(2):
            for jj in range(3):
                plt.plot(ap_list[:, ii, jj], label = '[' + str(ii) + ', ' + str(jj) + ']')
        plt.legend()
        plt.subplot(1,3,3); plt.plot(s_list)
        plt.show()
    

_H_given_mask = (_H_given_mask - (-np.pi)) / (2 * np.pi) * (255 - 0)

af = affine_matrix().cuda(0)

opt_config = [{'params':af.parameters(), 'lr':1e-3,
               'betas':(0.9, 0.999)}]
optimizer = torch.optim.Adam(opt_config, eps=1e-8)

N_grad = 5000
loss_list = np.empty(shape = (1 + N_grad, )); loss_list[:] = np.NaN

for grad_it in tqdm(range(N_grad)):
    loss = torch.square(af() - ap().detach()).mean()

    loss_list[grad_it] = loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(loss_list); plt.show()


print('H: ')
print('Translation - t_x: ' + str(2 * y_.shape[2] * af.tx.detach().cpu().numpy()) + ' px, t_y: ' + str(2 * y_.shape[1] * af.ty.detach().cpu().numpy()) + ' px')
print('Rotation - theta: ' + str(180 * af.theta.detach().cpu().numpy() / np.pi) + ' deg')
print('Scaling - Sc_x: ' + str(1/af.scx.detach().cpu().numpy()) + ', Sc_y: ' + str(1/af.scy.detach().cpu().numpy()))
print('Shear - Sh_x: ' + str(-af.shy.detach().cpu().numpy()) + ', Sh_y: ' + str(-af.shx.detach().cpu().numpy()))


print('H inverse: ')
print('Translation - t_x: ' + str(-2 * y_.shape[2] * af.tx.detach().cpu().numpy()) + ' px, t_y: ' + str(-2 * y_.shape[1] * af.ty.detach().cpu().numpy()) + ' px')
print('Rotation - theta: ' + str(-180 * af.theta.detach().cpu().numpy() / np.pi) + ' deg')
print('Scaling - Sc_x: ' + str(af.scx.detach().cpu().numpy()) + ', Sc_y: ' + str(af.scy.detach().cpu().numpy()))
print('Shear - Sh_x: ' + str(af.shx.detach().cpu().numpy()) + ', Sh_y: ' + str(af.shy.detach().cpu().numpy()))


hf = hp.File(args.rec_save_path_prefix + 'H.h5', 'w')
hf.create_dataset('ap', data=ap().detach().cpu().numpy())
hf.create_dataset('s', data=s().detach().cpu().numpy())
hf.close()