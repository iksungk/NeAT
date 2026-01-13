import os 
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import sys
sys.path.append('./')
sys.path.append('./misc/')

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import h5py as hp
import time
import argparse
import cv2
import climage

from models import *
from utils import *
from losses import *
from neat_func import *
from tqdm import tqdm
from psf_torch import PsfGenerator3D


parser = argparse.ArgumentParser(description="Hyperparameters")

parser.add_argument('--dataset', '-d', type=str, default='')
parser.add_argument('--rec_save_path_prefix', type=str, default='../results/')
parser.add_argument('--excitation_wavelength', type=float, default=0.92)
parser.add_argument('--na_exc', type=float, default=1.1)
parser.add_argument('--psf_dz', type=float, default=0.2) # 0.2
parser.add_argument('--psf_dy', type=float, default=0.125) 
parser.add_argument('--psf_dx', type=float, default=0.125)
parser.add_argument('--n_obj', type=float, default=1.333)
parser.add_argument('--normalized', type=bool, default=True)

args = parser.parse_args()


N = int(1 / (2. * args.na_exc / args.excitation_wavelength / 1024) / args.psf_dx)

psf = PsfGenerator3D(psf_shape=(1, N, N), 
                     units=(args.psf_dz, args.psf_dx, args.psf_dx), # um
                     na_exc=args.na_exc, 
                     lam_detection=args.excitation_wavelength, 
                     n=args.n_obj) 

kmask = torch.ones_like(psf.kmask2).cuda(0)


# Load aberration without H.
hf = hp.File('../results/' + args.dataset + '/rec.h5', 'r')
wf_est = hf['wf'][()]

# Load estimated conjugation error.
hf = hp.File('../results/H.h5', 'r')
ap = hf['ap'][()]
s = hf['s'][()]

hf.close()


# Applying H inverse to the estimated aberration without H to correct conjugation error.
wf_tmp = np.copy(wf_est)

for ii in range(2):
    kmask = torch.ones_like(psf.kmask2).cuda(0)
    given = torch.zeros_like(psf.krho)

    for j in range(len(wf_tmp)):
        given += 2 * torch.pi * wf_tmp[j] * psf.zernike_polynomial(j+3, args.normalized) / 2

    given = torch.fft.fftshift(given)
    given = torch.fliplr(given)
    given = torch.flipud(given)
    A = get_affine_mat(given.shape[0], given.shape[1], 0., 0., -2.75, 1., 1.)
    given = affine_img(given.unsqueeze(0).unsqueeze(0), A)[0, 0].cuda(0).detach()
    Hinv_given = given.cuda(0)

    if ii == 0:
    # Identity.
        H1 = torch.eye(2, 3).cuda(0)
        s1 = torch.tensor([[1.]]).cuda(0)

    else:
    # Load H.
        H1 = torch.from_numpy(ap).cuda(0)
        s1 = torch.from_numpy(s).cuda(0)

    H1inv = torch.inverse(torch.cat((H1, torch.tensor([[0, 0, 1]]).cuda(0)), dim=0))[0:2]
    s1inv = torch.inverse(s1)

    Hinv_given = affine_img(Hinv_given.unsqueeze(0).unsqueeze(0), H1inv, s1inv)[0, 0] 
    Hinv_given_mask = (Hinv_given * kmask)[N//2-512:N//2+512, N//2-512:N//2+512]
    Hinv_given_mask = (Hinv_given_mask - (-np.pi)) / (2 * np.pi) * (255 - 0)
    Hinv_given_mask = torch.remainder(Hinv_given_mask, 255)

    Hinv_given = Hinv_given[N//2-512:N//2+512, N//2-512:N//2+512]
    Hinv_given = (Hinv_given - (-np.pi)) / (2 * np.pi) * (255 - 0)
    Hinv_given = torch.remainder(Hinv_given, 255)

    
    if not os.path.exists(args.rec_save_path_prefix + 'patterns/'):
        os.makedirs(args.rec_save_path_prefix + 'patterns/')


    if ii == 0:
        print("Saving uncorrected SLM pattern.")
        cv2.imwrite(args.rec_save_path_prefix + 'patterns/k_vis_' + args.dataset + '_without_H.bmp',
                    np.uint8(Hinv_given.detach().cpu().numpy()))
        print('Uncorrected SLM pattern saved as ' + 'patterns/k_vis_' + args.dataset + '_without_H.bmp.')

        # print(climage.convert(args.rec_save_path_prefix + 'patterns/k_vis_' + args.dataset + '_without_H.bmp', is_unicode=True, width=50))

    else:
        print("Saving conjugation-error-corrected SLM pattern.")
        cv2.imwrite(args.rec_save_path_prefix + 'patterns/k_vis_' + args.dataset + '_with_H.bmp',
                    np.uint8(Hinv_given.detach().cpu().numpy()))
        print('Uncorrected SLM pattern saved as ' + 'patterns/k_vis_' + args.dataset + '_with_H.bmp.')

        # print(climage.convert(args.rec_save_path_prefix + 'patterns/k_vis_' + args.dataset + '_with_H.bmp', is_unicode=True, width=50))
