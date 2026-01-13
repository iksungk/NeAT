import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from psf_torch import PsfGenerator3D

dtype = torch.cuda.FloatTensor


class adaptive_registration(nn.Module):
    def __init__(self, depth, include_scalar=False):
        super(adaptive_registration, self).__init__()

        self.depth = depth
        self.A00 = torch.nn.Parameter(torch.ones(self.depth, 1, device = 'cuda:0', requires_grad = True)).type(dtype)
        self.A01 = torch.nn.Parameter(torch.zeros(self.depth, 1, device = 'cuda:0', requires_grad = True)).type(dtype)
        self.A02 = torch.nn.Parameter(torch.zeros(self.depth, 1, device = 'cuda:0', requires_grad = True)).type(dtype)
        self.A10 = torch.nn.Parameter(torch.zeros(self.depth, 1, device = 'cuda:0', requires_grad = True)).type(dtype)
        self.A11 = torch.nn.Parameter(torch.ones(self.depth, 1, device = 'cuda:0', requires_grad = True)).type(dtype)
        self.A12 = torch.nn.Parameter(torch.zeros(self.depth, 1, device = 'cuda:0', requires_grad = True)).type(dtype)

        self.include_scalar = include_scalar
        if self.include_scalar:
            self.S = torch.nn.Parameter(torch.ones(self.depth, 1, device = 'cuda:0', requires_grad = True)).type(dtype)

    def forward(self, x):
        depth = self.depth
        
        A00 = self.A00
        A01 = self.A01
        A02 = self.A02
        A10 = self.A10
        A11 = self.A11
        A12 = self.A12

        if self.include_scalar:
            S = self.S
    
        y = torch.zeros_like(x).cuda(0)
        As = torch.zeros(depth, 2, 3).cuda(0)
        
        for num in range(depth):
            A = torch.cat((torch.cat((A00[num], A01[num], A02[num])).unsqueeze(0),
                           torch.cat((A10[num], A11[num], A12[num])).unsqueeze(0)), dim=0)
            As[num] = A

            _x = x[num].unsqueeze(0).unsqueeze(0)
            
            affine_mat = A.repeat(_x.shape[0],1,1)
            grid = F.affine_grid(affine_mat, _x.size()).type(dtype)

            if self.include_scalar:
                _x = F.grid_sample(_x, grid) * S[num]
            else:
                _x = F.grid_sample(_x, grid)

            y[num] = _x[0, 0]

        if self.include_scalar:
            return y, As, S
        else:
            return y, As


def get_affine_mat(Nx, Ny, tx, ty, theta, sx, sy):
    # tx, ty: units in pixels; Convention: +tx: > (to right), +ty: V (to down).
    # theta: units in degrees; Convention: +theta: counter-clockwise.
    # sx, sy: Convention: sx, sy > 1: zoom.
    if isinstance(tx, float):
        tx = torch.tensor(tx, requires_grad = False).cuda(0)
    if isinstance(ty, float):
        ty = torch.tensor(ty, requires_grad = False).cuda(0)
    if isinstance(theta, float):
        theta = torch.tensor(theta, requires_grad = False).cuda(0)
    if isinstance(sx, float):
        sx = torch.tensor(sx, requires_grad = False).cuda(0)
    if isinstance(sy, float):
        sy = torch.tensor(sy, requires_grad = False).cuda(0)

    tx = -2*tx/Nx
    ty = -2*ty/Ny
    theta = theta/180*np.pi
    sx = 1/sx
    sy = 1/sy
    
    C = torch.cos(theta)
    S = torch.sin(theta)
    
    return torch.tensor([[sx * C, -sx * S, sx * (tx * C - ty * S)],
                         [sy * S, sy * C, sy * (tx * S + ty * C)]], requires_grad = False).cuda(0)


def affine_img(x, A, s=None, dtype=dtype):
    affine_mat = A.repeat(x.shape[0],1,1)
    grid = F.affine_grid(affine_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid)

    if s is not None:
        x = x * s
    return x


class affine_matrix(nn.Module):
    def __init__(self):
        super(affine_matrix, self).__init__()

        self.tx = torch.nn.Parameter(torch.zeros(1, device = 'cuda:0', requires_grad = True)).type(dtype)
        self.ty = torch.nn.Parameter(torch.zeros(1, device = 'cuda:0', requires_grad = True)).type(dtype)

        self.theta = torch.nn.Parameter(torch.zeros(1, device = 'cuda:0', requires_grad = True)).type(dtype)

        self.scx = torch.nn.Parameter(torch.ones(1, device = 'cuda:0', requires_grad = True)).type(dtype)
        self.scy = torch.nn.Parameter(torch.ones(1, device = 'cuda:0', requires_grad = True)).type(dtype)

        self.shx = torch.nn.Parameter(torch.ones(1, device = 'cuda:0', requires_grad = True)).type(dtype)
        self.shy = torch.nn.Parameter(torch.ones(1, device = 'cuda:0', requires_grad = True)).type(dtype)

    def forward(self):
        C = torch.cos(self.theta)
        S = torch.sin(self.theta)

        scx = self.scx
        scy = self.scy

        shx = self.shx
        shy = self.shy

        tx = self.tx
        ty = self.ty

        A = torch.cat((torch.cat((C*scx+S*scy*shx, -S*scx+C*scy*shx, (C*scx+S*scy*shx)*tx+(-S*scx+C*scy*shx)*ty)).unsqueeze(0),
                       torch.cat((S*scy+C*scx*shy, C*scy-S*scx*shy, (S*scy+C*scx*shy)*tx+(C*scy-S*scx*shy)*ty)).unsqueeze(0)), dim = 0)
        
        return A


class linear_operator(nn.Module):
    def __init__(self, scale=True, rotation=True, translation=True):
        super(linear_operator, self).__init__()

        self.A = torch.nn.Parameter(torch.eye(2, 3, device = 'cuda:0', requires_grad = True)).type(dtype)

    def forward(self):
        return self.A


class scaling_coefficient(nn.Module):
    def __init__(self, scale=True, rotation=True, translation=True):
        super(scaling_coefficient, self).__init__()

        self.s = torch.nn.Parameter(torch.full((1, 1), 1.0, device = 'cuda:0', requires_grad = True)).type(dtype)

    def forward(self):
        return self.s


class weight_mask(nn.Module):
    def __init__(self, N=1024):
        super(weight_mask, self).__init__()

        self.mask = torch.nn.Parameter(torch.full((N, N), 1.0, device = 'cuda:0', requires_grad = True)).type(dtype)

    def forward(self):
        return self.mask


def draw_k_vis(args, _wf, pupil_mask=True):
    N = int(1 / (2. * args.na_exc / args.excitation_wavelength / 1024) / args.psf_dx)
    
    psf = PsfGenerator3D(psf_shape=(1, N, N), 
                         units=(args.psf_dz, args.psf_dx, args.psf_dx), # um
                         na_exc=args.na_exc, 
                         lam_detection=args.excitation_wavelength, 
                         n=args.n_obj) 

    if pupil_mask:
        kmask = torch.fft.fftshift(psf.kmask2).cuda(0)
    else: 
        kmask = torch.ones_like(psf.kmask2).cuda(0)
    
    k_vis = torch.zeros_like(psf.krho)
    for j in range(len(_wf)):
        k_vis += _wf[j] * psf.zernike_polynomial(j+3, args.normalized)
    
    k_vis = torch.fft.fftshift(k_vis)
    k_vis = torch.fliplr(k_vis)
    k_vis = torch.flipud(k_vis)
    # A = get_affine_mat(k_vis.shape[0], k_vis.shape[1], 0., 0., -2.75, 1., 1.)
    A = get_affine_mat(k_vis.shape[0], k_vis.shape[1], 0., 0., 0., 1., 1.)
    k_vis = affine_img(k_vis.unsqueeze(0).unsqueeze(0), A)[0, 0].cuda(0).detach()
    # 
    # k_vis = (k_vis - (-np.pi)) / (2 * np.pi) * 2.
    # k_vis = k_vis - k_vis.min()
    # k_vis = torch.remainder(k_vis * kmask, 255)
    k_vis[~kmask] = np.NaN
    k_vis = k_vis.detach().cpu().numpy()
    # [N//2-512:N//2+512, N//2-512:N//2+512]

    return k_vis