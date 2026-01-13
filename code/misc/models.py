import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange

dtype = torch.cuda.FloatTensor


# class estimate_alpha(nn.Module):
#     def __init__(self, init_value = 0.2):
#         super(estimate_alpha, self).__init__()
        
#         self.alpha = torch.nn.Parameter(torch.tensor(init_value, device = 'cuda:0', requires_grad = True)).type(dtype)
        
#     def forward(self):
#         return self.alpha


# class learnable_background(nn.Module):
#     def __init__(self, init_val = 0.1, image_stack = None, rank = 1):
#         super(learnable_background, self).__init__()

#         self.init_val = init_val
#         self.image_stack = image_stack
#         self.rank = rank
#         self.background_level_z = torch.nn.Parameter(torch.full((self.image_stack.shape[0], 1, 1, self.rank), self.init_val, device = 'cuda:0', requires_grad = True)).type(dtype)
#         self.background_level_y = torch.nn.Parameter(torch.full((1, self.image_stack.shape[1], 1, self.rank), self.init_val, device = 'cuda:0', requires_grad = True)).type(dtype)
#         self.background_level_x = torch.nn.Parameter(torch.full((1, 1, self.image_stack.shape[2], self.rank), self.init_val, device = 'cuda:0', requires_grad = True)).type(dtype)
    
#     def forward(self):
#         return (self.background_level_x * self.background_level_y * self.background_level_z).mean(-1)


class learnable_background_v2(nn.Module):
    def __init__(self, init_val = 0.1, image_stack = None, rank = 1):
        super(learnable_background_v2, self).__init__()

        self.init_val = init_val
        self.image_stack = image_stack
        self.rank = rank

        # self.tz = torch.nn.Parameter(self.init_val * torch.rand((self.rank, self.image_stack.shape[0], 1, 1), device = 'cuda:0', requires_grad = True)).type(dtype)
        # self.ty = torch.nn.Parameter(self.init_val * torch.rand((self.rank, 1, self.image_stack.shape[1], 1), device = 'cuda:0', requires_grad = True)).type(dtype)
        # self.tx = torch.nn.Parameter(self.init_val * torch.rand((self.rank, 1, 1, self.image_stack.shape[2]), device = 'cuda:0', requires_grad = True)).type(dtype)
        self.tz = torch.nn.Parameter(torch.full((self.rank, self.image_stack.shape[0], 1, 1), self.init_val, device = 'cuda:0', requires_grad = True)).type(dtype)
        self.ty = torch.nn.Parameter(torch.full((self.rank, 1, self.image_stack.shape[1], 1), self.init_val, device = 'cuda:0', requires_grad = True)).type(dtype)
        self.tx = torch.nn.Parameter(torch.full((self.rank, 1, 1, self.image_stack.shape[2]), self.init_val, device = 'cuda:0', requires_grad = True)).type(dtype)

    def forward(self):
        bck = self.tz * self.ty * self.tx
        bck = torch.sum(bck, 0)
    
        return torch.abs(bck)
    
    
class optimal_kernel(nn.Module):
    def __init__(self, max_val = 0.2, init_value = None, order_up_to = 5, piston_tip_tilt = False):
        super(optimal_kernel, self).__init__()
        
        if init_value is None:
            if not piston_tip_tilt:
                _k4 = torch.rand((1, ), device = 'cuda:0', requires_grad = True)
                
            else:
                _k4 = torch.rand((4, ), device = 'cuda:0', requires_grad = True)
                
            _k5 = torch.full((1, ), fill_value = 0.5, device = 'cuda:0', requires_grad = True)
            
            if order_up_to == 3:
                _k6_10 = torch.rand((5, ), device = 'cuda:0', requires_grad = True)
            
                self.k = torch.nn.Parameter(2 * max_val * torch.cat((_k4, _k5, _k6_10), 0) - max_val).type(dtype)
                
            elif order_up_to == 4:
                _k6_15 = torch.rand((10, ), device = 'cuda:0', requires_grad = True)

                self.k = torch.nn.Parameter(2 * max_val * torch.cat((_k4, _k5, _k6_15), 0) - max_val).type(dtype)
                
            elif order_up_to == 5:
                _k6_21 = torch.rand((16, ), device = 'cuda:0', requires_grad = True)

                self.k = torch.nn.Parameter(2 * max_val * torch.cat((_k4, _k5, _k6_21), 0) - max_val).type(dtype)
                
            elif order_up_to == 6:
                _k6_28 = torch.rand((23, ), device = 'cuda:0', requires_grad = True)

                self.k = torch.nn.Parameter(2 * max_val * torch.cat((_k4, _k5, _k6_28), 0) - max_val).type(dtype)
                
            elif order_up_to == 7:
                _k6_36 = torch.rand((31, ), device = 'cuda:0', requires_grad = True)

                self.k = torch.nn.Parameter(2 * max_val * torch.cat((_k4, _k5, _k6_36), 0) - max_val).type(dtype)
                
            elif order_up_to == 8:
                _k6_45 = torch.rand((40, ), device = 'cuda:0', requires_grad = True)

                self.k = torch.nn.Parameter(2 * max_val * torch.cat((_k4, _k5, _k6_45), 0) - max_val).type(dtype)
                
            elif order_up_to == 9:
                _k6_55 = torch.rand((50, ), device = 'cuda:0', requires_grad = True)

                self.k = torch.nn.Parameter(2 * max_val * torch.cat((_k4, _k5, _k6_55), 0) - max_val).type(dtype)
            
        else:
            # self.k = torch.nn.parameter.Parameter(torch.tensor(init_value, device = 'cuda:0', requires_grad = True), requires_grad = True).type(dtype)
            _k = init_value[0] * torch.ones((1, ), device = 'cuda:0', requires_grad = True)

            for num in range(1, len(init_value)):
                _k = torch.cat((_k, init_value[num] * torch.ones((1, ), device = 'cuda:0', requires_grad = True)))

            self.k = torch.nn.Parameter(_k).type(dtype)
            
    def forward(self):
        return self.k


class LinearNet(nn.Module):
    def __init__(self, input_ch, W, D_in):
        super(LinearNet, self).__init__()
        
        self.input_ch = input_ch
        self.W = W
        self.D_in = D_in
        
        self.linears = nn.ModuleList(
            [nn.Linear(input_ch, W, bias=False)] + [nn.Linear(W, W, bias=False) for i in range(D_in-1)]
        )
        
        self.last_layer = nn.Linear(W, input_ch, bias=False)
        
    def forward(self, x):
        h = x
        
        for i, _ in enumerate(self.linears):
            h = self.linears[i](h)
            h = F.relu(h)
        
        outputs = self.last_layer(h)
        
        return outputs


def input_coord_2d(image_width, image_height):
    tensors = [torch.linspace(-1, 1, steps = image_width), torch.linspace(-1, 1, steps = image_height)]
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = rearrange(mgrid, 'w h c -> (w h) c')
    
    return mgrid


def input_coord_3d(image_width, image_height, image_depth):
    tensors = [torch.linspace(-1, 1, steps = image_width), torch.linspace(-1, 1, steps = image_height), torch.linspace(-1, 1, steps = image_depth)]
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = rearrange(mgrid, 'w h d c -> (w h d) c')
    
    return mgrid


def radial_encoding(in_node, dia_digree, L_xy):
    s = torch.sin(torch.arange(0, 180, dia_digree) * np.pi / 180).unsqueeze(-1).cuda(0)
    c = torch.cos(torch.arange(0, 180, dia_digree) * np.pi / 180).unsqueeze(-1).cuda(0)
    
    fourier_mapping = torch.cat((s, c), axis = -1).T
    xy_freq = torch.matmul(in_node[:, :2], fourier_mapping)
    
    # Accommodate float values for L_xy.
    for l in range(int(np.floor(L_xy))):
        cur_freq = torch.cat((torch.sin(2 ** (l+L_xy-np.floor(L_xy)) * np.pi * xy_freq), torch.cos(2 ** (l+L_xy-np.floor(L_xy)) * np.pi * xy_freq)), axis = -1)
        
        if l == 0:
            tot_freq = cur_freq
        else:
            tot_freq = torch.cat((tot_freq, cur_freq), axis = -1)
    
    # for l in range(L_z):
    #     cur_freq = torch.cat((torch.sin(2 ** l * np.pi * in_node[:, 2].unsqueeze(-1)), torch.cos(2 ** l * np.pi * in_node[:, 2].unsqueeze(-1))), axis = -1)
    #     tot_freq = torch.cat((tot_freq, cur_freq), axis = -1)
        
    
    return tot_freq


def radial_encoding_3d(in_node, dia_digree, L_xy, L_z):
    s = torch.sin(torch.arange(0, 180, dia_digree) * np.pi / 180).unsqueeze(-1).cuda(0)
    c = torch.cos(torch.arange(0, 180, dia_digree) * np.pi / 180).unsqueeze(-1).cuda(0)
    
    fourier_mapping = torch.cat((s, c), axis = -1).T
    xy_freq = torch.matmul(in_node[:, :2], fourier_mapping)
    
    for l in range(L_xy):
        cur_freq = torch.cat((torch.sin(2 ** l * np.pi * xy_freq), torch.cos(2 ** l * np.pi * xy_freq)), axis = -1)
        
        if l == 0:
            tot_freq = cur_freq
        else:
            tot_freq = torch.cat((tot_freq, cur_freq), axis = -1)
    
    for l in range(L_z):
        cur_freq = torch.cat((torch.sin(2 ** l * np.pi * in_node[:, 2].unsqueeze(-1)), torch.cos(2 ** l * np.pi * in_node[:, 2].unsqueeze(-1))), axis = -1)
        tot_freq = torch.cat((tot_freq, cur_freq), axis = -1)
    
    return tot_freq


class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12
        Inputs:
            x: (B, self.in_channels)
        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)

    
class NF(nn.Module):
    def __init__(self,
                 D=8, 
                 W=256,
                 skips=[2,4,6],
                 in_channels = -1,
                 out_channels = 1
                 ):

        super(NF, self).__init__()
        self.D = D
        self.W = W
        self.skips = skips

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.enc_last = nn.Linear(W, W)

        # output layers
        self.post = nn.Sequential(
                        nn.Linear(W, W//2),
                        nn.ReLU(True),
                        nn.Linear(W//2, out_channels))
        
    def forward(self, x):
        input_xyz = x

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        enc_last = self.enc_last(xyz_)
        obj = self.post(enc_last)
        
        return obj

    
class NF_Dropout(nn.Module):
    def __init__(self,
                 D=8, 
                 W=256,
                 skips=[4],
                 in_channels = 63,
                 out_channels = 180,
                 dropout_rate = 0.5
                 ):

        super(NF_Dropout, self).__init__()
        self.D = D
        self.W = W
        self.skips = skips
        
        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels, W)
            else:
                layer = nn.Linear(W, W)
            
            if i%2 == 0 and i > 0:
                layer = nn.Sequential(nn.Dropout(dropout_rate), layer, nn.ReLU(True))
            else:
                layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
            
        self.enc_last = nn.Linear(W, W)

        # output layers
        self.post = nn.Sequential(
                        nn.Dropout(dropout_rate),
                        nn.Linear(W, W//2),
                        nn.ReLU(True),
                        nn.Linear(W//2, out_channels))
                
    def forward(self, x):
        input_xyz = x

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        enc_last = self.enc_last(xyz_)
        obj = self.post(enc_last)
        
        return obj


class LinearNet(nn.Module):
    def __init__(self, input_ch, W, D_in, max_val):
        super(LinearNet, self).__init__()
        
        self.input_ch = input_ch
        self.W = W
        self.D_in = D_in
        self.max_val = max_val
        
        self.linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) for i in range(D_in-1)]
        )
    
        self.last_layer = nn.Linear(W, input_ch)
        self.linears.apply(self._init_weights)
        self.last_layer.apply(self._init_weights)
    
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain = 1.0)
            m.bias.data.fill_(0.0)
    
    
    def forward(self):
        # h = 2 * self.max_val * torch.rand((self.input_ch, )) - self.max_val
        h = torch.linspace(-self.max_val, self.max_val, self.input_ch)
        
        for i, _ in enumerate(self.linears):
            h = self.linears[i](h)
            h = F.relu(h)
        
        outputs = self.last_layer(h)
        
        return outputs
