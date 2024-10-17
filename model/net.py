import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.solvers import power_method, uball_project
from model.utils   import pre_process, post_process, calc_pad_2D, unpad, pre_process_3d, post_process_3d
from model.gabor   import ConvAdjoint2dGabor

def ST(x,t):
    """ shrinkage-thresholding operation. 
    """
    return x.sign()*F.relu(x.abs()-t)

class CDLNet(nn.Module):
    """ Convolutional Dictionary Learning Network:
    Interpretable denoising DNN with adaptive thresholds for robustness.
    """
    def __init__(self,
                 K = 3,            # num. unrollings
                 M = 64,           # num. filters in each filter bank operation
                 P = 7,            # square filter side length
                 s = 1,            # stride of convolutions
                 C = 1,            # num. input channels
                 t0 = 0,        # initial threshold
                 adaptive = False, # noise-adaptive thresholds
                 init = True):     # False -> use power-method for weight init
        super(CDLNet, self).__init__()
        
        # -- OPERATOR INIT --
        self.A = nn.ModuleList([nn.Conv2d(C, M, P, stride=s, padding=(P-1)//2, bias=False)  for _ in range(K)])
        self.B = nn.ModuleList([nn.ConvTranspose2d(M, C, P, stride=s, padding=(P-1)//2, output_padding=s-1, bias=False) for _ in range(K)])
        self.D = self.B[0]                              # alias D to B[0], otherwise unused as z0 is zero
        self.t = nn.Parameter(t0*torch.ones(K,2,M,1,1)) # 1st num layer, 2nd t_0 + t_1 _sighat, 3rd channel dim, 4th and 5th boardcast
        self.g = nn.Parameter(t0 * torch.ones(K,2,M,1,1))
        W = torch.randn(M, C, P, P)
        for k in range(K):
            self.A[k].weight.data = W.clone()
            self.B[k].weight.data = W.clone()

        # Don't bother running code if initializing trained model from state-dict
        if init:
            print("Running power-method on initial dictionary...")
            with torch.no_grad():
                DDt = lambda x: self.D(self.A[0](x))
                L = power_method(DDt, torch.rand(1,C,128,128), num_iter=200, verbose=False)[0]
                print(f"Done. L={L:.3e}.")

                if L < 0:
                    print("STOP: something is very very wrong...")
                    sys.exit()

            # spectral normalization (note: D is alised to B[0])
            for k in range(K):
                self.A[k].weight.data /= np.sqrt(L)
                self.B[k].weight.data /= np.sqrt(L)

        # set parameters
        self.K = K
        self.M = M
        self.P = P
        self.s = s
        self.t0 = t0
        self.adaptive = adaptive

    @torch.no_grad()
    def project(self):
        """ \ell_2 ball projection for filters, R_+ projection for thresholds
        """
        self.t.clamp_(0.0) 
        for k in range(self.K):
            self.A[k].weight.data = uball_project(self.A[k].weight.data)
            self.B[k].weight.data = uball_project(self.B[k].weight.data)

    def forward(self, y, sigma=None, mask=1):
        """ LISTA + D w/ noise-adaptive thresholds
        """ 
        yp, params, mask = pre_process(y, self.s, mask=mask)

        # THRESHOLD SCALE-FACTOR c
        c = 0 if sigma is None or not self.adaptive else sigma/255.0

        # LISTA
        z = ST(self.A[0](yp), self.t[0,:1] + c*self.t[0,1:2])
        for k in range(1, self.K):
            z = ST(z - self.A[k](mask*self.B[k](z) - yp), self.t[k,:1] + c*self.t[k,1:2])

        # DICTIONARY SYNTHESIS
        xphat = self.D(z)
        xhat  = post_process(xphat, params)
        return xhat, z

    def forward_generator(self, y, sigma=None, mask=1):
        """ same as forward but yeilds intermediate sparse codes
        """
        yp, params, mask = pre_process(y, self.s, mask=mask)
        c = 0 if sigma is None or not self.adaptive else sigma/255.0
        z = ST(self.A[0](yp), self.t[0,:1] + c*self.t[0,1:2]); yield z
        for k in range(1, self.K):
            z = ST(z - self.A[k](mask*self.B[k](z) - yp), self.t[k,:1] + c*self.t[k,1:2]); yield z
        xphat = self.D(z)
        xhat  = post_process(xphat, params)
        yield xhat
class ResidualBlock(nn.Module):
    """A basic residual block with two convolutional layers and a skip connection."""
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)

    def forward(self, x):
        identity = x  # Store input for the skip connection
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity  # Add skip connection
        out = self.relu(out)
        return out
class CDLNetVideo(nn.Module):
    """ Convolutional Dictionary Learning Network for Video Denoising with Residual Blocks"""
    def __init__(self,
                 K=3,            # num. unrollings
                 M=64,           # num. filters in each filter bank operation
                 P=(7, 7, 5),    # filter dimensions [height, width, depth]
                 s=1,            # stride of convolutions
                 C=1,            # num. input channels
                 t0=0,           # initial threshold
                 adaptive=False,
                 depth=3,
                 init=True,
                 residual=False):  # Add residual argument to toggle residual blocks
        super(CDLNetVideo, self).__init__()
        
        # -- OPERATOR INIT --
        self.A = nn.ModuleList([
            nn.Conv3d(C, M, P, stride=s, padding=(P[0]//2, P[1]//2, P[2]//2), bias=False) for _ in range(K)
        ])
        self.B = nn.ModuleList([
            nn.ConvTranspose3d(M, C, P, stride=s, padding=(P[0]//2, P[1]//2, P[2]//2), output_padding=s-1, bias=False) for _ in range(K)
        ])
        self.D = self.B[0]  # alias D to B[0], otherwise unused as z0 is zero
        self.t = nn.Parameter(t0 * torch.ones(K, 2, M, 1, 1, 1)) # learned thresholds (added one more dimension)
        
        # Residual blocks after each analysis convolution (if residual is True)
        self.residual = residual
        if self.residual:
            self.residual_blocks = nn.ModuleList([
                ResidualBlock(M, M) for _ in range(K)
            ])
        
        # set weights
        W = torch.randn(M, C, P[0], P[1], P[2])  # match new kernel size
        for k in range(K):
            self.A[k].weight.data = W.clone()
            self.B[k].weight.data = W.clone()
        
        # Don't bother running code if initializing trained model from state-dict
        if init:
            print("Running power-method on initial dictionary...")
            with torch.no_grad():
                DDt = lambda x: self.D(self.A[0](x))
                L = power_method(DDt, torch.rand(1, C, depth, 128, 128), num_iter=200, verbose=False)[0]
                print(f"Done. L={L:.3e}.")
                
                if L < 0:
                    print("STOP: something is very very wrong...")
                    sys.exit()
                
            # spectral normalization (note: D is aliased to B[0])
            for k in range(K):
                self.A[k].weight.data /= np.sqrt(L)
                self.B[k].weight.data /= np.sqrt(L)
        
        # set parameters
        self.K = K
        self.M = M
        self.P = P
        self.s = s
        self.t0 = t0
        self.adaptive = adaptive

    @torch.no_grad()
    def project(self):
        """ \ell_2 ball projection for filters, R_+ projection for thresholds """
        self.t.clamp_(0.0)
        for k in range(self.K):
            self.A[k].weight.data = uball_project(self.A[k].weight.data, dim=(2,3,4)) #onto the unit ball for 3D convolutions
            self.B[k].weight.data = uball_project(self.B[k].weight.data, dim=(2,3,4))

    def forward(self, y, sigma=None, mask=1):
        """ LISTA + D w/ noise-adaptive thresholds """
        yp, params, mask = pre_process_3d(y, self.s, mask=mask)
        
        # THRESHOLD SCALE-FACTOR c
        c = 0 if sigma is None or not self.adaptive else sigma / 255.0
        
        # LISTA with optional Residual Blocks
        z = ST(self.A[0](yp), self.t[0, :1] + c * self.t[0, 1:2])
        if self.residual:
            z = self.residual_blocks[0](z)  # Apply residual block after first convolution
        
        for k in range(1, self.K):
            z = ST(z - self.A[k](mask * self.B[k](z) - yp), self.t[k, :1] + c * self.t[k, 1:2])
            if self.residual:
                z = self.residual_blocks[k](z)  # Apply residual block after each convolution
        
        # DICTIONARY SYNTHESIS
        xphat = self.D(z)
        xhat = post_process_3d(xphat, params)
        return xhat, z

    def forward_generator(self, y, sigma=None, mask=1):
        """ same as forward but yields intermediate sparse codes """
        yp, params, mask = pre_process_3d(y, self.s, mask=mask)
        c = 0 if sigma is None or not self.adaptive else sigma / 255.0
        z = ST(self.A[0](yp), self.t[0, :1] + c * self.t[0, 1:2]); yield z
        if self.residual:
            z = self.residual_blocks[0](z)
        for k in range(1, self.K):
            z = ST(z - self.A[k](mask * self.B[k](z) - yp), self.t[k, :1] + c * self.t[k, 1:2]); yield z
            if self.residual:
                z = self.residual_blocks[k](z)
        xphat = self.D(z)
        xhat = post_process(xphat, params)
        yield xhat

def prox_CSR(u, z_prev, lambd, gamma):
    """
    Proximal operator of the CSR penalty using soft-thresholding.

    Parameters:
        u (torch.Tensor): Current tensor.
        z_prev (torch.Tensor): Previous frame's tensor.
        lambd (torch.Tensor): Outer threshold.
        gamma (torch.Tensor): Inner threshold.

    Returns:
        torch.Tensor: Result after applying the proximal operator.
    """
    return ST(ST(u - z_prev - lambd * torch.sign(z_prev), lambd * gamma) + z_prev + lambd * torch.sign(z_prev), lambd)

def prox_CSR_f2(u, z_prev, z_after, lambd, gamma1, gamma2):
    """
    Proximal operator of the CSR penalty using soft-thresholding.

    Parameters:
        u (torch.Tensor): Current tensor.
        z_prev (torch.Tensor): Previous frame's tensor.
        z_after (torch.Tensor): Afters frame's tensor.
        lambd (torch.Tensor): Outer threshold.
        gamma1 (torch.Tensor): Inner threshold.
        gamma2 (torch.Tensor): Midder threshold.
    Returns:
        torch.Tensor: Result after applying the proximal operator.
    """
    Ca = z_prev + lambd * torch.sign(z_prev) + lambd * gamma2 * torch.sign(z_prev - z_after)
    Cb = z_after + lambd * torch.sign(z_after) + lambd * gamma1 * torch.sign(z_after - z_prev)
    inner = ST(u - Ca, gamma1*lambd)
    midder = ST(inner - Cb + lambd * gamma1 * torch.sign(u - Ca), gamma2*lambd)
    return ST(midder + Cb - lambd * gamma1 * torch.sign(u - Ca), lambd)

class CDLNet_CSR_old(nn.Module):
    """ Convolutional Dictionary Learning Network:
    Interpretable denoising DNN with adaptive thresholds for robustness.
    """
    def __init__(self,
                 K=3,            # num. unrollings
                 M=64,           # num. filters in each filter bank operation
                 P=7,            # square filter side length
                 s=1,            # stride of convolutions
                 C=1,            # num. input channels
                 t0=0,           # initial threshold
                 adaptive=False, # noise-adaptive thresholds
                 init=True):     # False -> use power-method for weight init
        super(CDLNet_CSR, self).__init__()
        
        # -- OPERATOR INIT --
        self.A = nn.ModuleList([nn.Conv2d(C, M, P, stride=s, padding=(P-1)//2, bias=False) for _ in range(K)])
        self.B = nn.ModuleList([nn.ConvTranspose2d(M, C, P, stride=s, padding=(P-1)//2, output_padding=s-1, bias=False) for _ in range(K)])
        self.D = self.B[0]  # alias D to B[0], otherwise unused as z0 is zero
        self.t = nn.Parameter(t0 * torch.ones(K, 2, M, 1, 1))  # learned thresholds
        self.g = nn.Parameter(t0 * torch.ones(K,2,M,1,1))
        # set weights 
        W = torch.randn(M, C, P, P)
        for k in range(K):
            self.A[k].weight.data = W.clone()
            self.B[k].weight.data = W.clone()
        
        # Don't bother running code if initializing trained model from state-dict
        if init:
            print("Running power-method on initial dictionary...")
            with torch.no_grad():
                DDt = lambda x: self.D(self.A[0](x))
                L = power_method(DDt, torch.rand(1, C, 128, 128), num_iter=200, verbose=False)[0]
                print(f"Done. L={L:.3e}.")
                
                if L < 0:
                    print("STOP: something is very very wrong...")
                    sys.exit()
            
            # spectral normalization (note: D is aliased to B[0])
            for k in range(K):
                self.A[k].weight.data /= np.sqrt(L)
                self.B[k].weight.data /= np.sqrt(L)
        
        # set parameters
        self.K = K
        self.M = M
        self.P = P
        self.s = s
        self.t0 = t0
        self.adaptive = adaptive

    @torch.no_grad()
    def project(self):
        """ \ell_2 ball projection for filters, R_+ projection for thresholds """
        self.t.clamp_(0.0) 
        for k in range(self.K):
            self.A[k].weight.data = uball_project(self.A[k].weight.data)
            self.B[k].weight.data = uball_project(self.B[k].weight.data)

    def forward(self, y, z_prev=None, sigma=None, mask=1):
        """
        LISTA + D with temporal information for 3D denoising.

        Parameters:
            y (torch.Tensor): Noisy input.
            z_prev (torch.Tensor, optional): Previous frame's sparse code. Defaults to None.
            sigma (float, optional): Noise level. Defaults to None.
            mask (torch.Tensor, optional): Mask for convolutions. Defaults to 1.

        Returns:
            tuple: (Denoised output, Current sparse code)
        """ 
        yp, params, mask = pre_process(y, self.s, mask=mask)

        # THRESHOLD SCALE-FACTOR c
        c = 0 if sigma is None or not self.adaptive else sigma / 255.0

        # Initialize z
        if z_prev is None:
            z = ST(self.A[0](yp), self.t[0, :1] + c * self.t[0, 1:2])
        else:
            # Incorporate temporal information using Prox_CSR for the first iteration if needed
            z = prox_CSR(self.A[0](yp), z_prev, self.t[0, :1] + c * self.t[0, 1:2], self.g[0, :1] + c * self.g[0, 1:2])

        # Iterative updates
        for k in range(1, self.K):
            if z_prev is not None:
                u = z - self.A[k](mask * self.B[k](z) - yp)
                z = prox_CSR(u, z_prev, self.t[k, :1] + c * self.t[k, 1:2], self.g[k, :1] + c * self.g[k, 1:2])
            else:
                u = z - self.A[k](mask * self.B[k](z) - yp)
                z = ST(u, self.t[k, :1] + c * self.t[k, 1:2])

        # DICTIONARY SYNTHESIS
        xphat = self.D(z)
        xhat = post_process(xphat, params)
        return xhat, z

class CDLNet_CSR(nn.Module):
    """ Convolutional Dictionary Learning Network:
    Interpretable denoising DNN with adaptive thresholds for robustness.
    """
    def __init__(self,
                 K=3,            # num. unrollings
                 M=64,           # num. filters in each filter bank operation
                 P=7,            # square filter side length
                 s=1,            # stride of convolutions
                 C=1,            # num. input channels
                 t0=0,           # initial threshold
                 adaptive=False, # noise-adaptive thresholds
                 init=True):     # False -> use power-method for weight init
        super(CDLNet_CSR, self).__init__()
        
        # -- OPERATOR INIT --
        self.A = nn.ModuleList([nn.Conv2d(C, M, P, stride=s, padding=(P-1)//2, bias=False) for _ in range(K)])
        self.B = nn.ModuleList([nn.ConvTranspose2d(M, C, P, stride=s, padding=(P-1)//2, output_padding=s-1, bias=False) for _ in range(K)])
        self.A2 = nn.ModuleList([nn.Conv2d(C, M, P, stride=s, padding=(P-1)//2, bias=False) for _ in range(K)])
        self.B2 = nn.ModuleList([nn.ConvTranspose2d(M, C, P, stride=s, padding=(P-1)//2, output_padding=s-1, bias=False) for _ in range(K)])
        self.D = self.B[0]  # alias D to B[0], otherwise unused as z0 is zero
        self.t = nn.Parameter(t0 * torch.ones(K, 2, M, 1, 1))  # learned thresholds
        self.t2 = nn.Parameter(t0 * torch.ones(K, 2, M, 1, 1))  # learned thresholds
        self.g = nn.Parameter(t0 * torch.ones(K,2,M,1,1))
        # set weights 
        W = torch.randn(M, C, P, P)
        for k in range(K):
            self.A[k].weight.data = W.clone()
            self.B[k].weight.data = W.clone()
        
        # Don't bother running code if initializing trained model from state-dict
        if init:
            print("Running power-method on initial dictionary...")
            with torch.no_grad():
                DDt = lambda x: self.D(self.A[0](x))
                L = power_method(DDt, torch.rand(1, C, 128, 128), num_iter=200, verbose=False)[0]
                print(f"Done. L={L:.3e}.")
                
                if L < 0:
                    print("STOP: something is very very wrong...")
                    sys.exit()
            
            # spectral normalization (note: D is aliased to B[0])
            for k in range(K):
                self.A[k].weight.data /= np.sqrt(L)
                self.B[k].weight.data /= np.sqrt(L)
        
        # set parameters
        self.K = K
        self.M = M
        self.P = P
        self.s = s
        self.t0 = t0
        self.adaptive = adaptive

    @torch.no_grad()
    def project(self):
        """ \ell_2 ball projection for filters, R_+ projection for thresholds """
        self.t.clamp_(0.0) 
        for k in range(self.K):
            self.A[k].weight.data = uball_project(self.A[k].weight.data)
            self.B[k].weight.data = uball_project(self.B[k].weight.data)

    def forward(self, y, z_prev=None, sigma=None, mask=1):
        """
        LISTA + D with temporal information for 3D denoising.

        Parameters:
            y (torch.Tensor): Noisy input.
            z_prev (torch.Tensor, optional): Previous frame's sparse code. Defaults to None.
            sigma (float, optional): Noise level. Defaults to None.
            mask (torch.Tensor, optional): Mask for convolutions. Defaults to 1.

        Returns:
            tuple: (Denoised output, Current sparse code)
        """ 
        yp, params, mask = pre_process(y, self.s, mask=mask)

        # THRESHOLD SCALE-FACTOR c
        c = 0 if sigma is None or not self.adaptive else sigma / 255.0

        # Initialize z
        if z_prev is None:
            z = ST(self.A2[0](yp), self.t2[0, :1] + c * self.t2[0, 1:2])
        else:
            # Incorporate temporal information using Prox_CSR for the first iteration if needed
            z = prox_CSR(self.A[0](yp), z_prev, self.t[0, :1] + c * self.t[0, 1:2], self.g[0, :1] + c * self.g[0, 1:2])

        # Iterative updates
        for k in range(1, self.K):
            if z_prev is not None:
                u = z - self.A[k](mask * self.B[k](z) - yp)
                z = prox_CSR(u, z_prev, self.t[k, :1] + c * self.t[k, 1:2], self.g[k, :1] + c * self.g[k, 1:2])
            else:
                u = z - self.A2[k](mask * self.B2[k](z) - yp)
                z = ST(u, self.t2[k, :1] + c * self.t2[k, 1:2])

        # DICTIONARY SYNTHESIS
        xphat = self.D(z)
        xhat = post_process(xphat, params)
        return xhat, z
class CDLNet_CSRf2(nn.Module):
    """ Convolutional Dictionary Learning Network:
    Interpretable denoising DNN with adaptive thresholds for robustness, using prox_CSR_f2.
    """
    def __init__(self,
                 K=3,            # num. unrollings
                 M=64,           # num. filters in each filter bank operation
                 P=7,            # square filter side length
                 s=1,            # stride of convolutions
                 C=1,            # num. input channels
                 t0=0,           # initial threshold
                 adaptive=False, # noise-adaptive thresholds
                 init=True):     # False -> use power-method for weight init
        super(CDLNet_CSRf2, self).__init__()
        
        # -- OPERATOR INIT --
        self.A = nn.ModuleList([nn.Conv2d(C, M, P, stride=s, padding=(P-1)//2, bias=False) for _ in range(K)])
        self.B = nn.ModuleList([nn.ConvTranspose2d(M, C, P, stride=s, padding=(P-1)//2, output_padding=s-1, bias=False) for _ in range(K)])
        self.D = self.B[0]  # alias D to B[0], otherwise unused as z0 is zero
        self.t = nn.Parameter(t0 * torch.ones(K, 2, M, 1, 1))  # learned thresholds
        self.g1 = nn.Parameter(t0 * torch.ones(K, 2, M, 1, 1)) # learned inner thresholds
        self.g2 = nn.Parameter(t0 * torch.ones(K, 2, M, 1, 1)) # learned middle thresholds
        # set weights 
        W = torch.randn(M, C, P, P)
        for k in range(K):
            self.A[k].weight.data = W.clone()
            self.B[k].weight.data = W.clone()
        
        # Don't bother running code if initializing trained model from state-dict
        if init:
            print("Running power-method on initial dictionary...")
            with torch.no_grad():
                DDt = lambda x: self.D(self.A[0](x))
                L = power_method(DDt, torch.rand(1, C, 128, 128), num_iter=200, verbose=False)[0]
                print(f"Done. L={L:.3e}.")
                
                if L < 0:
                    print("STOP: something is very very wrong...")
                    sys.exit()
            
            # spectral normalization (note: D is aliased to B[0])
            for k in range(K):
                self.A[k].weight.data /= np.sqrt(L)
                self.B[k].weight.data /= np.sqrt(L)
        
        # set parameters
        self.K = K
        self.M = M
        self.P = P
        self.s = s
        self.t0 = t0
        self.adaptive = adaptive

    @torch.no_grad()
    def project(self):
        """ \ell_2 ball projection for filters, R_+ projection for thresholds """
        self.t.clamp_(0.0) 
        for k in range(self.K):
            self.A[k].weight.data = uball_project(self.A[k].weight.data)
            self.B[k].weight.data = uball_project(self.B[k].weight.data)

    def forward(self, y, z_prev=None, z_after=None, sigma=None, mask=1):
        """
        LISTA + D with temporal information for 3D denoising, utilizing prox_CSR_f2.

        Parameters:
            y (torch.Tensor): Noisy input.
            z_prev (torch.Tensor, optional): Previous frame's sparse code. Defaults to None.
            z_after (torch.Tensor, optional): After frame's sparse code. Defaults to None.
            sigma (float, optional): Noise level. Defaults to None.
            mask (torch.Tensor, optional): Mask for convolutions. Defaults to 1.

        Returns:
            tuple: (Denoised output, Current sparse code)
        """ 
        yp, params, mask = pre_process(y, self.s, mask=mask)

        # THRESHOLD SCALE-FACTOR c
        c = 0 if sigma is None or not self.adaptive else sigma / 255.0

        if z_prev is None and z_after is not None:
            z = prox_CSR(self.A[0](yp), z_after, self.t[0, :1] + c * self.t[0, 1:2], self.g2[0, :1] + c * self.g2[0, 1:2])
        elif z_after is None and z_prev is not None:
            z = prox_CSR(self.A[0](yp), z_prev, self.t[0, :1] + c * self.t[0, 1:2], self.g1[0, :1] + c * self.g1[0, 1:2])
        elif z_after is not None and z_prev is not None:
            z = prox_CSR_f2(self.A[0](yp), z_prev, z_after, self.t[0, :1] + c * self.t[0, 1:2], 
                           self.g1[0, :1] + c * self.g1[0, 1:2], self.g2[0, :1] + c * self.g2[0, 1:2])
        else:
            z = ST(self.A[0](yp), self.t[0, :1] + c * self.t[0, 1:2])

        for k in range(1, self.K):
            u = z - self.A[k](mask * self.B[k](z) - yp)
            if z_prev is None and z_after is not None:
                z = prox_CSR(u, z_after, self.t[k, :1] + c * self.t[k, 1:2], self.g2[k, :1] + c * self.g2[k, 1:2])
            elif z_after is None and z_prev is not None:
                z = prox_CSR(u, z_prev, self.t[k, :1] + c * self.t[k, 1:2], self.g1[k, :1] + c * self.g1[k, 1:2])
            elif z_after is not None and z_prev is not None:
                z = prox_CSR_f2(u, z_prev, z_after, self.t[k, :1] + c * self.t[k, 1:2], 
                            self.g1[k, :1] + c * self.g1[k, 1:2], self.g2[k, :1] + c * self.g2[k, 1:2])
            else:
                z = ST(u, self.t[k, :1] + c * self.t[k, 1:2])
        # DICTIONARY SYNTHESIS
        xphat = self.D(z)
        xhat = post_process(xphat, params)
        return xhat, z
class GDLNet(nn.Module):
    """ Gabor Dictionary Learning Network:
    """
    def __init__(self,
                 K = 3,            # num. unrollings
                 M = 64,           # num. filters in each filter bank operation
                 P = 7,            # square filter side length
                 s = 1,            # stride of convolutions
                 C = 1,            # num. input channels
                 t0 = 0,           # initial threshold
                 order = 1,        # mixture of gabor order
                 adaptive = False, # noise-adaptive thresholds
                 shared = "",      # which gabor parameters to share (e.g. "a_psi_w0_alpha")
                 init = True):     # False -> use power-method for weight init
        super(GDLNet, self).__init__()
        
        # -- operator init --
        self.A = nn.ModuleList([ConvAdjoint2dGabor(M, C, P, stride=s, order=order) for _ in range(K)])
        self.B = nn.ModuleList([ConvAdjoint2dGabor(M, C, P, stride=s, order=order) for _ in range(K)])
        self.D = self.B[0]                              # alias D to B[0], otherwise unused as z0 is zero
        self.t = nn.Parameter(t0*torch.ones(K,2,M,1,1)) # learned thresholds

        # set weights 
        alpha = torch.randn(order, M, C, 1, 1)
        a     = torch.randn(order, M, C, 2)
        w0    = torch.randn(order, M, C, 2)
        psi   = torch.randn(order, M, C)

        for k in range(K):
            self.A[k].alpha.data = alpha.clone()
            self.A[k].a.data     = a.clone()
            self.A[k].w0.data    = w0.clone()
            self.A[k].psi.data   = psi.clone()
            self.B[k].alpha.data = alpha.clone()
            self.B[k].a.data     = a.clone()
            self.B[k].w0.data    = w0.clone()
            self.B[k].psi.data   = psi.clone()

            # Gabor parameter sharing
            if k > 0:
                if "alpha" in shared:
                    self.A[k].alpha = self.A[0].alpha
                    # never share alpha (scale) with final dictionary (B[0])
                    if k > 1:
                        self.B[k].alpha = self.B[1].alpha
                if "a_" in shared:
                    self.A[k].a     = self.A[0].a
                    self.B[k].a     = self.B[0].a
                if "w0" in shared:
                    self.A[k].w0    = self.A[0].w0
                    self.B[k].w0    = self.B[0].w0
                if "psi" in shared:
                    self.A[k].psi   = self.A[0].psi
                    self.B[k].psi   = self.B[0].psi

        # Don't bother running code if initializing trained model from state-dict
        if init:
            print("Running power-method on initial dictionary...")
            with torch.no_grad():
                DDt = lambda x: self.D(self.A[0].T(x))
                L = power_method(DDt, torch.rand(1,C,128,128), num_iter=200, verbose=False)[0]
                print(f"Done. L={L:.3e}.")

                if L < 0:
                    print("STOP: something is very very wrong...")
                    sys.exit()

            # spectral normalization (note: D is alised to B[0])
            for k in range(K):
                self.A[k].alpha.data /= np.sqrt(L)
                self.B[k].alpha.data /= np.sqrt(L)
                if "alpha" in shared:
                    self.B[1].alpha.data /= np.sqrt(L)
                    break

        # set parameters
        self.K = K
        self.M = M
        self.P = P
        self.s = s
        self.t0 = t0
        self.order = order
        self.adaptive = adaptive

    @torch.no_grad()
    def project(self):
        """ \ell_2 ball projection for filters, R_+ projection for thresholds
        """
        self.t.clamp_(0.0) 

    def forward(self, y, sigma=None, mask=1):
        """ LISTA + D w/ noise-adaptive thresholds
        """ 
        yp, params, mask = pre_process(y, self.s, mask=mask)

        # THRESHOLD SCALE-FACTOR c
        c = 0 if sigma is None or not self.adaptive else sigma/255.0

        # LISTA
        z = ST(self.A[0].T(yp), self.t[0,:1] + c*self.t[0,1:2])
        for k in range(1, self.K):
            z = ST(z - self.A[k].T(mask*self.B[k](z) - yp), self.t[k,:1] + c*self.t[k,1:2])

        # DICTIONARY SYNTHESIS
        xphat = self.D(z)
        xhat  = post_process(xphat, params)
        return xhat, z

    def forward_generator(self, y, sigma=None, mask=1):
        """ same as forward but yeilds intermediate sparse codes
        """
        yp, params, mask = pre_process(y, self.s, mask=mask)
        c = 0 if sigma is None or not self.adaptive else sigma/255.0
        z = ST(self.A[0].T(yp), self.t[0,:1] + c*self.t[0,1:2]); yield z
        for k in range(1, self.K):
            z = ST(z - self.A[k].T(mask*self.B[k](z) - yp), self.t[k,:1] + c*self.t[k,1:2]); yield z
        xphat = self.D(z)
        xhat  = post_process(xphat, params)
        yield xhat

class DnCNN(nn.Module):
	"""
	DnCNN implementation taken from github.com/SaoYan/DnCNN-PyTorch
	"""
	def __init__(self, Co=1, Ci=1, K=17, M=64, P=3):
		super(DnCNN, self).__init__()
		pad = (P-1)//2
		layers = []
		layers.append(nn.Conv2d(Ci, M, P, padding=pad, bias=True))
		layers.append(nn.ReLU(inplace=True))

		for _ in range(K-2):
			layers.append(nn.Conv2d(M, M, P, padding=pad, bias=False))
			layers.append(nn.BatchNorm2d(M))
			layers.append(nn.ReLU(inplace=True))

		layers.append(nn.Conv2d(M, Co, P, padding=pad, bias=True))
		self.dncnn = nn.Sequential(*layers)

	def project(self):
		return

	def forward(self, y, *args, **kwargs):
		n = self.dncnn(y)
		return y-n, n

class FFDNet(DnCNN):
	""" Implementation of FFDNet.
	"""
	def __init__(self, C=1, K=17, M=64, P=3):
		super(FFDNet, self).__init__(Ci=4*C+1, Co=4*C, K=K, M=M, P=P)
	
	def forward(self, y, sigma_n, **kwargs):
		pad = calc_pad_2D(*y.shape[2:], 2)
		yp  = F.pad(y, pad, mode='reflect')
		noise_map = (sigma_n/255.0)*torch.ones(1,1,yp.shape[2]//2,yp.shape[3]//2,device=y.device)
		z = F.pixel_unshuffle(yp, 2)
		z = torch.cat([z, noise_map], dim=1)
		z = self.dncnn(z)
		xhatp = F.pixel_shuffle(z, 2)
		xhat  = unpad(xhatp, pad)
		return xhat, noise_map
