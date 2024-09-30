import torch
import torch.nn.functional as F
import numpy as np

def pre_process(x, stride, mask=1):
    """ image preprocessing: stride-padding and mean subtraction.
    """
    params = []
    # mean-subtract
    if torch.is_tensor(mask):
        xmean = x.sum(dim=(1,2,3), keepdim=True) / mask.sum(dim=(1,2,3), keepdim=True)
    else:
        xmean = x.mean(dim=(1,2,3), keepdim=True)
    x = mask*(x - xmean)
    params.append(xmean)
    # pad signal for stride
    pad = calc_pad_2D(*x.shape[2:], stride)
    x = F.pad(x, pad, mode='reflect')
    if torch.is_tensor(mask):
        mask = F.pad(mask, pad, mode='reflect')
    params.append(pad)
    return x, params, mask

def post_process(x, params):
    """ undoes image pre-processing given params
    """
    # unpad
    pad = params.pop()
    x = unpad(x, pad)
    # add mean
    xmean = params.pop()
    x = x + xmean
    return x

def calc_pad_1D(L, M):
    """ Return pad sizes for length L 1D signal to be divided by M
    """
    if L%M == 0:
        Lpad = [0,0]
    else:
        Lprime = np.ceil(L/M) * M
        Ldiff  = Lprime - L
        Lpad   = [int(np.floor(Ldiff/2)), int(np.ceil(Ldiff/2))]
    return Lpad

def calc_pad_2D(H, W, M):
    """ Return pad sizes for image (H,W) to be divided by size M
    (H,W): input height, width
    output: (padding_left, padding_right, padding_top, padding_bottom)
    """
    return (*calc_pad_1D(W,M), *calc_pad_1D(H,M))

def conv_pad(x, ks, mode):
    """ Pad a signal for same-sized convolution
    """
    pad = (int(np.floor((ks-1)/2)), int(np.ceil((ks-1)/2)))
    return F.pad(x, (*pad, *pad), mode=mode)

def unpad(I, pad):
    """ Remove padding from 2D signalstack"""
    if pad[3] == 0 and pad[1] > 0:
        return I[..., pad[2]:, pad[0]:-pad[1]]
    elif pad[3] > 0 and pad[1] == 0:
        return I[..., pad[2]:-pad[3], pad[0]:]
    elif pad[3] == 0 and pad[1] == 0:
        return I[..., pad[2]:, pad[0]:]
    else:
        return I[..., pad[2]:-pad[3], pad[0]:-pad[1]]

def pre_process_3d(x, stride, mask=1):
    """ 3D video preprocessing: stride-padding and mean subtraction.
    """
    params = []
    # mean-subtract
    if torch.is_tensor(mask):
        xmean = x.sum(dim=(1,2,3,4), keepdim=True) / mask.sum(dim=(1,2,3,4), keepdim=True)
    else:
        xmean = x.mean(dim=(1,2,3,4), keepdim=True)
    x = mask * (x - xmean)
    params.append(xmean)
    # pad signal for stride
    pad = calc_pad_3D(*x.shape[2:], stride)
    x = F.pad(x, pad, mode='reflect')
    if torch.is_tensor(mask):
        mask = F.pad(mask, pad, mode='reflect')
    params.append(pad)
    return x, params, mask

def post_process_3d(x, params):
    """ Undoes 3D video pre-processing given params
    """
    # unpad
    pad = params.pop()
    x = unpad_3d(x, pad)
    # add mean
    xmean = params.pop()
    x = x + xmean
    return x

def calc_pad_3D(D, H, W, M):
    """ Return pad sizes for 3D data (D, H, W) to be divided by size M
    (D, H, W): input depth, height, width
    output: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
    """
    pad_w = calc_pad_1D(W, M)
    pad_h = calc_pad_1D(H, M)
    pad_d = calc_pad_1D(D, M)
    return (*pad_w, *pad_h, *pad_d)  # (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)

def unpad_3d(I, pad):
    """ Remove padding from 3D signal stack
    pad: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
    """
    pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back = pad
    if pad_back == 0 and pad_right > 0:
        return I[..., pad_front:, pad_top:-pad_bottom, pad_left:-pad_right]
    elif pad_back > 0 and pad_right == 0:
        return I[..., pad_front:-pad_back, pad_top:, pad_left:]
    elif pad_back == 0 and pad_right == 0:
        return I[..., pad_front:, pad_top:, pad_left:]
    else:
        return I[..., pad_front:-pad_back, pad_top:-pad_bottom, pad_left:-pad_right]

