from math import ceil, floor
import torch
from torch import nn
import torch.nn.functional as func
from tools.spectral_tools import gen_mtf
import numpy as np


def xcorr_torch(img_1, img_2, half_width):
    """
        A PyTorch implementation of Cross-Correlation Field computation.

        Parameters
        ----------
        img_1 : Torch Tensor
            First image on which calculate the cross-correlation. Dimensions: 1, 1, H, W
        img_2 : Torch Tensor
            Second image on which calculate the cross-correlation. Dimensions: 1, 1, H, W
        half_width : int
            The semi-size of the window on which calculate the cross-correlation

        Return
        ------
        L : Torch Tensor
            The cross-correlation map between img_1 and img_2

        """

    w = ceil(half_width)
    ep = 1e-20
    img_1 = img_1.double()
    img_2 = img_2.double()

    img_1 = func.pad(img_1, (w, w, w, w))
    img_2 = func.pad(img_2, (w, w, w, w))

    img_1_cum = torch.cumsum(torch.cumsum(img_1, dim=-1), dim=-2)
    img_2_cum = torch.cumsum(torch.cumsum(img_2, dim=-1), dim=-2)

    img_1_mu = (img_1_cum[:, :, 2*w:, 2*w:] - img_1_cum[:, :, :-2*w, 2*w:] - img_1_cum[:, :, 2*w:, :-2*w] + img_1_cum[:, :, :-2*w, :-2*w]) / (4*w**2)
    img_2_mu = (img_2_cum[:, :, 2*w:, 2*w:] - img_2_cum[:, :, :-2*w, 2*w:] - img_2_cum[:, :, 2*w:, :-2*w] + img_2_cum[:, :, :-2*w, :-2*w]) / (4*w**2)

    img_1 = img_1[:, :, w:-w, w:-w] - img_1_mu
    img_2 = img_2[:, :, w:-w, w:-w] - img_2_mu

    img_1 = func.pad(img_1, (w, w, w, w))
    img_2 = func.pad(img_2, (w, w, w, w))

    i2_cum = torch.cumsum(torch.cumsum(img_1**2, dim=-1), dim=-2)
    j2_cum = torch.cumsum(torch.cumsum(img_2**2, dim=-1), dim=-2)
    ij_cum = torch.cumsum(torch.cumsum(img_1*img_2, dim=-1), dim=-2)

    sig2_ij_tot = (ij_cum[:, :, 2*w:, 2*w:] - ij_cum[:, :, :-2*w, 2*w:] - ij_cum[:, :, 2*w:, :-2*w] + ij_cum[:, :, :-2*w, :-2*w])
    sig2_ii_tot = (i2_cum[:, :, 2*w:, 2*w:] - i2_cum[:, :, :-2*w, 2*w:] - i2_cum[:, :, 2*w:, :-2*w] + i2_cum[:, :, :-2*w, :-2*w])
    sig2_jj_tot = (j2_cum[:, :, 2*w:, 2*w:] - j2_cum[:, :, :-2*w, 2*w:] - j2_cum[:, :, 2*w:, :-2*w] + j2_cum[:, :, :-2*w, :-2*w])

    sig2_ii_tot = torch.clip(sig2_ii_tot, ep, sig2_ii_tot.max().item())
    sig2_jj_tot = torch.clip(sig2_jj_tot, ep, sig2_jj_tot.max().item())

    L = sig2_ij_tot / ((sig2_ii_tot * sig2_jj_tot) ** 0.5 + ep)

    return L


def local_corr_mask(img_in, ratio, sensor, device, kernel=8):
    """
        Compute the threshold mask for the structural loss.

        Parameters
        ----------
        img_in : Torch Tensor
            The test image, already normalized and with the MS part upsampled with ideal interpolator.
        ratio : int
            The resolution scale which elapses between MS and PAN.
        sensor : str
            The name of the satellites which has provided the images.
        device : Torch device
            The device on which perform the operation.
        kernel : int
            The semi-width for local cross-correlation computation.
            (See the cross-correlation function for more details)

        Return
        ------
        mask : PyTorch Tensor
            Local correlation field stack, composed by each MS and PAN. Dimensions: Batch, B, H, W.

        """

    I_PAN = torch.clone(torch.unsqueeze(img_in[:, -1, :, :], dim=1))
    I_MS = torch.clone(img_in[:, :-1, :, :])

    MTF_kern = gen_mtf(ratio, sensor)[:, :, 0]
    MTF_kern = np.expand_dims(MTF_kern, axis=(0, 1))
    MTF_kern = torch.from_numpy(MTF_kern).type(torch.float32)
    pad = floor((MTF_kern.shape[-1] - 1) / 2)

    padding = nn.ReflectionPad2d(pad)

    depthconv = nn.Conv2d(in_channels=1,
                          out_channels=1,
                          groups=1,
                          kernel_size=MTF_kern.shape,
                          bias=False)

    depthconv.weight.data = MTF_kern
    depthconv.weight.requires_grad = False
    depthconv.to(device)
    I_PAN = padding(I_PAN)
    I_PAN = depthconv(I_PAN)
    mask = xcorr_torch(I_PAN, I_MS, kernel)
    mask = 1.0 - mask

    return mask.float()
