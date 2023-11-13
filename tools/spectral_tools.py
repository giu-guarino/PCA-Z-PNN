import torch
import numpy as np
from torch.nn.functional import conv2d
from torch.nn.functional import pad


def normalize_prisma(img, nbits, nbands):
    return img / (np.sqrt(nbands)*(2**nbits))

def denormalize_prisma(img, nbits, nbands):
    return img * (np.sqrt(nbands)*(2**nbits))

def mtf_kernel_to_torch(h):
    """
        Compute the estimated MTF filter kernels for the supported satellites and calculate the spatial bias between
        each Multi-Spectral band and the Panchromatic (to implement the coregistration feature).
        Parameters
        ----------
        h : Numpy Array
            The filter based on Modulation Transfer Function.
        Return
        ------
        h : Tensor array
            The filter based on Modulation Transfer Function reshaped to Conv2d kernel format.
        """

    h = np.moveaxis(h, -1, 0)
    h = np.expand_dims(h, axis=1)
    h = h.astype(np.float32)
    h = torch.from_numpy(h).type(torch.float32)
    return h

def fsamp2(hd):
    """
        Compute fir filter with window method
        Parameters
        ----------
        hd : float
            Desired frequency response (2D)
        Return
        ------
        h : Numpy array
            The fir Filter
    """
    hd = np.rot90(np.fft.fftshift(np.rot90(hd, 2)), 2)
    h = np.fft.fftshift(np.fft.ifft2(hd))
    h = np.rot90(h, 2)
    h = np.real(h)

    return h


def fir_filter_wind(f1, f2):
    """
        Compute fir filter with window method
        Parameters
        ----------
        f1 : float
            Desired frequency response (2D)
        f2 : Numpy Array
            The filter kernel (2D)
        Return
        ------
        h : Numpy array
            The fir Filter
    """

    hd = f1
    w1 = f2
    n = w1.shape[0]
    m = n
    t = np.arange(start=-(n-1)/2, stop=(n-1)/2 + 1) * 2/(n-1)
    t1, t2 = np.meshgrid(t, t)
    t12 = np.sqrt(t1 ** 2 + t2 ** 2)

    d = np.asarray(((t12 < t[0]) + (t12 > t[-1])).flatten()).nonzero()
    dd = (t12 < t[0]) + (t12 > t[-1])

    t12[dd] = 0

    w = np.interp(t12.flatten(),t, w1).reshape(t12.shape)
    w[dd] = 0
    h = fsamp2(hd) * w

    return h


def fspecial_gauss(size, sigma):
    """
        Function to mimic the 'fspecial' gaussian MATLAB function
        Parameters
        ----------
        size : Tuple
            The dimensions of the kernel. Dimension: H, W
        sigma : float
            The frequency of the gaussian filter
        Return
        ------
        h : Numpy array
            The Gaussian Filter of sigma frequency and size dimension
        """
    m, n = [(ss - 1.) / 2. for ss in size]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def nyquist_filter_generator(nyquist_freq, ratio, kernel_size):
    """
        Compute the estimeted MTF filter kernels.
        Parameters
        ----------
        nyquist_freq : Numpy Array or List
            The MTF frequencies
        ratio : int
            The resolution scale which elapses between MS and PAN.
        kernel_size : int
            The size of the kernel (Only squared kernels have been implemented).
        Return
        ------
        kernel : Numpy array
            The filter based on Modulation Transfer Function.
    """
    assert isinstance(nyquist_freq, (np.ndarray, list)), 'Error: GNyq must be a list or a ndarray'
    if isinstance(nyquist_freq, list):
        nyquist_freq = np.asarray(nyquist_freq)
        nyquist_freq = np.reshape(nyquist_freq, (1,nyquist_freq.shape[0]))

    nbands = nyquist_freq.shape[1]

    kernel = np.zeros((kernel_size, kernel_size, nbands))  # generic kerenel (for normalization purpose)
    fcut = 1 / np.double(ratio)

    for j in range(nbands):
        alpha = np.sqrt(((kernel_size - 1) * (fcut / 2)) ** 2 / (-2 * np.log(nyquist_freq[0, j])))
        H = fspecial_gauss((kernel_size, kernel_size), alpha)
        Hd = H / np.max(H)
        h = np.kaiser(kernel_size, 0.5)
        h = np.real(fir_filter_wind(Hd, h))
        if ratio != 6: # TO DO: Delete for HyperSpectral
            h = np.clip(h, a_min=0, a_max=np.max(h))
            h = h / np.sum(h)
        else:
            h = np.real(h)
        kernel[:, :, j] = h

    return kernel

def gen_mtf(ratio, sensor='none', kernel_size=41, nbands=3):
    """
        Compute the estimated MTF filter kernels for the supported satellites.
        Parameters
        ----------
        ratio : int
            The resolution scale which elapses between MS and PAN.
        sensor : str
            The name of the satellites which has provided the images.
        kernel_size : int
            The size of the kernel (Only squared kernels have been implemented).
        Return
        ------
        kernel : Numpy array
            The filter based on Modulation Transfer Function for the desired satellite.
        """
    GNyq = []

    if sensor == 'S2-10':
        GNyq = [0.275, 0.28, 0.25, 0.24]
    elif sensor == 'S2-10-PAN':
        GNyq = [0.26125] * nbands
    elif sensor == 'S2-20':
        GNyq = [0.365, 0.33, 0.34, 0.32, 0.205, 0.235]
    elif sensor == 'S2-60':
        GNyq = [0.3175, 0.295, 0.30]
    elif sensor == 'S2-60_bis':
        GNyq = [0.3175, 0.295]
    elif sensor == 'WV3':
        GNyq = [0.325, 0.355, 0.360, 0.350, 0.365, 0.360, 0.335, 0.315] ## TO REMOVE
    else:
        GNyq = [0.3] * nbands

    h = nyquist_filter_generator(GNyq, ratio, kernel_size)

    return h



def mtf(img, sensor, ratio, mode='replicate'):
    h = gen_mtf(ratio, sensor, nbands=img.shape[1])

    h = mtf_kernel_to_torch(h).type(img.dtype).to(img.device)
    img_lp = conv2d(pad(img, (h.shape[-2] // 2, h.shape[-2] // 2, h.shape[-1] // 2, h.shape[-1] // 2), mode=mode), h, padding='valid', groups=img.shape[1])

    return img_lp
