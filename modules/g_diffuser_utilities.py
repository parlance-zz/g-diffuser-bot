#import skimage
#from skimage.exposure import match_histograms
#from skimage import color, transform

import numpy as np

# common utility functions for g-diffuser-lib input / output processing

def fft2(data):
    if data.ndim > 2: # multiple channels
        out_fft = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex128)
        for c in range(data.shape[2]):
            c_data = data[:,:,c]
            out_fft[:,:,c] = np.fft.fft2(np.fft.fftshift(c_data),norm="ortho")
            out_fft[:,:,c] = np.fft.ifftshift(out_fft[:,:,c])
    else: # single channel
        out_fft = np.zeros((data.shape[0], data.shape[1]), dtype=np.complex128)
        out_fft[:,:] = np.fft.fft2(np.fft.fftshift(data),norm="ortho")
        out_fft[:,:] = np.fft.ifftshift(out_fft[:,:])
    
    return out_fft
   
def ifft2(data):
    if data.ndim > 2: # multiple channels
        out_ifft = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex128)
        for c in range(data.shape[2]):
            c_data = data[:,:,c]
            out_ifft[:,:,c] = np.fft.ifft2(np.fft.fftshift(c_data),norm="ortho")
            out_ifft[:,:,c] = np.fft.ifftshift(out_ifft[:,:,c])
    else: # single channel
        out_ifft = np.zeros((data.shape[0], data.shape[1]), dtype=np.complex128)
        out_ifft[:,:] = np.fft.ifft2(np.fft.fftshift(data),norm="ortho")
        out_ifft[:,:] = np.fft.ifftshift(out_ifft[:,:])
        
    return out_ifft
            
def get_gaussian(width, height, std=3.14, mode="gaussian"): # simple gaussian kernel
    window_scale_x = float(width / min(width, height))  # for non-square aspect ratios we still want a circular gaussian
    window_scale_y = float(height / min(width, height)) 
    if mode == "gaussian":
        x = (np.arange(width) / width * 2. - 1.) * window_scale_x
        kx = np.exp(-x*x * std)
        if window_scale_x != window_scale_y:
            y = (np.arange(height) / height * 2. - 1.) * window_scale_y
            ky = np.exp(-y*y * std)
        else:
            y = x; ky = kx
        return np.outer(kx, ky)
    elif mode == "linear_gradient":
        x = (np.arange(width) / width * 2. - 1.) * window_scale_x
        if window_scale_x != window_scale_y:
            y = (np.arange(height) / height * 2. - 1.) * window_scale_y
        else: y = x
        return np.clip(1. - np.sqrt(np.add.outer(x*x, y*y)) * std / 3.14, 0., 1.)
    else:
        raise Exception("Error: Unknown mode in get_gaussian: {0}".format(mode))

def convolve(data1, data2):      # fast convolution with fft
    if data1.ndim != data2.ndim: # promote to rgb if mismatch
        if data1.ndim < 3: data1 = np_img_grey_to_rgb(data1)
        if data2.ndim < 3: data2 = np_img_grey_to_rgb(data2)
    return ifft2(fft2(data1) * fft2(data2))

def gaussian_blur(data, std=3.14, mode="gaussian"):
    width = data.shape[0]
    height = data.shape[1]
    kernel = get_gaussian(width, height, std, mode=mode)
    return np.real(convolve(data, kernel / np.sqrt(np.sum(kernel*kernel))))
 
def normalize_image(data):
    normalized = data - np.min(data)
    normalized_max = np.max(normalized)
    assert(normalized_max > 0.)
    return normalized / normalized_max
 
def np_img_rgb_to_grey(data):
    if data.ndim == 2: return data
    return np.sum(data, axis=2)/3.
    
def np_img_grey_to_rgb(data):
    if data.ndim == 3: return data
    return np.expand_dims(data, 2) * np.ones((1, 1, 3))

"""
def np_img_rgb_to_hsv(data):
    return color.rgb2hsv(data)

def np_img_hsv_to_rgb(data):
    return color.hsv2rgb(data)

def hsv_blend_image(image, match_to, hsv_mask=None):
    width = image.shape[0]
    height = image.shape[1]
    
    if type(hsv_mask) != np.ndarray:
        hsv_mask = np.ones((width, height, 3))
    image_hsv = np_img_rgb_to_hsv(image)
    match_to_hsv = np_img_rgb_to_hsv(match_to)
    
    return np_img_hsv_to_rgb(image_hsv * (1.-hsv_mask) + hsv_mask * match_to_hsv)
"""