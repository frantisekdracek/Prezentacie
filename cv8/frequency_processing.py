#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 11:49:36 2023

@author: frantisekdracek
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/Users/frantisekdracek/Docs/Modelovacie-a-renderovacie-techniky/cv8/pics/flowers.jpg', cv2.IMREAD_GRAYSCALE)
    
plt.imshow(img, cmap='gray')

img = np.array(img)

ftt = np.fft.fft2(img)
ftt = np.fft.fftshift(ftt)






magnitude = np.abs(ftt)

plt.imshow((255*magnitude)/(magnitude).max(), cmap='gray')

plt.imshow(np.log(255*magnitude)/np.log(magnitude).max(), cmap='gray')

def get_filter(size, ftt):
    s = ftt.shape
    filter1 = np.zeros(shape = s )
    filter1[int(s[0]/2) -size : int(s[0]/2) + size, int(s[1]/2) - size : int(s[1]/2) + size] = 1
    bool_mask = (filter1==1)
    return bool_mask

bool_mask = get_filter(50, ftt)
filtered_ftt = ftt
filtered_ftt[~bool_mask] =0
filtered_magnitude = np.log(np.abs(filtered_ftt))
freq_img = (255*filtered_magnitude)/(filtered_magnitude).max()
plt.imshow(freq_img, cmap='gray')

filtered_ftt_shift = np.fft.ifftshift(filtered_ftt)
img_filtered  = np.fft.ifft2(filtered_ftt_shift)

plt.imshow(np.real(img_filtered), cmap='gray')
