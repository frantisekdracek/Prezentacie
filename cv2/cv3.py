#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 15:07:05 2023

@author: frantisekdracek
"""

import numpy as np
import cv2 
import random
import matplotlib.pyplot as plt


def linearWeight(z):

    z_min, z_max = 0., 255.
    if z <= (z_min + z_max) / 2:
        return z - z_min
    return z_max - z
    

def sampleIntensities(images):
    z_min, z_max = 0, 255
    num_intensities = z_max - z_min + 1
    P = len(images)
    intensity_values = np.zeros((num_intensities, P), dtype=np.uint8)

    # Find the middle image to use as the source for pixel intensity locations
    mid_img = images[P // 2]

    for i in range(z_min, z_max + 1):
        rows, cols = np.where(mid_img == i)
        if len(rows) != 0:
            idx = random.randrange(len(rows))
            for j in range(P):
                intensity_values[i, j] = images[j][rows[idx], cols[idx]]
    return intensity_values

def computeResponseCurve(intensity_samples, log_exposures, smoothing_lambda, weighting_function):
    z_min, z_max = 0, 255
    intensity_range = 255  # difference between min and max possible pixel value for uint8
    num_samples = intensity_samples.shape[0]
    num_images = len(log_exposures)

    # NxP + [(Zmax-1) - (Zmin + 1)] + 1 constraints; N + 256 columns
    mat_A = np.zeros((num_images * num_samples + intensity_range, num_samples + intensity_range + 1), dtype=np.float64)
    mat_b = np.zeros((mat_A.shape[0], 1), dtype=np.float64)

    # 1. Add data-fitting constraints:
    k = 0
    for i in range(num_samples):
        for j in range(num_images):
            z_ij = intensity_samples[i, j]
            w_ij = weighting_function(z_ij)
            mat_A[k, z_ij] = w_ij
            mat_A[k, (intensity_range + 1) + i] = -w_ij
            mat_b[k, 0] = w_ij * log_exposures[j]
            k += 1

    # 2. Add smoothing constraints:
    for z_k in range(z_min + 1, z_max):
        w_k = weighting_function(z_k)
        mat_A[k, z_k - 1] = w_k * smoothing_lambda
        mat_A[k, z_k    ] = -2 * w_k * smoothing_lambda
        mat_A[k, z_k + 1] = w_k * smoothing_lambda
        k += 1

    # 3. Add color curve centering constraint:
    mat_A[k, (z_max - z_min) // 2] = 1

    inv_A = np.linalg.pinv(mat_A)
    x = np.dot(inv_A, mat_b)
    print(x.shape)

    g = x[0: intensity_range + 1]
    return g[:, 0]

def computeRadianceMap(images, log_exposure_times, response_curve, weighting_function):

    img_shape = images[0].shape
    img_rad_map = np.zeros(img_shape, dtype=np.float64)

    num_images = len(images)
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            g = np.array([response_curve[images[k][i, j]] for k in range(num_images)])
            w = np.array([weighting_function(images[k][i, j]) for k in range(num_images)])
            SumW = np.sum(w)
            if SumW > 0:
                img_rad_map[i, j] = np.sum(w * (g - log_exposure_times) / SumW)
            else:
                img_rad_map[i, j] = g[num_images // 2] - log_exposure_times[num_images // 2]
    return img_rad_map


path = "//Users//frantisekdracek//Docs//Modelovacie-a-renderovacie-techniky//cv2//night01"

images = []
for filename in ['01.bmp', '02.bmp', '03.bmp', '04.bmp', '05.bmp', '06.bmp', '07.bmp', '08.bmp', '09.bmp', '10.bmp', '11.bmp', '12.bmp', '13.bmp', '14.bmp']:
    images.append(cv2.imread('{}//{}'.format(path, filename)))
exposure_times = []
with open('{}//image_list.txt'.format(path)) as f:
    lines = f.readlines()
    for line in lines[1:]:
        exposure_times.append(line.split(' ')[1])
            
images_r = [img[:,:,0] for img in images] 
images_g = [img[:,:,1] for img in images] 
images_b = [img[:,:,2] for img in images] 
            
exposure_times = np.array(exposure_times, dtype=np.float32)
log_exposures = np.log(exposure_times)
image = cv2.cvtColor(images[6], cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()

S_r = sampleIntensities(images_r)
S_g = sampleIntensities(images_g)
S_b = sampleIntensities(images_b)

   
RC_r = computeResponseCurve(S_r, log_exposures, 10, linearWeight)
RC_g = computeResponseCurve(S_g, log_exposures, 10, linearWeight)
RC_b = computeResponseCurve(S_b, log_exposures, 10, linearWeight)

fig = plt.figure(constrained_layout=False,figsize=(7,7))   
plt.plot(RC_r,'r')
plt.plot(RC_g,'g')
plt.plot(RC_b,'b')
plt.show()


radianceMap_r = computeRadianceMap(images_r, log_exposures, RC_r, linearWeight)
radianceMap_g = computeRadianceMap(images_g, log_exposures, RC_g, linearWeight)
radianceMap_b = computeRadianceMap(images_b, log_exposures, RC_b, linearWeight)
fig = plt.figure('RMr')   
plt.imshow(radianceMap_r, cmap = 'Reds_r')
plt.show()
fig = plt.figure('RMg')   
plt.imshow(radianceMap_g)
plt.show()
fig = plt.figure('RMb')   
plt.imshow(radianceMap_b)
plt.show()