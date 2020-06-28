#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import numpy as np
from numpy import fft
from skimage import io
import matplotlib.pyplot as plt
import math

# 二维空间滤波器
def filter_2D(img, conv):
    for i in np.arange(1, img.shape[0] - 1):
        for j in np.arange(1, img.shape[1] - 1):
            img[i, j] = np.sum(img[i-1:i+2, j-1:j+2] * conv)


# 频域高斯滤波
def gaussian_f(img):
    M,N = img.shape
    m = (M+1)/2
    n = (N+1)/2
    F = fft.fftshift(fft.fft2(img))
    H = np.zeros((M,N))
    G = np.zeros((M,N))
    for i in range(M):
        for j in range(N):
            d = (i - m)*(i - m) + (j - n)*(j - n)
            H[i, j] = math.exp(-d ** (10/11)/1800)

    G = H * F
    F00 = fft.ifft2(fft.ifftshift(G)).astype(np.float)
    return F00, H



def rand_noise(img, q=0.503):  # q > 0.5
    m,n = img.shape
    res = img.copy()
    print(m, n)
    array1 = np.random.rand(m * n).reshape(m, n)
    array2 = np.random.rand(m * n).reshape(m, n)
    array1 *= q
    array1 = np.around(array1)
    array2 = (np.around(array2) * 255).astype(np.ubyte)  # 0/255 椒盐噪声

    res[array1 > 0] = array2[array1 > 0]
    return res


def getSNR(noise, img):
    nf = fft.fftshift(fft.fft2(noise))
    ns = fft.fftshift(fft.fft2(img))
    return (np.abs(ns) ** 2) / (np.abs(nf) ** 2)


def wiener(img, H):

    K = 0.15
    F = fft.fftshift(fft.fft2(img))
    HH = np.abs(H) ** 2

    # X = F / (H+1e-3) * (HH / (HH + K)) # 直接逆滤波

    X = F / H * (HH / (HH + K))  # wiener

    m,n = X.shape
    for i in range(m):
        for j in range(n):
            if math.sqrt((i - m/2)**2 + (j - n/2)**2) >= 100:
                X[i, j] = F[i, j]

    X00 = fft.ifft2(fft.ifftshift(X)).astype(np.float)
    return X00


# 原图
img = io.imread("gray.jpg")
plt.subplot(131)
plt.imshow(img)

# 频域高斯滤波
img2, H = gaussian_f(img)

# 添加椒盐噪声
img3 = rand_noise(img2)

# 求信噪比
noise = img3 - img2
SNR = getSNR(noise, img2)
plt.subplot(132)
plt.imshow(img3)


# 维纳滤波
img_reverse = wiener(img3, H)
plt.subplot(133)
plt.imshow(img_reverse)
plt.show()
img_reverse[img_reverse >= 255] = 255
img_reverse[img_reverse <= 0] = 0
io.imsave("img2process.jpg", img3.astype(np.ubyte))
io.imsave("imgprocessed.jpg", img_reverse.astype(np.ubyte))