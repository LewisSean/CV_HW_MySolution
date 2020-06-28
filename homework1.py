#!/usr/bin/env python
# _*_ coding:utf-8 _*_
from skimage import io
import numpy as np


def getPerspectMat(src, dst):  # src/dst (4,2)
    solverMat = np.mat(np.zeros((8, 8)))
    for i in np.arange(0,8,2):
        j = int(i/2)
        solverMat[i, 0] = src[j,0]
        solverMat[i, 1] = src[j, 1]
        solverMat[i,2] = 1
        solverMat[i,6] = - src[j,0] * dst[j,0]
        solverMat[i,7] = - src[j,1] * dst[j,0]
        solverMat[i+1, 3] = src[j,0]
        solverMat[i+1, 4] = src[j, 1]
        solverMat[i+1,5] = 1
        solverMat[i+1,6] = - src[j,0] * dst[j,1]
        solverMat[i+1,7] = - src[j,1] * dst[j,1]
    Y = np.array(dst.flat).T
    matA = np.dot(solverMat.I, Y).flat
    matA = np.append(matA, 1)
    return np.mat(matA.reshape(3, 3))


def transformImg(img, transMat):
    m, n = img.shape
    # print(m,n)  # 383 510
    mat_4points = np.mat([[0, 0, 1], [0, n-1, 1], [m-1, 0, 1], [m-1, n-1, 1]]).T
    dst_4points = transMat * mat_4points
    # print(dst_4points)
    dst_4points = dst_4points / dst_4points[2]
    print(dst_4points)
    # print(mat_4points)
    min_x = np.min(dst_4points[0])
    max_x = np.max(dst_4points[0])
    min_y = np.min(dst_4points[1])
    max_y = np.max(dst_4points[1])
    h = np.round(max_x - min_x)
    w = np.round(max_y - min_y)
    print(h, w)   # 283 881
    dst = np.zeros((int(h), int(w)))

    sum = 0

# 此处赋值一定要注意i，j的对应关系
    for i in np.arange(h):
        for j in np.arange(w):
            x = min_x + i
            y = min_y + j
            # x = j
            # y = i
            src_point = transMat.I * np.mat([[x], [y], [1]])
            raw = (src_point[0:2, :] / src_point[2, 0]).flat
            if 0 <= raw[1] <= n-1 and 0 <= raw[0] <= m-1:
                dst[int(i), int(j)] = img[int(np.round(raw[0])), int(np.round(raw[1]))]
                sum += 1

    print(sum)
    return dst


img = io.imread("gray.jpg")
m, n = img.shape
fixp_src = np.array([[0, 0], [m, n], [m, 0], [0, n]])
fixp_dst = np.array([[0, int(n/4)], [int(1.5*m), int(n/2)], [int(m/2), 0], [int(0.5*m), n]])
perspective_mat = getPerspectMat(fixp_src, fixp_dst)
print(perspective_mat)
res = transformImg(img, perspective_mat)
io.imsave("gray_result.jpg", res.astype(np.ubyte))
print(" OK ")


