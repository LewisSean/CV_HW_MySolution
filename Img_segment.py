#!/usr/bin/env python
# _*_ coding:utf-8 _*_
from skimage.segmentation import slic,mark_boundaries
from skimage import io, color, filters, morphology
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.cluster import KMeans


def getEdge(segments,label1,label2,selem=0):
    if label1 == label2:
        raise ValueError('label1 == label2!')

    conv = np.zeros((3,3))
    if selem == 0:  # 四连通
        conv[np.array([0,1,1,1,2]), np.array([1,0,1,2,1])] = 1
    else: conv[::1] = 1  # 八连通

    mask1 = (segments == label1)
    mask2 = (segments == label2)
    mask1_dila = morphology.binary_dilation(mask1, conv)
    edge2 = ((mask1_dila & mask2) == True).copy()
    mask2_dila = morphology.binary_dilation(mask2, conv)
    edge1 = ((mask2_dila & mask1) == True).copy()

    return edge1, edge2


def isAreasSame(img,segments,sobel,label1,label2):
    if img[segments == label2].shape[0] == 0 or img[segments == label1].shape[0] == 0:
        return False
    average1_rgb = np.sum(img[segments == label1], axis=0) / img[segments == label1].shape[0]
    average2_rgb = np.sum(img[segments == label2], axis=0) / img[segments == label2].shape[0]
    sum1 = 0
    for i in average1_rgb - average2_rgb:
        sum1 += math.fabs(i)
    if sum1 <= 8:
        return True
    elif 8 < sum1 < 25:
        edge1, edge2 = getEdge(segments,label1,label2,selem=0)
        return isEdgesConnected(sobel,edge1,edge2,method=0)
    else: return False


def isEdgesConnected(sobel,edge1,edge2,method=0):
    if method == 0:
        if np.sum(edge1) == 0 or np.sum(edge2) == 0: return False
        grad1 = np.median(sobel[edge1].flat)
        grad2 = np.median(sobel[edge2].flat)
        if np.sum(edge1) >= 2 and np.sum(edge2) >= 2 and math.fabs(grad1 - grad2) < 7 / 255:
            return True
        else: return False
    elif method == 1:
        pass


def mergeArea(segments, label1, label2):
    segments[segments == label2] = label1


def divideArea(segments, gray):
    lists = []
    for i in np.arange(max(segments.flat) + 1):
        histogram = np.bincount(gray[segments == i].flat)
        km = KMeans(n_clusters=2)
        km.fit(histogram.reshape(-1,1))
        index1 = int(km.cluster_centers_[0][0])
        index2 = int(km.cluster_centers_[1][0])
        sum1 = np.sum(histogram[np.where(histogram == index1)])

        sum2 = np.sum(histogram[np.where(histogram == index2)])
        if math.fabs(index2 - index1) >= 90 and sum1 != 0 and sum2 != 0 and sum1 / sum2 > 1/5 and sum1 / sum2 < 5 :
            print('divide')
            lists.append([i, int((index1 + index2)/2)])

    for i in lists: print(i)

    for i in np.arange(len(lists) - 1,-1, -1):
        segments[segments > lists[i][0]] += 1
        mask1 = mask2 = np.zeros(segments.shape,dtype=bool)
        mask1[np.where(segments == lists[i][0])] = 1
        mask2[np.where(gray > lists[i][1])] = 1
        mask1 = mask1 * mask2
        segments[mask1] += 1
        print("OK")


img = io.imread("imgs\\pool.png")
print(img.shape)

segments = slic(img, n_segments=800,compactness=10, enforce_connectivity=True)


gray = color.rgb2gray(img)
gray1 = np.around(gray * 255).astype(np.int)
sobel = filters.sobel(gray)  # both edges

print(segments.max())  # 连通数-1
print(segments.min())  # 0
area=np.bincount(segments.flat)
out1=mark_boundaries(img,segments)

sum_merge = 0
for i in np.arange(segments.max()):
    maxCon = i
    for j in np.arange(1, min(segments.max() + 1 - i, int(math.sqrt(area.shape[0]) + 5))):

         if isAreasSame(img, segments, sobel, i, i + j):
            print("merge:","area",i,"and",i+j)
            maxCon = i + j
            mergeArea(segments, i, i + j)
            sum_merge +=1
    segments[segments == i] = maxCon

print("总的合并次数：", sum_merge)

area = np.bincount(segments.flat)
out2 = mark_boundaries(segments, segments)

tmp = segments.astype(np.float) / np.max(segments)
tmp = np.power(tmp, 0.5)

plt.subplot(121)
plt.imshow(tmp)
plt.subplot(122)
plt.imshow(out2)
plt.show()
