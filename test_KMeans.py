from skimage import io
import numpy as np


def load_img(file_path):
    img = io.imread(file_path)
    img = img.astype(np.float) / 256.0
    return img.reshape(img.shape[0]*img.shape[1],3)


def randCent(img, k):
    base = np.min(img, axis=0)
    range = np.max(img, axis=0) - base
    centers = base * np.ones((k, 3)) + range * np.random.rand(k, 3)
    return centers



def kmeans(img, k, centers):
    m, n = np.shape(img)
    clusterMap = np.zeros((m,))
    flag = True
    maxNum = 15
    threshold = 1/254
    num = maxNum
    while flag and num:
        num -= 1
        print("round: ",maxNum - num)
        vecD = np.zeros((m,k))
        for i in np.arange(k):
            vecD[:,i] = np.sum((img - centers[i]) * (img - centers[i]), axis=1)
        for i in np.arange(m):
            clusterMap[i] = np.where( vecD[i,:] == np.min(vecD[i,:]))[0][0]

        # print(np.max(clusterMap), np.min(clusterMap))
        print(np.bincount(clusterMap.astype(np.int).flat))
        flag = False
        for i in np.arange(k):
            if not np.sum(clusterMap == i):
                continue
            aver = np.average(img[np.where(clusterMap == i)], axis=0)
            diff = np.fabs(aver - centers[i])
            if np.sum(diff) < threshold*3:
                continue
            else:
                flag = True
                for i in np.arange(n):
                    if diff[i] > threshold:
                        centers[i] = aver[i]

    return clusterMap, centers


k = 10
# 1、导入数据
print("---------- 1.load data ------------")
img = load_img("imgs\\pool.png")
# 2、利用kMeans++聚类
print("---------- 2.run kmeans++ ------------")


centers = randCent(img, k)
clusterMap, centers = kmeans(img, k, centers)

print("---------- 3.finish and save result ------------")
im = io.imread("imgs\\pool.png")
m, n, _ = im.shape
pic_new = np.zeros((m*n,3))
for i in np.arange(k):
    pic_new[np.where(clusterMap == i)] = centers[i].copy()

pic_new = np.around(pic_new.reshape((m, n, 3)) * 255).astype(np.ubyte)
io.imsave("KMeans_result.jpg", pic_new)


