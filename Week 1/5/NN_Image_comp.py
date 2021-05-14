# -*- coding: utf-8 -*-
"""
Created on Sun May  2 12:51:04 2021

@author: RISHBANS
"""


from skimage.io import imread,imshow
imshow('car.jpg')

img = imread('car.jpg')
print(img.shape)

#Image processing needs them to converted in scale of 0-1
print(img)
img = img/255
print(img)
print(img.shape)

#Similar color pixel belongs to same cluster
#The centroid of the cluster will be representing the entire cluster

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

kmeans = KMeans(n_clusters=6)
img_tf = img.reshape(1280*1920,3)
print(img_tf.shape)
kmeans.fit(img_tf)
kmeans.cluster_centers_
plt.imshow(kmeans.cluster_centers_)

#Replace color of each pixel by its centroid's color
img_com = kmeans.cluster_centers_[[kmeans.labels_]]

imshow(img_com.reshape(1280,1920,3))





