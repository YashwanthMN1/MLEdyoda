# -*- coding: utf-8 -*-
"""
Created on Sun May  2 19:00:22 2021

@author: RISHBANS
"""


from sklearn.datasets import load_iris
iris = load_iris()
feature_data = iris.data
feature_data = feature_data[:,[0,1]]
print(feature_data.shape)
print(iris.target)

import numpy as np
import pandas as pd
class MyKMeans:
    def __init__(self,n_clusters=3):
        self.n_clusters = n_clusters
        
    def my_fit(self, feature_data, centroids=[[4.8,3.2],[5.5,2.5],[7.5,3]]):
        self.feature_data = np.array(feature_data)
        self.data_centroid = []
        self.centroids = np.array(centroids)
            
    def assign_data_to_centroid(self):
        fd = self.feature_data.reshape(1,150,2)
        cd = self.centroids.reshape(1,3,2)
        distances = np.sqrt(np.sum(np.square(fd - cd),axis=2))
        self.data_centroid = np.argmin(distances,axis=1)
        self.distance_centroid = np.min(distances,axis=1)
        return self.data_centroid
    
    def recalculate_centroid(self):
        def f(d):
            res = np.round([d['F1'].sum()/d.shape[0],d['F2'].sum()/d.shape[0]],3)
            self.centroids.append(np.array(res))
            
        self.centroids = []
        print(self.data_centroid)
        print(self.feature_data[:,0])
        print(self.feature_data[:,1])
        df = pd.DataFrame({'Centroid':self.data_centroid,'F1':self.feature_data[:,0],'F2':self.feature_data[:,1]})
        print(df)
        df.groupby('Centroid').apply(f)
        self.centroids = np.array(self.centroids)  
        
    def my_cost(self):
        return self.distance_centroid.sum()
    
myKmeans = MyKMeans(n_clusters=3)
myKmeans.my_fit(feature_data)
#Current Centroid
print(myKmeans.centroids)

#Default centroids
print(myKmeans.recalculate_centroid())

