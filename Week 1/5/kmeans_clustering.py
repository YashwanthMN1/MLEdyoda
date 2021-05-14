# -*- coding: utf-8 -*-
"""
Created on Fri May  7 10:49:37 2021

@author: RISHBANS
"""

import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3, 4]].values

#fit : Creates an similarity matrix for X using the selected similarity, then applies clustering to this similarity matrix
#fit_predict : Performs clustering on X and returns cluster labels
from sklearn.cluster import KMeans
k_means = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
k_means.fit(X)
print(k_means.labels_)

k_means_1 = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
print(k_means_1.fit_predict(X))




# Using the elbow method to find the optimal number of clusters
wcss = []
for i in range(1, 11):
    k_means = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    k_means.fit_predict(X)
    wcss.append(k_means.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()