# -*- coding: utf-8 -*-
"""
Created on Sat May  1 21:11:46 2021

@author: RISHBANS
"""
from cmath import sqrt
X = [11,22,33]
Y = [24,21,4]

print(sqrt((11-24)^2 + (22-21)^2 + (33-4)^2))



from sklearn.metrics.pairwise import euclidean_distances,cosine_distances,manhattan_distances
import numpy as np
import pandas as pd

X = [[0, 1], [1, 1]]
# distance between rows of X
print(euclidean_distances(X, X))
print(euclidean_distances(X))
# get distance to origin
print(euclidean_distances(X, [[0, 0]]))
print(np.sqrt(2))


print(manhattan_distances(X, X))
print(manhattan_distances(X, [[0, 0]]))

print(cosine_distances(X, X))


house_data = pd.read_csv('https://raw.githubusercontent.com/edyoda/data-science-complete-tutorial/master/Data/house_rental_data.csv.txt', index_col='Unnamed: 0')
house_data = house_data[['Sqft','Bedroom','Price']]
print(np.round(euclidean_distances(house_data[:5]),2))




