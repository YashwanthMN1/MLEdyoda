# -*- coding: utf-8 -*-
"""
Created on Sun May 23 11:16:27 2021

@author: RISHBANS
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Job_Exp.csv')
X = dataset.iloc[:, [0]].values
y = dataset.iloc[:, 1].values

# Applying the Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
dt_r = DecisionTreeRegressor(random_state = 0)
dt_r.fit(X, y)

# Predicting a new result
y_pred = dt_r.predict([[27]])
print(y_pred)

# Visualising
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'green')
plt.plot(X_grid, dt_r.predict(X_grid), color = 'red')
plt.ylabel('getting Job Chance (%)')
plt.xlabel('Years of Exp.')