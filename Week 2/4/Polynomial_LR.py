# -*- coding: utf-8 -*-
"""
Created on Sun May  9 15:10:23 2021

@author: RISHBANS
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Company_Performance.csv')
X = dataset.iloc[:, [0]].values
y = dataset.iloc[:, 1].values

# Fitting Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
#poly_reg.fit(X_poly, y)
lin_reg_poly = LinearRegression()
lin_reg_poly.fit(X_poly, y)
y_pred = lin_reg_poly.predict(X_poly)


# Visualising -> Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Size of Company (Linear Regression)')
plt.xlabel('No. of Year in Operation ')
plt.ylabel('No. of Emp')
plt.show()



# Visualising -> Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, y_pred, color = 'blue')
plt.title('Size of Company (Polynomial Regression)')
plt.xlabel('No. of Year in Operation')
plt.ylabel('No. of Emp')
plt.show()