# -*- coding: utf-8 -*-
"""
Created on Sun May  9 14:59:09 2021

@author: RISHBANS
"""

# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Company_Profit.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

#Check R-square on training data
print(linear_model.score(X_train, y_train))

# Predicting the Test set results
y_predict = linear_model.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'green')
plt.plot(X_train, linear_model.predict(X_train), color = 'red')
plt.title('Training Set -> Startup_Yrs_Operation Vs Profit')
plt.xlabel('Startup_Yrs_Operation')
plt.ylabel('Profit')
plt.show()

# Comapre the Values 
#from mpl_toolkits.mplot3d import Axes3D
x_serial = list(range(1, len(y_predict) + 1))
plt.scatter(x_serial, y_predict, color = 'red')
plt.scatter(x_serial, y_test, color = 'blue')
plt.title('y_pred(red) vs y_test(blue)')
plt.xlabel('Serial Number')
plt.ylabel('Profit')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'black')
plt.plot(X_train, linear_model.predict(X_train), color = 'red')
plt.title('Test Set -> Startup_Yrs_Operation Vs Profit')
plt.xlabel('Startup_Yrs_Operation')
plt.ylabel('Profit')
plt.show()