# -*- coding: utf-8 -*-
"""
Created on Sun May  9 15:06:10 2021

@author: RISHBANS
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Multiple Linear Regression
# Importing the dataset
dataset = pd.read_csv("Largecap_Balancesheet.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 5].values

# Encoding categorical data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
# The first argument is an array called transformers, which is a list of tuples- Name, Transformer, Column
#remainder - by default it will drop other columns, but we want other columns
ct = ColumnTransformer([("Country", OneHotEncoder(), [4])], remainder = 'passthrough')
X = ct.fit_transform(X)

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
print(regressor.score(X_train, y_train))

x_serial = list(range(1, len(y_pred) + 1))
plt.scatter(x_serial, y_pred, color = 'red')
plt.scatter(x_serial, y_test, color = 'blue')
plt.title('y_pred(red) vs y_test(blue)')
plt.xlabel('Serial Number')
plt.ylabel('Profit')
plt.show()