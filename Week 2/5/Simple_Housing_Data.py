# -*- coding: utf-8 -*-
"""
Created on Mon May 10 10:24:45 2021

@author: RISHBANS
"""

from sklearn.datasets import california_housing
data = california_housing.fetch_california_housing()
data.data.shape
data.feature_names
data.target_names

import pandas as pd
house_data = pd.DataFrame(data.data, columns=data.feature_names)
house_data.describe()
house_data['Price'] = data.target


X = house_data.iloc[:, 0:8].values
y = house_data.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
#Check R-square on training data


from sklearn.metrics import r2_score

y_pred = linear_model.predict(X_test)
print(linear_model.score(X_test, y_test))
print(r2_score(y_test, y_pred))