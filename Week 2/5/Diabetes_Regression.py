# -*- coding: utf-8 -*-
"""
Created on Tue May 11 11:40:02 2021

@author: RISHBANS
"""

from sklearn.datasets import load_diabetes
data = load_diabetes()
data.data.shape
data.feature_names
data.target

import pandas as pd
house_data = pd.DataFrame(data.data, columns=data.feature_names)
house_data.describe()
house_data['Progress'] = data.target

X = house_data.iloc[:, 0:10].values
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

import matplotlib.pyplot as plt
x_serial = list(range(1, len(y_pred) + 1))
plt.scatter(x_serial, y_pred, color = 'red')
plt.scatter(x_serial, y_test, color = 'blue')
plt.title('y_pred(red) vs y_test(blue)')
plt.xlabel('Serial Number')
plt.ylabel('Profit')
plt.show()



from sklearn.ensemble import GradientBoostingRegressor
reg1 = GradientBoostingRegressor(random_state=1)
reg1.fit(X_train, y_train)
y_pred = reg1.predict(X_test)
print(reg1.score(X_test, y_test))
print(r2_score(y_test, y_pred))

from sklearn.ensemble import RandomForestRegressor
reg1 = RandomForestRegressor(random_state=1)
reg1.fit(X_train, y_train)
y_pred = reg1.predict(X_test)
print(reg1.score(X_test, y_test))
print(r2_score(y_test, y_pred))
