# -*- coding: utf-8 -*-
"""
Created on Tue May 11 17:22:28 2021

@author: RISHBANS
"""

import pandas as pd

dataset = pd.read_csv('Fish.csv')

X = dataset.iloc[:, 2:7].values
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
split_test_size = 0.30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_test_size, random_state=42) 


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
#Check R-square on training data


from sklearn.metrics import r2_score

y_pred = linear_model.predict(X_test)
print(linear_model.score(X_test, y_test))
print(r2_score(y_test, y_pred))



