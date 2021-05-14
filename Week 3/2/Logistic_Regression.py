# -*- coding: utf-8 -*-
"""
Created on Fri May 14 08:04:16 2021

@author: RISHBANS
"""

import pandas as pd
dataset = pd.read_csv("HR.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 7].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Try without scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


from sklearn.linear_model import LogisticRegression
#in case of non scaled data use max_iter=400
lr = LogisticRegression()
lr.fit(X_train, y_train)

lr.score(X_test,y_test)





