# -*- coding: utf-8 -*-
"""
Created on Wed May 19 16:19:32 2021

@author: RISHBANS
"""

#Import Libraries
import pandas as pd                 # pandas is a dataframe library
import matplotlib.pyplot as plt      # matplotlib.pyplot plots data

#Read the data
df = pd.read_csv("pima-data.csv")

#Check the Correlation
df.corr()
#Delete the correlated feature
del df['skin']

#Data Molding
diabetes_map = {True : 1, False : 0}
df['diabetes'] = df['diabetes'].map(diabetes_map)

#Splitting the data
from sklearn.model_selection import train_test_split

#This will copy all columns from 0 to 7(8 - second place counts from 1)
X = df.iloc[:, 0:8]
y = df.iloc[:, 8]

split_test_size = 0.30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_test_size, random_state=42) 

#Imputing
from sklearn.impute import SimpleImputer 

#Impute with mean all 0 readings
fill_0 = SimpleImputer(missing_values=0, strategy="mean")

X_train = fill_0.fit_transform(X_train)
X_test = fill_0.fit_transform(X_test)


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
#C= Inverse of regularization strength
lr_model =LogisticRegression(C=0.7, max_iter=250)
lr_model.fit(X_train, y_train.ravel())
lr_predict_test = lr_model.predict(X_test)
print(lr_model.score(X_train,y_train))

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# K Fold =3
kfold = KFold(n_splits=3, random_state=7)

#.ravel will convert that array shape to (n, )
result = cross_val_score(lr_model, X_train, y_train.ravel(), cv=kfold, scoring='accuracy')
print(result.mean())

from sklearn.model_selection import GridSearchCV
import time

#C regularization, smaller values specify stronger regularization
#dual : Dual or primal formulation. The dual formulation is only implemented for l2 penalty with liblinear solver. 
#Prefer dual=False when n_samples > n_features

max_iter=[100,110,120,130,140]
C = [1.0,1.5,2.0,2.5]
param_grid = dict(max_iter=max_iter,C=C)
lr = LogisticRegression(penalty='l2')
grid = GridSearchCV(estimator=lr, param_grid=param_grid, cv = 3, n_jobs=-1)

start_time = time.time()
grid_result = grid.fit(X_train, y_train.ravel())
# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print("Execution time: " + str((time.time() - start_time)) + ' ms')






