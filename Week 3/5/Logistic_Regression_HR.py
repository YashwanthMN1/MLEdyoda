# -*- coding: utf-8 -*-
"""
Created on Fri May 14 08:04:16 2021

@author: RISHBANS
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
y_pred = lr.predict(X_test)
print(lr.score(X_test,y_test))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)

#Applying K Fold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
# K Fold =3
kfold = KFold(n_splits=3, random_state=7)
#.ravel will convert that array shape to (n, )
result = cross_val_score(lr, X_train, y_train.ravel(), cv=kfold, scoring='accuracy')
print(result.mean())


#Applying Grid Search
from sklearn.model_selection import GridSearchCV
import time
dual=[True,False]
max_iter=[100,110,120,130,140]
C = [1.0,1.5,2.0,2.5]
param_grid = dict(dual=dual,max_iter=max_iter,C=C)
lr = LogisticRegression(penalty='l2')
grid = GridSearchCV(estimator=lr, param_grid=param_grid, cv = 3, n_jobs=-1)

start_time = time.time()
grid_result = grid.fit(X_train, y_train.ravel())
# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print("Execution time: " + str((time.time() - start_time)) + ' ms')



#Using Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

digit_pipeline = make_pipeline(StandardScaler(), LogisticRegression())

digit_pipeline.fit(X_train, y_train)
y_pred = digit_pipeline.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

#A negative coefficient means that higher value of the corresponding feature pushes the classification more towards the negative class
print(digit_pipeline.steps[1][1].coef_)
#Find most influential feature
print(np.std(X_train, 0)*digit_pipeline.steps[1][1].coef_)

from sklearn.feature_selection import SelectKBest, f_classif
digit_pipeline = make_pipeline(StandardScaler(), SelectKBest(k=3, score_func=f_classif),
                               LogisticRegression(C=0.7, max_iter= 100))

print(digit_pipeline)

digit_pipeline.fit(X_train, y_train)
y_pred = digit_pipeline.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

from sklearn.model_selection import GridSearchCV
params = {'selectkbest__k': [1,2 ,3 ,4,5,6], 
          'logisticregression__C': [0.7, 1, 1.3, 1.5],
          'logisticregression__max_iter': [100,120, 140, 160]}

gs = GridSearchCV(digit_pipeline, param_grid=params, cv = 3)

gs.fit(X_train, y_train)

print(gs.best_params_)
print(gs.best_score_)
y_pred = gs.predict(X_test)
print(y_pred)



import numpy as np    
from sklearn.linear_model import LogisticRegression

x1 = np.random.randn(100)
x2 = 4*np.random.randn(100)
x3 = 0.5*np.random.randn(100)
y = (3 + x1 + x2 + x3 + 0.2*np.random.randn()) > 0
X = np.column_stack([x1, x2, x3])

m = LogisticRegression()
m.fit(X, y)

# The estimated coefficients will all be around 1:
print(m.coef_)

# Those values, however, will show that the second parameter
# is more influential
print(np.std(X, 0)*m.coef_)


