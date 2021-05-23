# -*- coding: utf-8 -*-
"""
Created on Sun May 23 11:22:55 2021

@author: RISHBANS
"""

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.33, random_state = 42, shuffle = True)

# building our decision tree classifier and fitting the model
dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(X_train, y_train)

# predicting on the train and the test data and assessing the accuracies
from sklearn.metrics import accuracy_score

pred_train = dt.predict(X_train)
pred_test = dt.predict(X_test)
train_accuracy = accuracy_score(y_train, pred_train)
test_accuracy = accuracy_score(y_test, pred_test)
print('Training accuracy is: {0}'.format(train_accuracy))
print('Testing accuracy is: {0}'.format(test_accuracy))

print(dt.predict(X_test))

