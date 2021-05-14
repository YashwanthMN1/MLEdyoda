# -*- coding: utf-8 -*-
"""
Created on Sun May  9 12:08:40 2021

@author: RISHBANS
"""

import pandas as pd
exam_data = pd.read_csv('exams.csv')
exam_data.head(5)

#1. data standardization
from sklearn import preprocessing
#exam_data[['math score']] = exam_data[['math score']].astype(float)
#exam_data[['reading score']] = exam_data[['reading score']].astype(float)
#exam_data[['writing score']] = exam_data[['writing score']].astype(float)
exam_data[['math score']] = preprocessing.scale(exam_data[['math score']])
exam_data[['reading score']] = preprocessing.scale(exam_data[['reading score']])
exam_data[['writing score']] = preprocessing.scale(exam_data[['writing score']])
exam_data.head(5)

#df_plot = exam_data[['math score', 'reading score', 'writing score']].copy()
#df_plot.plot.kde()

#2. Label Encoding
le = preprocessing.LabelEncoder()
exam_data['gender'] = le.fit_transform(exam_data['gender'].astype(str))
exam_data.head()
print(le.classes_)

#3. One Hot Encoding
exam_data = pd.get_dummies(exam_data, columns=['race/ethnicity','parental level of education', 'lunch', 
                                               'test preparation course'])
exam_data.head(5)

######################################
X = exam_data.iloc[:, 1:18].values
y = exam_data.iloc[:, 0].values

from sklearn.model_selection import train_test_split
split_test_size = 0.30

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_test_size, random_state=42) 


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 11)
#ravel to convert from 2D to 1D
classifier.fit(X_train, y_train.ravel())
y_ravel = y_train.ravel()

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred, target_names=["yes", "no"]))

print(cm)
print(classifier.score(X_test,y_test))

