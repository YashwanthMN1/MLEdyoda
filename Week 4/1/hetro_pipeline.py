# -*- coding: utf-8 -*-
"""
Created on Tue May 18 12:05:38 2021

@author: RISHBANS
"""

import pandas as pd
hr_data = pd.read_csv("HR_comma_sep.csv")
hr_data.rename(columns={'sales':'dept'},inplace=True)

X = hr_data.drop(columns=['left'])
y = hr_data.left

obj_data = X.select_dtypes(include=['object'])
print(X.dtypes)
#satisfaction_level & last_evaluation don't need preprocessing
#number_project,average_montly_hours,time_spend_company,Work_accident,promotion_last_5years need MinMaxScaler
#dept & Salary need OrdinalEncoder

float_data = X.select_dtypes(include=['float'])
int_data = X.select_dtypes(include=['int64'])

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

#cat_pipeline = make_pipeline(SimpleImputer(),OrdinalEncoder())
#SimpleImpter is for handling missing data in pipeline

obj1_pipeline = make_pipeline(OrdinalEncoder())
obj2_pipeline = make_pipeline(OneHotEncoder())
int_pipeline = make_pipeline(MinMaxScaler(), SelectKBest(k=3,score_func=f_classif))
preprocessor = make_column_transformer(
              (obj1_pipeline, ['salary']),
              (obj2_pipeline, ['dept']),
              (int_pipeline,int_data.columns),
              remainder='passthrough'
)

pipeline = make_pipeline(preprocessor, RandomForestClassifier())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print(pipeline.score(X_test, y_test))

print(pipeline.steps[0][1].transformers)

#applying grid search
from sklearn.model_selection import GridSearchCV
params = {'columntransformer__pipeline-3__selectkbest__k':[2,3]}
gs = GridSearchCV(pipeline, param_grid=params, n_jobs=4, cv=5)
gs.fit(X_train, y_train)
print(gs.best_params_)
print(gs.best_score_)
print(gs.score(X_test, y_test))


