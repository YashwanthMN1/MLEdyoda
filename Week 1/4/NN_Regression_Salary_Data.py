import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:, 0:2].values
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

#Fitting Nearest Neighbour Regression to the Training set
from sklearn.neighbors import KNeighborsRegressor
NN_model = KNeighborsRegressor(n_neighbors = 3)
NN_model.fit(X_train, y_train)


print(NN_model.score(X_train, y_train))
print(NN_model.score(X_test, y_test))

y_predict = NN_model.predict(X_test)
print(y_predict)

from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = []
for k in range(1,20):
    NN_model = KNeighborsRegressor(n_neighbors = k)
    NN_model.fit(X_train, y_train)
    y_predict = NN_model.predict(X_test)
    
    error = sqrt(mean_squared_error(y_test, y_predict))
    rmse.append(error)
    print(k, error)
    
graph = pd.DataFrame(rmse)    
graph.plot()















