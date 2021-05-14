from sklearn.datasets import load_boston
import pandas as pd
boston = load_boston()
df = pd.DataFrame(boston.data)

print(boston.feature_names)
df.columns = boston.feature_names

df['House_Price'] = boston.target

print(df.isnull().values.any())

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

#Fitting Nearest Neighbour Regression to the Training set
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error 
from math import sqrt
NN_model = KNeighborsRegressor(n_neighbors = 3)
NN_model.fit(X_train, y_train)

print(NN_model.score(X_train, y_train))
print(NN_model.score(X_test, y_test))

y_predict = NN_model.predict(X_test)
print(y_predict)

rmse = sqrt(mean_squared_error(y_test,y_predict))
print(rmse)

#Finding value of K
rmse_val = [] 
for K in range(1, 20):
    NN_model = KNeighborsRegressor(n_neighbors = K)

    NN_model.fit(X_train, y_train)  
    y_predict = NN_model.predict(X_test) 
    rmse = sqrt(mean_squared_error(y_test,y_predict)) 
    rmse_val.append(rmse) 
    print('RMSE value k= ' , K , '->', rmse)

#Elbow Curve    
curve = pd.DataFrame(rmse_val) 
curve.plot()    


