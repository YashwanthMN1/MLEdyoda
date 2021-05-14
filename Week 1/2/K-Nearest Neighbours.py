# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 14:45:05 2020

@author: rishbans
"""
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values
X = X.astype(float)
y = y.astype(float)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#https://machine-arena.blogspot.com/2020/04/standardscaler-why-fittransform-for.html
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 11)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred, target_names=["yes", "no"]))

print(cm)

#Choose correct value of K
from matplotlib.colors import ListedColormap
import numpy as np
error_value = []
for i in range(1,25):
 classifier = KNeighborsClassifier(n_neighbors=i)
 classifier.fit(X_train,y_train)
 i_pred = classifier.predict(X_test)
 #error_value = total false prediction/ total number of prediction
 error_value.append(np.mean(i_pred != y_test))
 
print(error_value)

plt.figure(figsize=(10,6))
plt.plot(range(1,25),error_value,color='green', linestyle='dashed', 
         marker='o',markerfacecolor='orange', markersize=10)
plt.title('Error Value vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Value')
print("Minimum error:-",min(error_value),"at K =",error_value.index(min(error_value)) + 1)


from matplotlib.colors import ListedColormap
import numpy as np
#Define Variables
clf = classifier
h = 0.01
X_plot, z_plot = X_train, y_train 

#Standard Template to draw graph
x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh
Z = clf.predict(np.array([xx.ravel(), yy.ravel()]).T)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z,
             alpha = 0.7, cmap = ListedColormap(('blue', 'red')))


for i, j in enumerate(np.unique(z_plot)):
    plt.scatter(X_plot[z_plot == j, 0], X_plot[z_plot == j, 1],
                c = ['blue', 'red'][i], cmap = ListedColormap(('blue', 'red')), label = j)
   #X[:, 0], X[:, 1] 
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title('K Nearest Neighbours')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()

plt.show()

# Graph for Test Data
from matplotlib.colors import ListedColormap
import numpy as np
#Define Variables
clf = classifier
h = 0.01
X_plot, z_plot = X_test, y_test

#Standard Template to draw graph
x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh
Z = clf.predict(np.array([xx.ravel(), yy.ravel()]).T)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z,
             alpha = 0.7, cmap = ListedColormap(('blue', 'red')))


for i, j in enumerate(np.unique(z_plot)):
    plt.scatter(X_plot[z_plot == j, 0], X_plot[z_plot == j, 1],
                c = ['blue', 'red'][i], cmap = ListedColormap(('blue', 'red')), label = j)
   #X[:, 0], X[:, 1] 
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title('K Nearest Neighbours')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()

plt.show()

