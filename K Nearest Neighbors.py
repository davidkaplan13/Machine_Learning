"""
K Nearest Neighbors is a simple and effective machine learning classification algorithm
K is a number you can choose, and then neighbors are the data points from known data. We're looking for any number of the "nearest" neighbors.
Let's say K = 3, so then we're looking for the two closest neighboring points.

Applying K Nearest Neighbors to Breast Cancer Data
"""
import numpy as np
from sklearn import preprocessing,neighbors
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('/Users/davidkaplan/Desktop/breast-cancer-wisconsin.txt')
# Replacing any missing data noted by ? to -99999.
# This value is recognised as an outlier
df.replace('?', -99999, inplace=True)
df.drop(['id'],1,inplace=True)

X = np.array(df.drop(['class'] ,1)) #Features
y = np.array(df['class'])           #Labels

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2) #Shuffle and sort Data
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1]]) #No ID AND CLASS
example_measures = example_measures.reshape(len(example_measures),-1)
predict = clf.predict(example_measures)
print(predict)
