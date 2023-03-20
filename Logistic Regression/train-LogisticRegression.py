import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from LogisticRegression import LogisticRegression

bc = datasets.load_breast_cancer()
X,y = bc.data,bc.target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=7829)


lreg = LogisticRegression()
lreg.fit(X_train,y_train)

y_pred = lreg.predict(X_test)


def accuracy(y_pred,y_test):
    return np.sum(y_pred==y_test)/ len(y_test)


acc = accuracy(y_pred,y_test)
print(acc)
