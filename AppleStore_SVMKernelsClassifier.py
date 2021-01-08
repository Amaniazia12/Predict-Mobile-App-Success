from AppleStore_Milestone2 import *
from sklearn import tree
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt




print('\t\t\tSVM Classifier Model\t\t\t\n','*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
# we create an instance of SVM and fit out data.
C = 0.1  # SVM regularization parameter

LinearSVModel = svm.LinearSVC(C=C).fit(X_train, Y_train) #minimize squared hinge loss, One vs All
NonLinearSVModel = svm.SVC(kernel='rbf', gamma=0.5, C=C).fit(X_train, Y_train)
PolynomialSVModel = svm.SVC(kernel='poly', degree=2, C=C).fit(X_train, Y_train)
joblib.dump(LinearSVModel,'joblib_OneVSAllLinearSVModel.pkl')
joblib.dump(NonLinearSVModel,'joblib_NonLinearSVModel.pkl')
joblib.dump(PolynomialSVModel,'joblib_PolynomialSVModel.pkl')
title=''
for i, clf in enumerate((LinearSVModel, NonLinearSVModel, PolynomialSVModel)):

    if clf==LinearSVModel:
        title='Linear SVM Model'
    elif clf==NonLinearSVModel:
        title='NON Linear SVM Model'
    else:
        title = 'Polynomial SVM Model'
    predictions = clf.predict(X_train)
    accuracy = np.mean(predictions == Y_train)
    print('{} accuracy of train data is = '.format(title), accuracy)
    predictions = clf.predict(X_test)
    accuracy = np.mean(predictions == Y_test)
    print('{} accuracy of test data is = '.format(title),accuracy,'\n')
print('\n')


