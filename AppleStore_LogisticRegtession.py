from sklearn.linear_model import LogisticRegression
import numpy as np
from AppleStore_Milestone2 import *
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection


print('\t\t\t Logistic Regression Model mult(1 vs rest)\t\t\t\n','*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')

#logistic regression multiclass using one verses rest
LogisticRegressionModel = LogisticRegression(multi_class='ovr', class_weight='auto')

LogisticRegressionModel .fit(X_train, Y_train)
predictions = LogisticRegressionModel .predict(X_train)
#Generate a confusion matrix
score = LogisticRegressionModel .score(X_train, Y_train)
print("accuracy of train is ",score)

predictions = LogisticRegressionModel .predict(X_test)
#Generate a confusion matrix
score = LogisticRegressionModel .score(X_test, Y_test)
print("accuracy of test is ",score,'\n')
joblib.dump(LogisticRegressionModel,'joblib_LogisticRegressionModel.pkl')