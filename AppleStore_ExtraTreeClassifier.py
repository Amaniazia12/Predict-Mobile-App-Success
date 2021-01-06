from AppleStore_Milestone2 import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import ExtraTreeClassifier

print('\t\t\t Extra Tree Classifier Model \t\t\t\n','*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
ExtraTreeClassifierModel = ExtraTreeClassifier(random_state=0)
ExtraTreeClassifierModel=BaggingClassifier(ExtraTreeClassifierModel, random_state=0).fit(X_train, Y_train)

print('accuracy of training is : ',ExtraTreeClassifierModel.score(X_train, Y_train))
print('accuracy of testing is : ',ExtraTreeClassifierModel.score(X_test, Y_test),'\n')