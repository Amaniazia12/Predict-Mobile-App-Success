import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from AppleStore_Milestone2 import *

print('\t\t\t KNeighbors Classifier Model \t\t\t\n','*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')

KNeighborsClassifierModel = KNeighborsClassifier(n_neighbors=75)
KNeighborsClassifierModel.fit(X_train, Y_train)

y_trainprediction = KNeighborsClassifierModel.predict(X_train)
y_testprediction = KNeighborsClassifierModel.predict(X_test)
accuracyTrain=np.mean(y_trainprediction == Y_train)
accuracyTest=np.mean(y_testprediction == Y_test)

print ("The achieved accuracy train using KNN is " + str(accuracyTrain))
print ("The achieved accuracy test using KNN is " + str(accuracyTest),'\n')

'''
error = []

# Calculating error for K values between 1 and 40
for i in range(65, 75):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, Y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != Y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(65, 75), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()
'''