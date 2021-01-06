import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals import *
from AppleStore_Milestone2 import *

print('\t\t\t\t Random Forest Classifier Model \t\t\t\t\n','*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')

RandomForestClassifierModel = RandomForestClassifier(n_estimators = 9,random_state=0)
RandomForestClassifierModel.fit(X_train, Y_train)

y_predtrain = RandomForestClassifierModel.predict(X_train)
accuracy = np.mean(y_predtrain == Y_train)
print('Random forest accuracy train with prediction : ',accuracy)

# Predicting the Test set results
y_predtest = RandomForestClassifierModel.predict(X_test)

#print('MSE of test = ',np.sqrt(mean_squared_error(Y_test,y_pred )))

accuracy = np.mean(y_predtest == Y_test)
print('Random forest accuracy test with prediction : ',accuracy,'\n')
