import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.compose import _column_transformer
from AppleStore_PreProcessing import *
from AppleStore_Milestone2 import *
import joblib


newtestdata=pd.read_csv('AppleStore_training_classification.csv')
target_newtest=newtestdata.iloc[:,newtestdata.shape[1]-1]
#print('target new data \n',target_newtest)
drop_cols=('currency','vpp_lic','track_name','id')
newtestdata=drop_columns(newtestdata, drop_cols)

Y_newtest=[]
Y_newtest=create_classes(target_newtest,Y_newtest)
Y_newtest=np.array(Y_newtest)
#print('new y',Y_newtest)

loaded_model=joblib.load('joblib_imp_mostfreqModel.pkl')
p = loaded_model.transform(newtestdata.iloc[:, 4:7])
print(newtestdata.iloc[0,:])
newtestdata.iloc[:, 4] = p[:, 0]
newtestdata.iloc[:, 5] = p[:, 1]
newtestdata.iloc[:, 6] = p[:, 2]
#load label_encoderModel
cols=('ver','prime_genre')
label_encoderModel=joblib.load('joblib_label_encoderModel.pkl')
for col in cols:
 newtestdata.replace(label_encoderModel[col], inplace=True)

#load Hot_encoderModel
hot_encoderModel=joblib.load('joblib_hot_encoderModel.pkl')
newtestdata =hot_encoderModel.transform(newtestdata)

#remove Y
newtestdata = newtestdata.iloc[:, :newtestdata.shape[1] - 1]

#mean model
loaded_model=joblib.load('joblib_imp_mean_dataModel.pkl')
temp_mean = loaded_model.transform(newtestdata.iloc[:, :])
newtestdata.iloc[:, :] = temp_mean

#scaling
newtestdata = featureScaling(np.array(newtestdata))
newtestdata = pd.DataFrame(newtestdata)

#create features of X
X_newtest=pd.DataFrame()
X_newtest=createXData(X_newtest,newtestdata)
X_newtest=np.array(X_newtest)


