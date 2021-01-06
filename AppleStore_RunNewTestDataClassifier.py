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
Y_newtest=create_labels(target_newtest,Y_newtest)
Y_newtest=np.array(Y_newtest)
#print('new y',Y_newtest)

loaded_model=joblib.load('joblib_imp_mostfreqModel.pkl')
p = loaded_model.transform(newtestdata.iloc[:, 4:7])
print(newtestdata.iloc[0,:])
newtestdata.iloc[:, 4] = p[:, 0]
newtestdata.iloc[:, 5] = p[:, 1]
newtestdata.iloc[:, 6] = p[:, 2]

loaded_model=joblib.load('joblib_label_encoderModel.pkl')
newtestdata['ver']=loaded_model.transform(newtestdata['ver'])
newtestdata['prime_genre']=loaded_model.transform(newtestdata['prime_genre'])

loaded_model=joblib.load('joblib_hot_encoderModel.pkl')
#newtestdata =loaded_model.transform(newtestdata)

newtestdata = newtestdata.iloc[:, :newtestdata.shape[1] - 1]

loaded_model=joblib.load('joblib_imp_mean_dataModel.pkl')
temp_mean = loaded_model.transform(newtestdata.iloc[:, :3])
newtestdata.iloc[:, :3] = temp_mean

temp_mean = loaded_model.transform(newtestdata.iloc[:, 1:4])
newtestdata.iloc[:, 1:4] = temp_mean

temp_mean = loaded_model.transform(newtestdata.iloc[:, 10:])
newtestdata.iloc[:, 10:] = temp_mean

newtestdata = featureScaling(np.array(newtestdata))
newtestdata = pd.DataFrame(newtestdata)

X_newtest=pd.DataFrame()
X_newtest=createXData(X_newtest,newtestdata)
X_newtest=np.array(X_newtest)

print('new X \n',X_newtest)
print('new Y \n',Y_newtest)


