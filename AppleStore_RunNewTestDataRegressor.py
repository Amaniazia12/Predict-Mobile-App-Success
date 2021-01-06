import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.compose import _column_transformer
from AppleStore_PreProcessing import *
from AppleStore_Milestone_cop_1 import *
import joblib


newtestdata=pd.read_csv('AppleStore_training.csv')
drop_cols=('currency','vpp_lic','track_name','id')
newtestdata=drop_columns(newtestdata, drop_cols)
newtestdata=drop_row(newtestdata)

Y_newtest=[]
Y_NTF=pd.DataFrame()
Y_NTF,Y_newtest=createYDataReg(Y_NTF,Y_newtest, newtestdata)

loadedLabelenc_model=joblib.load('joblib_label_encoderModel.pkl')
newtestdata['ver']=loadedLabelenc_model.transform(list(newtestdata['ver'].values))
newtestdata['prime_genre']=loadedLabelenc_model.transform(list(newtestdata['prime_genre'].values))

loadedonehot_model=joblib.load('joblib_hot_encoderModel.pkl')
#newtestdata =loadedonehot_model.transform(newtestdata)


#Y_newtest=np.array(Y_newtest)
X_newtest=[]
X_NTF=pd.DataFrame()
X_NTF,X_newtest=createXDataReg(X_NTF,X_newtest, newtestdata)
X_newtest=np.array(X_newtest)

newtest_temp = []

for array in Y_newtest:
 for x in array:
    newtest_temp.append(x)
Y_newtest=newtest_temp
'''
print('new X data frame\n',X_NTF)
print('new X list\n',X_newtest)
print('new Y data frame\n',Y_NTF)
print('new Y list\n',Y_newtest)
'''