from AppleStore_Milestone2 import *
from AppleStore_DecissionTreeClassifier import *
from AppleStore_RandomForestClassifier import *
from AppleStore_Adaboost import *
from AppleStore_KNN import *
from AppleStore_GradientBoostingClassifier import *
from AppleStore_ExtraTreeClassifier import *
from AppleStore_LogisticRegtession import *
from AppleStore_SVMKernelsClassifier import *
from AppleStore_RunNewTestDataClassifier import *
import numpy as np
import joblib
#from sklearn.externals import joblib
joblib_file=[]
# Save RL_Model to file in the current working directory
joblib_file=np.array(['joblib_DecisionTreeClassifierModel.pkl'])
joblib.dump(DecisionTreeClassifierModel,joblib_file[0])

joblib_file =np.append(joblib_file,["joblib_GradientBoostingClassifierModel.pkl"],axis=0)
joblib.dump(GradientBoostingClassifierModel, joblib_file[1])

joblib_file =np.append(joblib_file,["joblib_RandomForestClassifierModel.pkl"],axis=0)
joblib.dump(RandomForestClassifierModel, joblib_file[2])

joblib_file =np.append(joblib_file,["joblib_ExtraTreeClassifierModel.pkl"],axis=0)
joblib.dump(ExtraTreeClassifierModel, joblib_file[3])

joblib_file =np.append(joblib_file,["joblib_AdaBoostClassifierModel.pkl"],axis=0)
joblib.dump(AdaBoostClassifierModel, joblib_file[4])

joblib_file =np.append(joblib_file,["joblib_KNN_ClassifierModel.pkl"],axis=0)
joblib.dump(KNeighborsClassifierModel, joblib_file[5])


'''
joblib_file =np.append(joblib_file,["joblib_BaggingClassifierModel.pkl"],axis=0)
joblib.dump(BaggingClassifierModel, joblib_file[6])
'''

joblib_file =np.append(joblib_file,["joblib_LogisticRegressionModel.pkl"],axis=0)
joblib.dump(LogisticRegressionModel, joblib_file[6])
'''
joblib_file =np.append(joblib_file,["joblib_Polynomial_SVM_Model.pkl"],axis=0)
joblib.dump(PolynomialSVModel, joblib_file[7])
'''

print('\t\t\t\t Reload & Test New Data File \t\t\t\t\n','*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
# Load from file and predict new test data
for i in range (joblib_file.shape[0]):
    loaded_model=joblib.load(joblib_file[i])
    predict=loaded_model.predict(X_newtest)
    accuracy = loaded_model.score(X_newtest, Y_newtest)
    print('the  accuracy of new test model {} :'.format(i+1),accuracy)





