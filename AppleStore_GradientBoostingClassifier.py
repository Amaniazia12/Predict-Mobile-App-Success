from AppleStore_Milestone2 import*
from sklearn.ensemble import GradientBoostingClassifier

print('\t\t\t Gradient Boosting Classifier Model \t\t\t\n','*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
GradientBoostingClassifierModel = GradientBoostingClassifier(n_estimators=350, learning_rate=0.5, max_depth=1, random_state=0)
GradientBoostingClassifierModel.fit(X_train, Y_train)
score = GradientBoostingClassifierModel.score(X_train, Y_train)
print("accuracy of training is ",score)
score1 = GradientBoostingClassifierModel.score(X_test, Y_test)
print("accuracy of testing is ",score1,'\n')
joblib.dump(GradientBoostingClassifierModel,'joblib_GradientBoostingClassifierModel.pkl')