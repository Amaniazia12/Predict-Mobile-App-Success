import numpy as np
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
import joblib
def drop_row(data):
    data.dropna(how='any', inplace=True)

    return data

def drop_columns(data,cols):
 for col in cols:

    data.drop(col, axis=1, inplace=True)

 return data


def OneHot_Encoder(data,col):

    hot_encoderModel = ce.OneHotEncoder(cols=col,use_cat_names =True )
    hot_encoderModel.fit(data)
    data =hot_encoderModel.transform(data)

    with open('joblib_hot_encoderModel.pkl','wb')as f:
     joblib.dump(hot_encoderModel,f)
     f.close()
    return data



def label_encoder(data, col):

    label_encoderModel = LabelEncoder()
    x_val = np.array(data[col].values)

    label_encoderModel.fit(list(data[col].values))
    data[col]=label_encoderModel.transform((list(data[col].values)))

    joblib.dump(label_encoderModel,'joblib_label_encoderModel.pkl')
    return data



def featureScaling(data):
    Normalized_data = np.zeros((data.shape[0], data.shape[1]));
    for i in range(data.shape[1]):
        Normalized_data[:, i] = (data[:, i] - min(data[:, i])) / (max(data[:, i]) - min(data[:, i]));
    return Normalized_data

    return Normalized_data








