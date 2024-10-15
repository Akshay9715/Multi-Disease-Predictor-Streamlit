import pickle
import pandas as pd
import numpy as np
from standardizing import *
import os


working_dir = os.path.dirname(os.path.abspath(__file__))
# Daibetes Predictor
def Daibetes_Predictor(data):
    loaded_model = pickle.load(open(f'{working_dir}/models/diabetes_model.pkl','rb'))
    test = Diabetes_Normalize(data)
    res = loaded_model.predict(test).round()
    if(res == 1):
        return "Person is Diabetic"
    else:
        return "person is Not Diabetic"

def Heart_Predictor(data):
    loaded_model = pickle.load(open(f'{working_dir}/models/Heart_disease_model.pkl','rb'))
    data = pd.DataFrame(data)
    data = data.values.astype(float)
    data = np.array(data).reshape(1,-1)
    res = loaded_model.predict(data)
    if(res == 1):
        return "Person Has Heart Disease"
    else:
        return "person Does NOT Have Heart Disease"

def Breast_Predictor(data):
    loaded_model = pickle.load(open(f'{working_dir}/models/Breast_model.pkl','rb'))
    test = Breast_Normalize(data)
    res = loaded_model.predict(test)
    if(res == 1):
        return "Person Has Breast Cancer"
    else:
        return "person Does NOT Have Breast Cancer"

def Paekinsons_Predictor(data):
    loaded_model = pickle.load(open(f'{working_dir}/models/Parkinsons_model.pkl','rb'))
    data = np.array(data).reshape(1,-1)
    res = loaded_model.predict(data)
    if(res == 1):
        return "Person Has Parkension's Disease"
    else:
        return "person Does NOT Have Parkension's Disease"