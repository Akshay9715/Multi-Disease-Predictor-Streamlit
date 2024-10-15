import streamlit as st
import numpy as np 
import pandas as pd

def Breast_inputs():
    col1, col2, col3 = st.columns(3)

    with col1:
        radius_mean = st.number_input("Radius Mean", min_value=0.0)
        texture_mean = st.number_input("Texture Mean", min_value=0.0)
        perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0)
        area_mean = st.number_input("Area Mean", min_value=0.0)
        smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0)

    with col2:
        compactness_mean = st.number_input("Compactness Mean", min_value=0.0)
        concavity_mean = st.number_input("Concavity Mean", min_value=0.0)
        concave_points_mean = st.number_input("Concave Points Mean", min_value=0.0)
        symmetry_mean = st.number_input("Symmetry Mean", min_value=0.0)
        fractal_dimension_mean = st.number_input("Fractal Dimension Mean", min_value=0.0)
    
    with col3:
        radius_se = st.number_input("Radius SE", min_value=0.0)
        texture_se = st.number_input("Texture SE", min_value=0.0)
        perimeter_se = st.number_input("Perimeter SE", min_value=0.0)
        area_se = st.number_input("Area SE", min_value=0.0)
        smoothness_se = st.number_input("Smoothness SE", min_value=0.0)

    col4, col5, col6 = st.columns(3)

    with col4:
        compactness_se = st.number_input("Compactness SE", min_value=0.0)
        concavity_se = st.number_input("Concavity SE", min_value=0.0)
        concave_points_se = st.number_input("Concave Points SE", min_value=0.0)
        symmetry_se = st.number_input("Symmetry SE", min_value=0.0)
        fractal_dimension_se = st.number_input("Fractal Dimension SE", min_value=0.0)

    with col5:
        radius_worst = st.number_input("Radius Worst", min_value=0.0)
        texture_worst = st.number_input("Texture Worst", min_value=0.0)
        perimeter_worst = st.number_input("Perimeter Worst", min_value=0.0)
        area_worst = st.number_input("Area Worst", min_value=0.0)
        smoothness_worst = st.number_input("Smoothness Worst", min_value=0.0)

    with col6:
        compactness_worst = st.number_input("Compactness Worst", min_value=0.0)
        concavity_worst = st.number_input("Concavity Worst", min_value=0.0)
        concave_points_worst = st.number_input("Concave Points Worst", min_value=0.0)
        symmetry_worst = st.number_input("Symmetry Worst", min_value=0.0)
        fractal_dimension_worst = st.number_input("Fractal Dimension Worst", min_value=0.0)

    st.write("---")
    inputs = [
        3.037183e+07,
    radius_mean,
    texture_mean,
    perimeter_mean,
    area_mean,
    smoothness_mean,
    compactness_mean,
    concavity_mean,
    concave_points_mean,
    symmetry_mean,
    fractal_dimension_mean,
    radius_se,
    texture_se,
    perimeter_se,
    area_se,
    smoothness_se,
    compactness_se,
    concavity_se,
    concave_points_se,
    symmetry_se,
    fractal_dimension_se,
    radius_worst,
    texture_worst,
    perimeter_worst,
    area_worst,
    smoothness_worst,
    compactness_worst,
    concavity_worst,
    concave_points_worst,
    symmetry_worst,
    fractal_dimension_worst
]
    inputs = pd.DataFrame(inputs)
    inputs = inputs.values.astype(float)

    return inputs




def Diabetes_inputs():
    first,second = st.columns(2)
    Pregnancies = first.text_input('Enter the Number of Pregnancies')
    Glucose = second.text_input('Enter Glucose Level')

    third,fourth = st.columns(2)
    BolldPressure = third.text_input('Enter Blood Pressure Value')
    SkinThickness = fourth.text_input('Enter Skin Thickness')

    fifth,sixth = st.columns(2)
    Insulin = fifth.text_input('Enter Insulin Level')
    BMI = sixth.text_input('Enter BMI')

    seventh,eighth = st.columns(2)
    DiabetesPedigreeFunction = seventh.text_input('Enter Deabetes Pedigree Function Value')
    Age = eighth.text_input('Enter your text age')
    st.write("---")
    return ([Pregnancies,Glucose,BolldPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])


def Parkinsons_inputs():
    col1, col2 = st.columns(2)

    with col1:
        MDVP_Fo = st.number_input("MDVP:Fo(Hz) - Avg vocal fundamental frequency", min_value=0.0)
        MDVP_Fhi = st.number_input("MDVP:Fhi(Hz) - Max vocal fundamental frequency", min_value=0.0)
        MDVP_Flo = st.number_input("MDVP:Flo(Hz) - Min vocal fundamental frequency", min_value=0.0)
        MDVP_Jitter_percent = st.number_input("MDVP:Jitter(%)", min_value=0.0)
    MDVP_Jitter_Abs = st.number_input("MDVP:Jitter(Abs)", min_value=0.0)
    MDVP_RAP = st.number_input("MDVP:RAP", min_value=0.0)

    column9, column10 = st.columns(2)
    with column9:
        MDVP_PPQ = st.number_input("MDVP:PPQ", min_value=0.0)
        Jitter_DDP = st.number_input("Jitter:DDP", min_value=0.0)
    MDVP_Shimmer = st.number_input("MDVP:Shimmer", min_value=0.0)
    MDVP_Shimmer_dB = st.number_input("MDVP:Shimmer(dB)", min_value=0.0)
    with column10:
        Shimmer_APQ3 = st.number_input("Shimmer:APQ3", min_value=0.0)
        Shimmer_APQ5 = st.number_input("Shimmer:APQ5", min_value=0.0)

    with col2:
        MDVP_APQ = st.number_input("MDVP:APQ", min_value=0.0)
        Shimmer_DDA = st.number_input("Shimmer:DDA", min_value=0.0)
        NHR = st.number_input("NHR", min_value=0.0)
        HNR = st.number_input("HNR", min_value=0.0)
    col4, col5= st.columns(2)

    with col4:
        RPDE = st.number_input("RPDE", min_value=0.0)
        D2 = st.number_input("D2", min_value=0.0)
    DFA = st.number_input("DFA", min_value=0.0)

    with col5:
        spread1 = st.number_input("Spread1", min_value=0.0)
        spread2 = st.number_input("Spread2", min_value=0.0)

    PPE = st.number_input("PPE", min_value=0.0)
    inputs_parkinson = [
    MDVP_Fo,
    MDVP_Fhi,
    MDVP_Flo,
    MDVP_Jitter_percent,
    MDVP_Jitter_Abs,
    MDVP_RAP,
    MDVP_PPQ,
    Jitter_DDP,
    MDVP_Shimmer,
    MDVP_Shimmer_dB,
    Shimmer_APQ3,
    Shimmer_APQ5,
    MDVP_APQ ,
    Shimmer_DDA,
    NHR,
    HNR,
    RPDE,
    D2,
    DFA,
    spread1,
    spread2,
    PPE
]
    inputs_parkinson = pd.DataFrame(inputs_parkinson)
    inputs_parkinson = inputs_parkinson.values.astype(float)
    return inputs_parkinson

def Heart_inputs():
    first,second = st.columns(2)
    age = first.text_input('Enter your AGE')
    sex = second.text_input('Enter Your Gender("1" For male , "0" For Female)')

    third,fourth = st.columns(2)
    cp = third.text_input('Enter Chest Pain Level')
    rbp = fourth.text_input('Enter Resting Blood Pressure')

    fifth,sixth = st.columns(2)
    sc = fifth.text_input('Enter Serum Cholestoral(in mg/dl)')
    fbs = sixth.text_input('Enter Fasting Blood Sugar Value(in ml/dl)')

    seventh,eighth = st.columns(2)
    rer = seventh.text_input('Enter Resting Electrocardiographic Result Value')
    mhra = eighth.text_input('Enter Maximum Heart Rate Achived')

    ninth,tenth = st.columns(2)
    eia = ninth.text_input('Enter Exercise Induced Angina')
    op = tenth.text_input('Enter Oldpeak Value')

    slop = st.text_input('Enter the slope of the peak exercise ST segment')
    vf = st.text_input('Enter number of major vessels (0-3) colored by flourosopy')
    thal = st.text_input('Enter THAL Value')
    st.write("---")
    

    inputs =[age,sex,cp,rbp,sc,fbs,rer,mhra,eia,op,slop,vf,thal]
    

    return inputs