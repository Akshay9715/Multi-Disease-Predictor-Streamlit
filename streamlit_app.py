import streamlit as st 
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from predictors import *
from inputs import *




working_dir = os.path.dirname(os.path.abspath(__file__))
st.sidebar.title('Select Prediction Model')
rad = st.sidebar.radio('',['Home','Heart Disease Predictor','Breast Cancer Predictor','Diabetes Predictor',"Parkinson's Disease Predictor"])

if rad == 'Home':

    # Title of the project
    st.title("Multi-Disease Predictor - Machine Learning Project")

    # Overview
    st.header("Project Overview")
    st.write("""
    The **Multi-Disease Predictor** is an advanced web application that leverages machine learning models to predict the likelihood of developing four critical health conditions: **heart disease**, **breast cancer**, **Parkinson’s disease**, and **diabetes**. 
    This application provides users with a simple and intuitive interface to input their health data, which is then analyzed by different machine learning algorithms to deliver an accurate health risk assessment.
    """)

    # Display an overall image for the project
    st.image(f"{working_dir}/images/Home.webp", caption="Multi-Disease Predictor Overview", use_column_width=True)

    # Motivation
    st.header("Motivation")
    st.write("""
    Early detection of life-threatening diseases can significantly improve treatment outcomes and save lives. 
    By developing an accessible web-based tool, the goal is to empower users with preliminary risk assessments, encouraging timely medical consultations for at-risk individuals. This project combines healthcare with machine learning to predict the onset of diseases based on user data.
    """)

    # Machine Learning Models Used
    st.header("Machine Learning Models Used")

    # Heart Disease Predictor
    st.subheader("Heart Disease Predictor")
    st.write("""
    - **Algorithm**: Logistic Regression  
    - **Accuracy**: 85%  
    The heart disease predictor analyzes factors such as cholesterol levels, blood pressure, and lifestyle habits. Logistic Regression, a binary classification algorithm, is used here to estimate heart disease risk.
    """)

    # Breast Cancer Predictor
    st.subheader("Breast Cancer Predictor")
    st.write("""
    - **Algorithm**: K-Nearest Neighbors (KNN)  
    - **Accuracy**: 94%  
    This predictor evaluates data such as tumor size and other medical features. Using KNN, a simple and effective classification algorithm, the model predicts breast cancer risk with high accuracy.
    """)

    # Parkinson’s Disease Predictor
    st.subheader("Parkinson’s Disease Predictor")
    st.write("""
    - **Algorithm**: Support Vector Machine (SVM)  
    - **Accuracy**: 87%  
    SVM is used for this predictor, utilizing neurological data to classify the likelihood of Parkinson’s disease. The model helps in early detection of the disease, providing users the opportunity to seek medical attention early.
    """)

    # Diabetes Predictor
    st.subheader("Diabetes Predictor")
    st.write("""
    - **Algorithm**: Artificial Neural Network (ANN)  
    - **Accuracy**: 79%  
    The diabetes predictor assesses factors like glucose levels, BMI, and family history using an ANN model. While its accuracy is 79%, it provides a preliminary risk analysis to prompt users for further medical testing.
    """)

    # Workflow Section
    st.header("Workflow")
    st.write("""
    1. **Data Collection**: Datasets for each disease were sourced from medical repositories, including UCI and Kaggle. Features such as cholesterol levels, tumor characteristics, neurological data, and glucose levels were included.
    2. **Data Preprocessing**: The data was cleaned, normalized, and split into training and testing sets. For some datasets, feature scaling and data balancing (e.g., SMOTE for breast cancer) were applied.
    3. **Model Training**: Each machine learning model was trained on disease-specific datasets. The models include:
        - Logistic Regression (Heart Disease)
        - K-Nearest Neighbors (Breast Cancer)
        - Support Vector Machine (Parkinson’s Disease)
        - Artificial Neural Network (Diabetes)
    4. **Evaluation**: The models were evaluated based on accuracy, precision, recall, and F1-score. The accuracies achieved were 85% for heart disease, 94% for breast cancer, 87% for Parkinson’s, and 79% for diabetes.
    5. **Deployment**: The models were deployed using **Streamlit**, creating a user-friendly web interface for users to input their data and receive instant predictions.
    """)

    # Technologies Used
    st.header("Technologies and Libraries Used")
    st.write("""
    - **Python** for data manipulation and model building
    - **Scikit-Learn** for implementing KNN, SVM, and Logistic Regression models
    - **TensorFlow/Keras** for building the ANN model
    - **Pandas** and **NumPy** for data manipulation
    - **Streamlit** for creating the user interface
    - **Matplotlib** and **Seaborn** for data visualization
    """)


if rad == 'Heart Disease Predictor':


    # Title for Heart Disease Predictor
    first,middle,last = st.columns([1,4,1])
    middle.header("Heart Disease Predictor")
    st.text("")
    img,desc = st.columns(2)
    # Description
    desc.write("""
    Heart disease refers to various conditions that affect the heart's structure and function, often caused by atherosclerosis. Early detection is essential for preventing serious complications. This heart disease predictor assesses your likelihood of having heart disease based on key health metrics.
    The model uses **Logistic Regression** and has an accuracy of **85%**.
    """)

    # Display an image related to heart disease
    img.image(f"{working_dir}/images/Heart.webp", caption="Heart Disease", width=300)

    # Input fields for user health data
    data= Heart_inputs()

    # Button to Predict
    diagnosis = '' 
    try:
    # Your code that might throw an error
        if st.button('Heart Disease Result'):
            diagnosis = Heart_Predictor(data)
            st.success(diagnosis)
    except :
        if(ValueError):
            st.error('Press "Enter" after Each Input')


if rad == 'Breast Cancer Predictor':
    # Title for Breast Cancer Predictor
    first,middle,last = st.columns([1,4,1])
    middle.header("Breast Cancer Predictor")
    st.text("")
    img,desc = st.columns(2)
    # Description
    desc.write("""
    Breast cancer is one of the most common cancers among women worldwide. Early detection is crucial for effective treatment.
    This predictor uses a **K-Nearest Neighbors (KNN)** model, which has an accuracy of **94%**, to assess your likelihood of having breast cancer based on various health metrics.
    """)

    # Display an image related to breast cancer
    img.image(f"{working_dir}/images/Breast.webp", caption="Breast Cancer",width=300)

    # Input fields for user health data
    data =Breast_inputs()

    # Button to Predict
    diagnosis = ''
    # Creating Button For Prediction 
    try:
    # Your code that might throw an error
        if st.button('Breast Disease Result'):
            diagnosis = Breast_Predictor(data)
            st.success(diagnosis)
    except Exception as e:
        st.error(e)
    

    


if rad == 'Diabetes Predictor':
    # Title for Diabetes Predictor
    first,middle,last = st.columns([1,4,1])
    middle.header("Diabetes Predictor")
    st.text("")

    img,desc = st.columns(2)
    # Description
    desc.write("""
    Diabetes is a chronic health condition that affects how your body turns food into energy. Early detection and management are vital to prevent complications.
    This predictor uses an **Artificial Neural Network (ANN)** model, which has an accuracy of **79%**, to assess your likelihood of having diabetes based on various health metrics.
    """)
  
    # Display an image related to diabetes
    img.image(f"{working_dir}/images/Diabetes.webp", caption="Diabetes", width=300)

    st.write("")

    # Input fields for user health data
    data = Diabetes_inputs()
    
    diagnosis = ''
    # Creating Button For Prediction 
    try:
    # Your code that might throw an error
        if st.button('Diabetes  Result'):
            diagnosis = Daibetes_Predictor(data)
            st.success(diagnosis)
    except:
        if(ValueError):
            st.error('Press "Enter" after Each Input')


if rad == "Parkinson's Disease Predictor":
    # Title for Parkinson's Disease Predictor
    first,middle,last = st.columns([1,6,1])
    middle.header("Parkinson's Disease Predictor")
    st.text("")
    img,desc = st.columns(2)
    # Description
    desc.write("""
    Parkinson's disease is a progressive neurodegenerative disorder that affects movement. Early diagnosis can help manage symptoms more effectively.
    This predictor uses a **Support Vector Machine (SVM)** model, which has an accuracy of **87%**, to assess the likelihood of Parkinson's disease based on various health metrics.
    """)

    # Display an image related to Parkinson's disease
    img.image(f"{working_dir}/images/Perkinson.webp", caption="Parkinson's Disease",width=300)

    # Input fields for user health data
    data = Parkinsons_inputs()
    st.write("---")

    # Button to Predict
    diagnosis = ''
    try:
    # Your code that might throw an error
        if st.button('Perkinson"s Disease Result'):
            diagnosis = Paekinsons_Predictor(data)
            st.success(diagnosis)
    except Exception as e:
        st.error(e)
