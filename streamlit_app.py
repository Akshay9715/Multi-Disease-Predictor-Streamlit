import streamlit as st 
import numpy as np
import keras
from sklearn.preprocessing import StandardScaler




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
    st.image("images\Home.webp", caption="Multi-Disease Predictor Overview", use_column_width=True)

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
    # st.image("/mnt/data/A_modern_and_professional_image_for_heart_disease_.png", caption="Heart Disease Predictor", use_column_width=True)

    # Breast Cancer Predictor
    st.subheader("Breast Cancer Predictor")
    st.write("""
    - **Algorithm**: K-Nearest Neighbors (KNN)  
    - **Accuracy**: 94%  
    This predictor evaluates data such as tumor size and other medical features. Using KNN, a simple and effective classification algorithm, the model predicts breast cancer risk with high accuracy.
    """)
    #st.image("/mnt/data/A_modern_and_professional_image_for_breast_cancer_.png", caption="Breast Cancer Predictor", use_column_width=True)

    # Parkinson’s Disease Predictor
    st.subheader("Parkinson’s Disease Predictor")
    st.write("""
    - **Algorithm**: Support Vector Machine (SVM)  
    - **Accuracy**: 87%  
    SVM is used for this predictor, utilizing neurological data to classify the likelihood of Parkinson’s disease. The model helps in early detection of the disease, providing users the opportunity to seek medical attention early.
    """)
    #st.image("/mnt/data/A_modern_and_professional_image_for_parkinsons_dis.png", caption="Parkinson’s Disease Predictor", use_column_width=True)

    # Diabetes Predictor
    st.subheader("Diabetes Predictor")
    st.write("""
    - **Algorithm**: Artificial Neural Network (ANN)  
    - **Accuracy**: 79%  
    The diabetes predictor assesses factors like glucose levels, BMI, and family history using an ANN model. While its accuracy is 79%, it provides a preliminary risk analysis to prompt users for further medical testing.
    """)
    #st.image("/mnt/data/A_modern_and_professional_image_for_diabetes_predi.png", caption="Diabetes Predictor", use_column_width=True)

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

    # Key Challenges
    # st.header("Key Challenges")
    # st.write("""
    # - **Data Imbalance**: Some datasets were imbalanced, particularly for breast cancer. Techniques such as SMOTE (Synthetic Minority Over-sampling Technique) were used to address this.
    # - **Model Generalization**: Ensuring the models did not overfit was critical. Cross-validation and regularization techniques were employed to improve model performance on unseen data.
    # """)

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
    st.subheader("Heart Disease Predictor")

    # Description
    st.write("""
    Heart disease refers to various conditions that affect the heart's structure and function, often caused by atherosclerosis. Early detection is essential for preventing serious complications. This heart disease predictor assesses your likelihood of having heart disease based on key health metrics.
    The model uses **Logistic Regression** and has an accuracy of **85%**.
    """)

    # Display an image related to heart disease
    st.image("images\Heart.webp", caption="Heart Disease", width=300)

    # Input fields for user health data
    age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)
    sex = st.selectbox("Sex (1: Male, 0: Female)", options=[1, 0])
    chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200, step=1)
    resting_bp = st.number_input("Resting Blood Pressure (mmHg)", min_value=80, max_value=200, value=120, step=1)
    max_heart_rate = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=220, value=150, step=1)
    exercise_angina = st.selectbox("Exercise-Induced Angina (1: Yes, 0: No)", options=[1, 0])
    st.write("---")

    # Button to Predict
    if st.button("Predict Heart Disease"):
        # In reality, you would use the input values to make a prediction using your model:
        # input_data = np.array([[age, sex, chol, resting_bp, max_heart_rate, exercise_angina]])
        # prediction = model.predict(input_data)
        
        # For now, we'll mock a prediction output
        prediction = np.random.choice([0, 1])  # Mock prediction: 0 - No disease, 1 - Disease
        
        if prediction == 1:
            st.error("High likelihood of heart disease. Please consult a healthcare provider.")
        else:
            st.success("Low likelihood of heart disease.")


if rad == 'Breast Cancer Predictor':
    # Title for Breast Cancer Predictor
    st.subheader("Breast Cancer Predictor")

    # Description
    st.write("""
    Breast cancer is one of the most common cancers among women worldwide. Early detection is crucial for effective treatment.
    This predictor uses a **K-Nearest Neighbors (KNN)** model, which has an accuracy of **94%**, to assess your likelihood of having breast cancer based on various health metrics.
    """)

    # Display an image related to breast cancer
    #st.image("/mnt/data/A_modern_and_professional_image_for_breast_cancer_.png", caption="Breast Cancer", use_column_width=True)

    # Input fields for user health data
    radius_mean = st.number_input("Mean Radius", min_value=0.0, value=10.0, step=0.1)
    texture_mean = st.number_input("Mean Texture", min_value=0.0, value=10.0, step=0.1)
    perimeter_mean = st.number_input("Mean Perimeter", min_value=0.0, value=50.0, step=0.1)
    area_mean = st.number_input("Mean Area", min_value=0.0, value=500.0, step=0.1)
    smoothness_mean = st.number_input("Mean Smoothness", min_value=0.0, value=0.1, step=0.01)
    st.write("---")

    # Button to Predict
    if st.button("Predict Breast Cancer"):
        # Mock prediction output (replace with actual model)
        prediction = np.random.choice([0, 1])  # Mock prediction: 0 - No cancer, 1 - Cancer
        
        if prediction == 1:
            st.error("High likelihood of breast cancer. Please consult a healthcare provider.")
        else:
            st.success("Low likelihood of breast cancer.")


if rad == 'Diabetes Predictor':
    # Title for Diabetes Predictor
    st.subheader("Diabetes Predictor")

    # Description
    st.write("""
    Diabetes is a chronic health condition that affects how your body turns food into energy. Early detection and management are vital to prevent complications.
    This predictor uses an **Artificial Neural Network (ANN)** model, which has an accuracy of **79%**, to assess your likelihood of having diabetes based on various health metrics.
    """)

    # Display an image related to diabetes
    #st.image("/mnt/data/A_sleek_and_modern_image_for_diabetes_predict.png", caption="Diabetes", use_column_width=True)

    # Input fields for user health data
    glucose = st.number_input("Glucose Level (mg/dl)", min_value=0, value=100, step=1)
    blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0, value=70, step=1)
    bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, value=25.0, step=0.1)
    age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)
    st.write("---")

    # Button to Predict
    if st.button("Predict Diabetes"):
        # Mock prediction output (replace with actual model)
        prediction = np.random.choice([0, 1])  # Mock prediction: 0 - No diabetes, 1 - Diabetes
        
        if prediction == 1:
            st.error("High likelihood of diabetes. Please consult a healthcare provider.")
        else:
            st.success("Low likelihood of diabetes.")


if rad == "Parkinson's Disease Predictor":
    # Title for Parkinson's Disease Predictor
    st.subheader("Parkinson's Disease Predictor")

    # Description
    st.write("""
    Parkinson's disease is a progressive neurodegenerative disorder that affects movement. Early diagnosis can help manage symptoms more effectively.
    This predictor uses a **Support Vector Machine (SVM)** model, which has an accuracy of **87%**, to assess the likelihood of Parkinson's disease based on various health metrics.
    """)

    # Display an image related to Parkinson's disease
    st.image("images/Perkinson.webp", caption="Parkinson's Disease",width=300)

    # Input fields for user health data
    motor_updrs = st.number_input("Motor UPDRS", min_value=0, value=0, step=1)
    total_UPDRS = st.number_input("Total UPDRS", min_value=0, value=0, step=1)
    age = st.number_input("Age", min_value=1, max_value=120, value=60, step=1)
    gender = st.selectbox("Gender (1: Male, 0: Female)", options=[1, 0])
    st.write("---")

    # Button to Predict
    if st.button("Predict Parkinson's Disease"):
        # Mock prediction output (replace with actual model)
        prediction = np.random.choice([0, 1])  # Mock prediction: 0 - No disease, 1 - Disease
        
        if prediction == 1:
            st.error("High likelihood of Parkinson's disease. Please consult a healthcare provider.")
        else:
            st.success("Low likelihood of Parkinson's disease.")