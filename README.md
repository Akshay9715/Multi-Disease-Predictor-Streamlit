# Multi-Disease Predictor Web Application

This project is a **Multi-Disease Predictor** web app built using **Streamlit**. The app allows users to predict the likelihood of various diseases based on input medical data. The diseases covered include:
- Heart Disease
- Breast Cancer
- Parkinson's Disease
- Diabetes

The models used for prediction in this project include **Artificial Neural Networks (ANN)**, **K-Nearest Neighbors (KNN)**, **Logistic Regression**, and **Support Vector Machine (SVM)**. Each model is trained with real-world medical datasets, providing reliable predictions for each disease.

## Live Demo

You can explore the web app and make predictions using the following link:

[Multi-Disease Predictor App](https://multi-disease-predictor-app.streamlit.app/)

## Project Overview

### Diseases and Models Used:
1. **Heart Disease**
   - Model: Logistic Regression
   - Accuracy: 85%
2. **Breast Cancer**
   - Model: K-Nearest Neighbors (KNN)
   - Accuracy: 94%
3. **Parkinson's Disease**
   - Model: Support Vector Machine (SVM)
   - Accuracy: 87%
4. **Diabetes**
   - Model: Artificial Neural Networks (ANN)
   - Accuracy: 79%

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Akshay9715/Multi-Disease-Predictor-Streamlit.git
   cd Multi-Disease-Predictor-Streamlit

2. Install the required dependencies:
     pip install -r requirements.txtInstall the required dependencies:

3. Run the Streamlit app:
     streamlit run app.py

**Technologies Used :-**
  1. Streamlit: For building the web interface
  2. Scikit-learn: For model development and evaluation
  3. Pandas: For data manipulation and analysis
  4. NumPy: For numerical computations
  5. Keras & TensorFlow: For building deep learning models
