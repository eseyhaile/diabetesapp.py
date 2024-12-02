import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve



data = pd.read_csv('processed_diabetes.csv')

# Streamlit app
st.markdown("<h1 style='text-align: center; color: white;'>Diabetes Prediction App</h1>", unsafe_allow_html=True)

with st.sidebar:
    add_plot = st.radio(
        "Would you like a plot to be displayed?",
        ("Yes", "No")
    )
# Input features
st.header('Enter Patient Details:')
highbp = st.number_input('HighBP', min_value=0, max_value=1)
highchol = st.number_input('HighChol', min_value=0, max_value=1)
bmi = st.slider('BMI', min_value=0.0, max_value = data['BMI'].max(), step = 1.0)
smoker = st.number_input('Smoker', min_value=0, max_value=1)
stroke = st.number_input('Stroke', min_value=0, max_value=1)
heartdiseaseorattack = st.number_input('HeartDiseaseorAttack', min_value=0, max_value=1)
physactivity = st.number_input('PhysActivity', min_value=0, max_value=1)
fruits = st.number_input('Fruits', min_value=0, max_value=1)
veggies = st.number_input('Veggies', min_value=0, max_value=1)
hvyAlcoholConsump = st.number_input('HvyAlcoholConsump', min_value=0, max_value=1)
anyhealthcare = st.number_input('AnyHealthcare', min_value=0, max_value=1)
genhlth = st.number_input('GenHlth', min_value=1, max_value=5)
menthealth = st.number_input('MentHlth', min_value=0)
physhealth = st.number_input('PhysHlth', min_value=0)
diffwalk = st.number_input('DiffWalk', min_value=0, max_value=1)
sex = st.number_input('Sex', min_value=0, max_value=1)
age = st.number_input('Age', min_value=18)
education = st.number_input('Education', min_value=1, max_value=6)
income = st.number_input('Income', min_value=1, max_value=8)


# Preidiction button
if st.button('Predict'):
    # This is a dataframe with the input features 
    input_data = pd.DataFrame({
        'HighBP': [highbp],
        'HighChol': [highchol],
        'BMI': [bmi],
        'Smoker': [smoker],
        'Stroke': [stroke],
        'HeartDiseaseorAttack': [heartdiseaseorattack],
        'PhysActivity': [physactivity],
        'Fruits': [fruits],
        'Veggies': [veggies],
        'HvyAlcoholConsump': [hvyAlcoholConsump],
        'AnyHealthcare': [anyhealthcare],
        'GenHlth': [genhlth],
        'MentHlth': [menthealth],
        'PhysHlth': [physhealth],
        'DiffWalk': [diffwalk],
        'Sex': [sex],
        'Age': [age],
        'Education': [education],
        'Income': [income]
    })


    # We use the trained model to make a predicition 
    prediction = log_reg.predict(input_data)

    # Display prediction
    if prediction[0] == 1:
        st.write('Prediction: The patient is likely to have diabetes.')
    else:
        st.write('Prediction: The patient is likely not to have diabetes.')


#These are the performance metrics of our logreg model
st.header('Model Performance')
st.write(f'Accuracy: {accuracy:.2f}')
st.write('Confusion Matrix:')
st.write(conf_matrix)
st.write('Classification Report:')
st.write(class_report)

# A receiver operator characteristic (roc) curve is a graph that shows how well a binary classifier model performs at different threshold values
st.header('ROC Curve')
fpr, tpr, thresholds = roc_curve(y_test, log_reg.predict_proba(X_test)[:, 1])
fig = go.Figure(data=go.Scatter(x=fpr, y=tpr, mode='lines'))
fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
fig.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
st.plotly_chart(fig)

#This visualization is a precision recall curve that shows the relationship between The ratio of true positives to the sum of true and false positives (precision)
#and The ratio of true positives to the sum of true positives and false negatives (recall)
st.header('Precision-Recall Curve')
precision, recall, thresholds = precision_recall_curve(y_test, log_reg.predict_proba(X_test)[:, 1])
fig = go.Figure(data=go.Scatter(x=recall, y=precision, mode='lines'))
fig.update_layout(xaxis_title='Recall', yaxis_title='Precision')
st.plotly_chart(fig)

#This is the feature importance section which we will use coefficients for because it is a logistic regression model 
st.header('Feature Importance')
feature_importance = pd.DataFrame({'Feature': X.columns, 'Coefficient': log_reg.coef_[0]})
feature_importance = feature_importance.sort_values(by='Coefficient', ascending=False)
st.write(feature_importance)