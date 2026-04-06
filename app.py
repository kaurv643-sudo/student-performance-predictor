import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt

# Title
st.title("🎓 Student Performance Predictor")

# Load data
data = pd.read_csv("data.csv")

X = data[['study_hours', 'attendance', 'previous_grades']]
y = data['passed']

# Train model
model = LogisticRegression()
model.fit(X, y)

# Input fields
study = st.number_input("Study Hours", min_value=0.0)
attendance = st.number_input("Attendance", min_value=0.0)
grades = st.number_input("Previous Grades", min_value=0.0)

# Predict button
if st.button("Predict"):
    result = model.predict([[study, attendance, grades]])
    
    if result[0] == 1:
        st.success("Student will PASS 🎉")
    else:
        st.error("Student will FAIL ❌")

# Graph button
if st.button("Show Graph"):
    fig = sns.pairplot(data, hue='passed')
    st.pyplot(fig)