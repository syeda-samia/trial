import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Student Dropout Dashboard", layout="wide")

# ---------------- LOAD FILES ----------------
model = joblib.load("dropout_model.pkl")
scaler = joblib.load("scaler.pkl")
num_imputer = joblib.load("num_imputer.pkl")
cat_imputer = joblib.load("cat_imputer.pkl")
model_columns = joblib.load("model_columns.pkl")

df = pd.read_csv("student_dropout_dataset_v3.csv")
df = df.drop(columns=["Student_ID"])
df['Semester'] = df['Semester'].str.extract('(\d+)').astype(float)

numeric_cols = [
    'Age', 'Family_Income', 'Study_Hours_per_Day',
    'Attendance_Rate', 'Assignment_Delay_Days',
    'Travel_Time_Minutes', 'Stress_Index',
    'GPA', 'Semester_GPA', 'CGPA', 'Semester'
]

categorical_cols = [
    'Gender', 'Internet_Access', 'Part_Time_Job',
    'Scholarship', 'Department', 'Parental_Education'
]

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Dataset Overview",
    "Dropout Analysis",
    "Model Insights",
    "Predict Dropout"
])

# ---------------- PAGE 1 ----------------
if page == "Dataset Overview":
    st.title("Student Dropout Dataset Overview")

    st.write("Shape of dataset:", df.shape)
    st.dataframe(df.head())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

# ---------------- PAGE 2 ----------------
elif page == "Dropout Analysis":
    st.title("Dropout Analysis")

    fig, ax = plt.subplots()
    df['Dropout'].value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10,6))
    sns.heatmap(df[numeric_cols].corr(), cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

# ---------------- PAGE 3 ----------------
elif page == "Model Insights":
    st.title("Model Feature Importance")

    importances = pd.Series(model.feature_importances_, index=model_columns)
    top_features = importances.sort_values(ascending=False).head(15)

    fig3, ax3 = plt.subplots(figsize=(10,6))
    top_features.plot(kind='bar', ax=ax3)
    st.pyplot(fig3)

# ---------------- PAGE 4 ----------------
elif page == "Predict Dropout":
    st.title("Predict Student Dropout")

    st.write("Enter student details:")

    input_data = {}

    for col in numeric_cols:
        input_data[col] = st.number_input(col, value=0.0)

    for col in categorical_cols:
        input_data[col] = st.selectbox(col, df[col].unique())

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])

        # Preprocessing same as notebook
        input_df[numeric_cols] = num_imputer.transform(input_df[numeric_cols])
        input_df[categorical_cols] = cat_imputer.transform(input_df[categorical_cols])
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        input_df = pd.get_dummies(input_df, drop_first=True)
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        prob = model.predict_proba(input_df)[0][1]
        pred = model.predict(input_df)[0]

        st.subheader("Prediction Result")
        st.write(f"Dropout Probability: **{prob:.2f}**")

        if pred == 1:
            st.error("⚠️ This student is likely to Dropout")
        else:
            st.success("✅ This student is likely to Continue")