# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import base64
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ---------- Background Styling ----------
def add_background(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{

        
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_background("loan_bg.jpg")  # background image

# ---------- Page Setup ----------
st.set_page_config(page_title="Loan Prediction App", layout="wide")
st.markdown("<h1 style='text-align: center; color: white;'>🏦 Loan Approval Prediction Dashboard</h1>", unsafe_allow_html=True)

# ---------- Load & Preprocess Data ----------
@st.cache_data
def load_data():
    df = pd.read_csv("loan_data.csv")
    return df.dropna()

df = load_data()
df_original = df.copy()

# Custom Gender Encoding (Assuming 'Gender' is present)
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].replace({1: 'Male', 0: 'Female'})

# Label Encoding for all categoricals
label_encoders = {}
df_clean = df.copy()
for col in df_clean.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])
    label_encoders[col] = le

# Split into Features and Target
target_col = 'Loan_Status'
X = df_clean.drop(columns=[target_col])
y = df_clean[target_col]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train SVM model

model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.sidebar.success(f"Model Accuracy: {acc * 100:.2f}%")

# ---------- Visualizations ----------
st.subheader("📊 Data Insights")
viz1, viz2 = st.columns(2)

with viz1:
    st.markdown("**Loan Approval Distribution**")
    fig1, ax1 = plt.subplots()
    df[target_col].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, ax=ax1)
    ax1.axis("equal")
    st.pyplot(fig1)

with viz2:
    if 'ApplicantIncome' in df.columns and 'LoanAmount' in df.columns:
        st.markdown("**Income vs Loan Amount**")
        fig2, ax2 = plt.subplots()
        ax2.scatter(df['ApplicantIncome'], df['LoanAmount'], alpha=0.5)
        ax2.set_xlabel("Applicant Income")
        ax2.set_ylabel("Loan Amount")
        st.pyplot(fig2)

st.markdown("---")

# ---------- Prediction Form ----------
st.subheader("📥 Enter Loan Application Details")

input_data = {}
cols = st.columns(2)
for i, col_name in enumerate(X.columns):
    raw_col = df[col_name]
    if df_original[col_name].dtype == 'object':
        options = df_original[col_name].unique().tolist()
        input_data[col_name] = cols[i % 2].selectbox(f"{col_name}:", options)
    else:
        val = float(df_original[col_name].mean())
        input_data[col_name] = cols[i % 2].number_input(f"{col_name}:", value=val)

# ---------- Predict Button ----------
if st.button("🔍 Predict"):
    input_df = pd.DataFrame([input_data])
    for col in input_df.columns:
        if col in label_encoders:
            input_df[col] = label_encoders[col].transform(input_df[col])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][prediction]
    result = "✅ Loan Approved" if prediction == 1 else "❌ Loan Rejected"
    st.success(f"Prediction: {result} ({proba * 100:.2f}%)")

    # ---------- Downloadable Report ----------
    output = input_df.copy()
    output['Prediction'] = result
    csv = output.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="loan_prediction_report.csv">📥 Download Report</a>'
    st.markdown(href, unsafe_allow_html=True)

# ---------- Feature Importance ----------
st.markdown("---")
st.subheader("📌 Feature Importance (SVM Coefficients)")
if hasattr(model, 'coef_'):
    coef = model.coef_[0]
    importance_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': coef})
    importance_df = importance_df.sort_values(by='Coefficient', key=abs, ascending=False)

    fig_imp, ax = plt.subplots()
    ax.barh(importance_df['Feature'], importance_df['Coefficient'])
    ax.set_title("SVM Feature Importance")
    st.pyplot(fig_imp)

