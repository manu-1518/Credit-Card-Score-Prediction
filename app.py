#Webpage using streamlit
import streamlit as st
import pandas as pd
import joblib
from preprocess_data import preprocess

# Load model and expected columns
model = joblib.load("xgb_model.pkl")
expected_columns = joblib.load("expected_columns.pkl")

st.title("Credit Score Prediction App")

st.markdown("Fill in the details below to predict if the credit score is good (0) or bad (1).")

# Input fields (must match training features)
user_input = {
    "Status": st.selectbox("Status", ["A11", "A12", "A13", "A14"]),
    "Duration": st.number_input("Duration (in months)", min_value=1, value=12),
    "CreditHistory": st.selectbox("Credit History", ["A30", "A31", "A32", "A33", "A34"]),
    "Purpose": st.selectbox("Purpose", ["A40", "A41", "A42", "A43", "A44", "A45", "A46", "A47", "A48", "A49"]),
    "CreditAmount": st.number_input("Credit Amount", min_value=0, value=1000),
    "Savings": st.selectbox("Savings", ["A61", "A62", "A63", "A64", "A65"]),
    "Employment": st.selectbox("Employment", ["A71", "A72", "A73", "A74", "A75"]),
    "InstallmentRate": st.slider("Installment Rate (1 = highest)", 1, 4, 2),
    "PersonalStatusSex": st.selectbox("Personal Status & Sex", ["A91", "A92", "A93", "A94", "A95"]),
    "OtherDebtors": st.selectbox("Other Debtors", ["A101", "A102", "A103"]),
    "ResidenceSince": st.slider("Residence Since (years)", 1, 4, 2),
    "Property": st.selectbox("Property", ["A121", "A122", "A123", "A124"]),
    "Age": st.number_input("Age", min_value=18, max_value=100, value=30),
    "OtherInstallmentPlans": st.selectbox("Other Installment Plans", ["A141", "A142", "A143"]),
    "Housing": st.selectbox("Housing", ["A151", "A152", "A153"]),
    "ExistingCredits": st.slider("Existing Credits", 1, 4, 1),
    "Job": st.selectbox("Job", ["A171", "A172", "A173", "A174"]),
    "NumDependents": st.selectbox("Number of Dependents", [1, 2]),
    "OwnTelephone": st.selectbox("Own Telephone", ["A191", "A192"]),
    "ForeignWorker": st.selectbox("Foreign Worker", ["A201", "A202"])
}

input_df = pd.DataFrame([user_input])

# Preprocess user input
X_input = preprocess(input_df, is_train=False)

# Align with expected columns from training
for col in expected_columns:
    if col not in X_input.columns:
        X_input[col] = 0
X_input = X_input[expected_columns]

# Predict
if st.button("Predict Credit Score"):
    try:
        prediction = model.predict(X_input)[0]
        result = "Good Credit (0)" if prediction == 0 else "Bad Credit (1)"
        st.success(f"Prediction: {result}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
