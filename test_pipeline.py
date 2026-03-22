import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import pickle
import torch
import warnings
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

scaler = joblib.load('saved_models/scaler.pkl')

hospital_encoder = joblib.load("saved_models/Hospital_Type_encoder.pkl")
insurance_encoder = joblib.load("saved_models/Insurance_Type_encoder.pkl")
diagnosis_encoder = joblib.load("saved_models/Diagnosis_Code_encoder.pkl")
procedure_encoder = joblib.load("saved_models/Procedure_Code_encoder.pkl")
day_encoder = joblib.load("saved_models/Claim_Day_encoder.pkl")
proc_freq_map = joblib.load("saved_models/proc_freq_map.pkl")
diag_freq_map = joblib.load("saved_models/diag_freq_map.pkl")
avg_claim_diag_map = joblib.load("saved_models/avg_claim_diag_map.pkl")

with open('saved_models/feature_columns.pkl', "rb") as f:
    feature_columns = pickle.load(f)
feature_columns = [c for c in feature_columns if c != "Fraud"]

with open('saved_models/hybrid_feature_columns.pkl', "rb") as f:
    hybrid_feature_columns = pickle.load(f)

rf_model = joblib.load('saved_models/rf_model.pkl')
xgb_hybrid = joblib.load('saved_models/xgb_hybrid.pkl')
cnn_model = tf.keras.models.load_model('saved_models/cnn_model.h5', compile=False)

def predict(input_data):
    input_copy = input_data.copy()

    try: input_copy['Hospital_Type'] = hospital_encoder.transform([str(input_data['Hospital_Type'])])[0]
    except: input_copy['Hospital_Type'] = -1
    try: input_copy['Insurance_Type'] = insurance_encoder.transform([str(input_data['Insurance_Type'])])[0]
    except: input_copy['Insurance_Type'] = -1
    try: input_copy['Diagnosis_Code'] = diagnosis_encoder.transform([str(input_data['Diagnosis_Code'])])[0]
    except: input_copy['Diagnosis_Code'] = -1
    try: input_copy['Procedure_Code'] = procedure_encoder.transform([str(input_data['Procedure_Code'])])[0]
    except: input_copy['Procedure_Code'] = -1
    try: input_copy['Claim_Day'] = day_encoder.transform([str(input_data['Claim_Day'])])[0]
    except: input_copy['Claim_Day'] = -1

    input_copy['Procedure_Frequency'] = proc_freq_map.get(int(input_data['Procedure_Code']), 1)
    input_copy['Diag_Frequency'] = diag_freq_map.get(str(input_data['Diagnosis_Code']), 1)
    input_copy['Avg_Claim_By_Diag'] = avg_claim_diag_map.get(str(input_data['Diagnosis_Code']), input_data['Claim_Amount'])

    df = pd.DataFrame([input_copy])
    df_encoded = pd.get_dummies(df)

    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)
    df_scaled = scaler.transform(df_encoded)

    rf_pred = rf_model.predict(df_encoded)[0]
    cnn_input = df_scaled.reshape(1, df_scaled.shape[1], 1)
    cnn_prob = cnn_model.predict(cnn_input, verbose=0)[0][0]
    cnn_pred = int(cnn_prob > 0.5)

    return rf_pred, cnn_pred, cnn_prob

# Legit Claim
legit = {
    "Claim_Amount": 2000,
    "Procedure_Code": "99213",
    "Number_of_Procedures": 2,
    "Days_Admitted": 2,
    "Patient_Age": 45,
    "Hospital_Type": "Clinic",
    "Insurance_Type": "Private",
    "Diagnosis_Code": "M54",
    "Claim_Day": "Wednesday",
    "Patient_Gender": "F"
}

# Fraud Claim
fraud = {
    "Claim_Amount": 9500,
    "Procedure_Code": "99215",
    "Number_of_Procedures": 9,
    "Days_Admitted": 22,
    "Patient_Age": 65,
    "Hospital_Type": "Clinic",
    "Insurance_Type": "Government",
    "Diagnosis_Code": "E11",
    "Claim_Day": "Monday",
    "Patient_Gender": "M"
}

print("Running Dummy Predicts...")
l_rf, l_cnn, l_cnn_prob = predict(legit)
print(f"Legit Case -> RF: {l_rf}, CNN: {l_cnn} (prob {l_cnn_prob:.4f})")

f_rf, f_cnn, f_cnn_prob = predict(fraud)
print(f"Fraud Case -> RF: {f_rf}, CNN: {f_cnn} (prob {f_cnn_prob:.4f})")
