import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate_model():
    print("Loading datasets and models for manual evaluation...")
    
    # 1. Load the dataset
    df = pd.read_csv("dataset/medicare_india_10000.csv")
    
    # 2. Extract features and labels exactly as done in training
    base_fields = ['Claim_Amount', 'Procedure_Code', 'Number_of_Procedures', 'Days_Admitted',
                   'Patient_Age', 'Patient_Gender', 'Hospital_Type', 'Insurance_Type',
                   'Diagnosis_Code', 'Claim_Day']
    
    df_filtered = df[base_fields + ['Fraud']].copy()
    
    # Load required encoders and frequency maps
    hospital_enc = joblib.load("saved_models/Hospital_Type_encoder.pkl")
    insurance_enc = joblib.load("saved_models/Insurance_Type_encoder.pkl")
    diagnosis_enc = joblib.load("saved_models/Diagnosis_Code_encoder.pkl")
    procedure_enc = joblib.load("saved_models/Procedure_Code_encoder.pkl")
    day_enc = joblib.load("saved_models/Claim_Day_encoder.pkl")
    gender_enc = joblib.load("saved_models/Patient_Gender_encoder.pkl")
    
    proc_freq_map = joblib.load("saved_models/proc_freq_map.pkl")
    diag_freq_map = joblib.load("saved_models/diag_freq_map.pkl")
    avg_claim_diag_map = joblib.load("saved_models/avg_claim_diag_map.pkl")
    scaler = joblib.load("saved_models/scaler.pkl")
    
    print("Applying preprocessing...")
    # Apply derived features
    df_filtered['Procedure_Frequency'] = df_filtered['Procedure_Code'].map(proc_freq_map).fillna(1)
    df_filtered['Diag_Frequency'] = df_filtered['Diagnosis_Code'].map(diag_freq_map).fillna(1)
    df_filtered['Avg_Claim_By_Diag'] = df_filtered['Diagnosis_Code'].map(avg_claim_diag_map)
    df_filtered['Cost_Per_Day'] = df_filtered.apply(lambda r: r['Claim_Amount'] / max(1, r['Days_Admitted']), axis=1)
    df_filtered['Claim_Deviation_Pct'] = ((df_filtered['Claim_Amount'] - df_filtered['Avg_Claim_By_Diag']) / df_filtered['Avg_Claim_By_Diag'].clip(lower=1)) * 100
    
    # Encode categorical columns
    df_filtered['Hospital_Type'] = hospital_enc.transform(df_filtered['Hospital_Type'].astype(str))
    df_filtered['Insurance_Type'] = insurance_enc.transform(df_filtered['Insurance_Type'].astype(str))
    df_filtered['Diagnosis_Code'] = diagnosis_enc.transform(df_filtered['Diagnosis_Code'].astype(str))
    df_filtered['Procedure_Code'] = procedure_enc.transform(df_filtered['Procedure_Code'].astype(str))
    df_filtered['Claim_Day'] = day_enc.transform(df_filtered['Claim_Day'].astype(str))
    df_filtered['Patient_Gender'] = gender_enc.transform(df_filtered['Patient_Gender'].astype(str))
    
    X = df_filtered.drop("Fraud", axis=1)
    y = df_filtered["Fraud"]
    
    # 3. Data Splitting: Ensure test split perfectly matches train.py (80/20 stratify=y random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale test set
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Load a primary model (e.g., Random Forest or XGBoost baseline)
    print("Loading XGBoost core model for prediction...")
    model = joblib.load("saved_models/xgb_model.pkl")
    
    # 5. Generate Predictions on Test data using adjusted threshold for imbalanced data
    print("Generating predictions on unseen test split using optimized classification threshold (0.35)...")
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Tune threshold to prioritize catching fraud (Recall)
    threshold = 0.35
    y_pred = (y_prob >= threshold).astype(int)
    
    # 6. Calculate Metrics
    print("\n" + "="*50)
    print("   MODEL EVALUATION REPORT (XGBoost Core - Adjusted Threshold)")
    print("="*50)
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print("Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))
    
    # 7. Visualize Results: Confusion Matrix
    print("Generating Confusion Matrix plot (saving to 'confusion_matrix.png')...")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Legitimate', 'Predicted Fraud'],
                yticklabels=['Actual Legitimate', 'Actual Fraud'])
    plt.title('Medicare Fraud Detection - Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("✅ Evaluation complete! Open 'confusion_matrix.png' to view the visual diagnostic.")

if __name__ == "__main__":
    evaluate_model()
