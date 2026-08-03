"""
Enterprise ML/DL Training Pipeline — Indian Healthcare Fraud Detection
=======================================================================
Level 1: Data Loading & Validation
Level 2: Feature Engineering & Preprocessing
Level 3: Deep Feature Extraction (CNN, Transformer, Autoencoder)
Level 4: Graph Neural Network (GNN/GraphSAGE)
Level 5: Hybrid Fusion Classifier (XGBoost on fused embeddings)
Level 6: Performance Evaluation & Model Persistence
"""

import pandas as pd
import numpy as np
import os
import pickle
import joblib
import traceback
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from models.autoencoder import create_autoencoder
from models.cnn import create_cnn
from models.transformer import create_transformer
from models.tree_models import get_decision_tree, get_random_forest, get_xgboost, get_hybrid_xgb
from models.gnn import GraphSAGE
import itertools

if not hasattr(np, "bool"):
    np.bool = bool

os.makedirs("saved_models", exist_ok=True)

print("=" * 65)
print("  ENTERPRISE MEDICARE FRAUD DETECTION — TRAINING PIPELINE")
print("  Dataset: Indian Hospital Claims (INR Pricing)")
print("=" * 65)

# ─────────────────────────────────────────────
# LEVEL 1: DATA LOADING & VALIDATION
# ─────────────────────────────────────────────
print("\n[LEVEL 1] Loading Indian Hospital Dataset...")

DATASET_PATH = "dataset/medicare_india_10000.csv"
try:
    df = pd.read_csv(DATASET_PATH, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(DATASET_PATH, encoding="latin1")

print(f"  ✅ Loaded {len(df):,} records | Columns: {list(df.columns)}")
print(f"  Fraud Rate: {df['Fraud'].mean()*100:.1f}%")
print(f"  Claim Amount Range: ₹{df['Claim_Amount'].min():,.0f} – ₹{df['Claim_Amount'].max():,.0f}")
print(f"  Mean Claim: ₹{df['Claim_Amount'].mean():,.0f}")

# ─────────────────────────────────────────────
# LEVEL 2: FEATURE ENGINEERING & PREPROCESSING
# ─────────────────────────────────────────────
print("\n[LEVEL 2] Feature Engineering...")

base_fields = ['Claim_Amount', 'Procedure_Code', 'Number_of_Procedures', 'Days_Admitted',
               'Patient_Age', 'Patient_Gender', 'Hospital_Type', 'Insurance_Type',
               'Diagnosis_Code', 'Claim_Day']

df_filtered = df[base_fields + ['Fraud']].copy()

# --- Derived Features (Intelligence Layer) ---
# Frequency maps (billing pattern analysis)
proc_freq_map = df_filtered['Procedure_Code'].value_counts().to_dict()
diag_freq_map = df_filtered['Diagnosis_Code'].value_counts().to_dict()
avg_claim_diag_map = df_filtered.groupby('Diagnosis_Code')['Claim_Amount'].mean().to_dict()
avg_claim_hosp_map = df_filtered.groupby('Hospital_Type')['Claim_Amount'].mean().to_dict()
avg_claim_proc_map = df_filtered.groupby('Procedure_Code')['Claim_Amount'].mean().to_dict()

df_filtered['Procedure_Frequency'] = df_filtered['Procedure_Code'].map(proc_freq_map)
df_filtered['Diag_Frequency'] = df_filtered['Diagnosis_Code'].map(diag_freq_map)
df_filtered['Avg_Claim_By_Diag'] = df_filtered['Diagnosis_Code'].map(avg_claim_diag_map)
# Cost-per-day ratio (key fraud signal for Indian hospital billing)
df_filtered['Cost_Per_Day'] = df_filtered.apply(
    lambda r: r['Claim_Amount'] / max(1, r['Days_Admitted']), axis=1)
# Claim deviation from diagnosis average (outlier signal)
df_filtered['Claim_Deviation_Pct'] = (
    (df_filtered['Claim_Amount'] - df_filtered['Avg_Claim_By_Diag']) /
    df_filtered['Avg_Claim_By_Diag'].clip(lower=1)
) * 100

# Save frequency maps
joblib.dump(proc_freq_map,       "saved_models/proc_freq_map.pkl")
joblib.dump(diag_freq_map,       "saved_models/diag_freq_map.pkl")
joblib.dump(avg_claim_diag_map,  "saved_models/avg_claim_diag_map.pkl")
joblib.dump(avg_claim_hosp_map,  "saved_models/avg_claim_hosp_map.pkl")
joblib.dump(avg_claim_proc_map,  "saved_models/avg_claim_proc_map.pkl")

# --- Label Encoding for Categorical Columns ---
cat_cols = ['Hospital_Type', 'Insurance_Type', 'Diagnosis_Code', 'Procedure_Code', 'Claim_Day', 'Patient_Gender']
for col in cat_cols:
    le = LabelEncoder()
    df_filtered[col] = le.fit_transform(df_filtered[col].astype(str))
    joblib.dump(le, f"saved_models/{col}_encoder.pkl")
    print(f"  Encoded '{col}': {list(le.classes_)}")

# --- Feature Matrix ---
X = df_filtered.drop("Fraud", axis=1)
y = df_filtered["Fraud"]

# Save feature column order (critical for inference alignment)
feature_columns_path = "saved_models/feature_columns.pkl"
if os.path.exists(feature_columns_path):
    os.remove(feature_columns_path)
with open(feature_columns_path, "wb") as f:
    pickle.dump(X.columns.tolist(), f)
print(f"\n  Feature columns ({len(X.columns)}): {X.columns.tolist()}")

# --- Train/Test Split & SMOTE ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n  Train size: {len(X_train):,} | Test size: {len(X_test):,}")
print(f"  Train fraud rate before SMOTE: {y_train.mean()*100:.1f}%")

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"  Train size after SMOTE: {len(X_train_resampled):,} (balanced)")

# --- Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled  = scaler.transform(X_test)
joblib.dump(scaler, "saved_models/scaler.pkl")

# Class weights for deep models
classes = np.unique(y_train_resampled)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_resampled)
class_weights = dict(zip(classes, weights))
print(f"  Class weights: {class_weights}")

# ─────────────────────────────────────────────
# LEVEL 3a: BASE TREE MODELS
# ─────────────────────────────────────────────
print("\n[LEVEL 3a] Training Base Tree Models...")

dt_model = get_decision_tree()
dt_model.fit(X_train_resampled, y_train_resampled)
joblib.dump(dt_model, "saved_models/dt_model.pkl")
dt_acc = accuracy_score(y_test, dt_model.predict(X_test))
print(f"  ✅ Decision Tree  — Test Accuracy: {dt_acc*100:.2f}%")

rf_model = get_random_forest()
rf_model.fit(X_train_resampled, y_train_resampled)
joblib.dump(rf_model, "saved_models/rf_model.pkl")
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
print(f"  ✅ Random Forest  — Test Accuracy: {rf_acc*100:.2f}%"
      f" | AUC: {roc_auc_score(y_test, rf_model.predict_proba(X_test)[:,1]):.4f}")

xgb_model = get_xgboost()
xgb_model.fit(X_train_resampled, y_train_resampled,
              verbose=False)
joblib.dump(xgb_model, "saved_models/xgb_model.pkl")
xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))
print(f"  ✅ XGBoost        — Test Accuracy: {xgb_acc*100:.2f}%"
      f" | AUC: {roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:,1]):.4f}")

# ─────────────────────────────────────────────
# LEVEL 3b: AUTOENCODER (Anomaly Detection)
# ─────────────────────────────────────────────
print("\n[LEVEL 3b] Training Autoencoder (Anomaly Detection)...")

autoencoder = create_autoencoder(X_train_scaled.shape[1])
# Train ONLY on legitimate claims — fraud = reconstruction anomaly
legit_train = X_train_scaled[y_train_resampled == 0]
early_stop_ae = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_reducer_ae = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5)
autoencoder.fit(
    legit_train, legit_train,
    epochs=30,
    batch_size=64,
    validation_split=0.15,
    verbose=1,
    callbacks=[early_stop_ae, lr_reducer_ae]
)
autoencoder.save("saved_models/autoencoder_model.h5")

# Compute anomaly scores (reconstruction error)
train_rec  = autoencoder.predict(X_train_scaled, verbose=0)
test_rec   = autoencoder.predict(X_test_scaled, verbose=0)
ae_scores_train = np.mean(np.power(X_train_scaled - train_rec, 2), axis=1).reshape(-1, 1)
ae_scores_test  = np.mean(np.power(X_test_scaled  - test_rec,  2), axis=1).reshape(-1, 1)

# Determine threshold (95th percentile of legit reconstruction error)
legit_test_scores = ae_scores_test[y_test == 0]
ae_threshold = float(np.percentile(legit_test_scores, 95))
joblib.dump(ae_threshold, "saved_models/ae_threshold.pkl")
print(f"  ✅ Autoencoder trained | Anomaly threshold: {ae_threshold:.4f}")

# ─────────────────────────────────────────────
# LEVEL 3c: CNN Feature Extractor
# ─────────────────────────────────────────────
print("\n[LEVEL 3c] Training CNN Feature Extractor...")

X_train_cnn = X_train_scaled.reshape(-1, X_train_scaled.shape[1], 1)
X_test_cnn  = X_test_scaled.reshape(-1, X_test_scaled.shape[1], 1)

cnn_model = create_cnn((X_train_cnn.shape[1], 1))
early_stop_cnn = EarlyStopping(monitor='val_auc', patience=8, restore_best_weights=True, mode='max')
lr_reducer_cnn = ReduceLROnPlateau(monitor='val_auc', patience=4, factor=0.5, mode='max')
cnn_model.fit(
    X_train_cnn, y_train_resampled,
    validation_split=0.15,
    epochs=50,
    batch_size=64,
    verbose=1,
    callbacks=[early_stop_cnn, lr_reducer_cnn],
    class_weight=class_weights
)
cnn_model.save("saved_models/cnn_model.h5")

# Build feature extractor (second-to-last Dense layer output)
cnn_extractor = tf.keras.Model(inputs=cnn_model.input, outputs=cnn_model.layers[-3].output)
cnn_features_train = cnn_extractor.predict(X_train_cnn, verbose=0)
cnn_features_test  = cnn_extractor.predict(X_test_cnn, verbose=0)
cnn_test_acc = accuracy_score(y_test, (cnn_model.predict(X_test_cnn, verbose=0) > 0.5).astype(int))
print(f"  ✅ CNN trained | Test Accuracy: {cnn_test_acc*100:.2f}%")

# ─────────────────────────────────────────────
# LEVEL 3d: Transformer Feature Extractor
# ─────────────────────────────────────────────
print("\n[LEVEL 3d] Training Transformer Feature Extractor...")

transformer_model = create_transformer((X_train_cnn.shape[1], 1))
early_stop_tr = EarlyStopping(monitor='val_auc', patience=8, restore_best_weights=True, mode='max')
lr_reducer_tr = ReduceLROnPlateau(monitor='val_auc', patience=4, factor=0.5, mode='max')
transformer_model.fit(
    X_train_cnn, y_train_resampled,
    validation_split=0.15,
    epochs=40,
    batch_size=64,
    verbose=1,
    callbacks=[early_stop_tr, lr_reducer_tr],
    class_weight=class_weights
)
transformer_model.save("saved_models/transformer_model.h5")

# Build transformer feature extractor
transformer_extractor = tf.keras.Model(
    inputs=transformer_model.input,
    outputs=transformer_model.layers[-3].output
)
transformer_features_train = transformer_extractor.predict(X_train_cnn, verbose=0)
transformer_features_test  = transformer_extractor.predict(X_test_cnn, verbose=0)
tr_test_acc = accuracy_score(y_test, (transformer_model.predict(X_test_cnn, verbose=0) > 0.5).astype(int))
print(f"  ✅ Transformer trained | Test Accuracy: {tr_test_acc*100:.2f}%")

# ─────────────────────────────────────────────
# LEVEL 4: GRAPH NEURAL NETWORK (GraphSAGE)
# ─────────────────────────────────────────────
print("\n[LEVEL 4] Training Graph Neural Network (GraphSAGE)...")
print("  Building claim graph: edges between claims sharing Diagnosis/Procedure codes...")

try:
    edge_index = []
    df_train_indexed = df_filtered.iloc[X_train.index].reset_index(drop=True)
    
    diag_groups = df_train_indexed.groupby('Diagnosis_Code').indices
    proc_groups = df_train_indexed.groupby('Procedure_Code').indices
    
    for _, indices in diag_groups.items():
        if len(indices) > 1:
            for i, j in itertools.combinations(indices[:30], 2):
                edge_index.append([i, j])
                edge_index.append([j, i])
                
    for _, indices in proc_groups.items():
        if len(indices) > 1:
            for i, j in itertools.combinations(indices[:30], 2):
                edge_index.append([i, j])
                edge_index.append([j, i])

    if not edge_index:
        edge_index = [[i, i+1] for i in range(len(X_train_scaled)-1)] + \
                     [[i+1, i] for i in range(len(X_train_scaled)-1)]

    edge_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x_tensor    = torch.tensor(X_train_scaled, dtype=torch.float)
    y_tensor    = torch.tensor(y_train_resampled.values, dtype=torch.long)
    data = Data(x=x_tensor, edge_index=edge_tensor, y=y_tensor)

    gnn_model = GraphSAGE(data.num_features, 64, 2)
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.005, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    gnn_model.train()
    for epoch in range(80):
        optimizer.zero_grad()
        out, _ = gnn_model(data.x, data.edge_index)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/80 — Loss: {loss.item():.4f}")
    
    torch.save(gnn_model.state_dict(), "saved_models/gnn.pt")
    print(f"  ✅ GNN (GraphSAGE) trained and saved.")

    gnn_model.eval()
    with torch.no_grad():
        _, gnn_embeds_train = gnn_model(x_tensor, edge_tensor)
        gnn_embeds_train = gnn_embeds_train.numpy()

        test_self_loops = [[i, i] for i in range(len(X_test_scaled))]
        test_edge = torch.tensor(test_self_loops, dtype=torch.long).t().contiguous()
        _, gnn_embeds_test = gnn_model(
            torch.tensor(X_test_scaled, dtype=torch.float), test_edge)
        gnn_embeds_test = gnn_embeds_test.numpy()

except Exception as e:
    print(f"  ⚠️ GNN failed, using zero-padding: {e}")
    traceback.print_exc()
    gnn_embeds_train = np.zeros((X_train_scaled.shape[0], 16))
    gnn_embeds_test  = np.zeros((X_test_scaled.shape[0], 16))

# ─────────────────────────────────────────────
# LEVEL 5: HYBRID FUSION CLASSIFIER
# ─────────────────────────────────────────────
print("\n[LEVEL 5] Building Hybrid Fused Feature Matrix & Training Final XGBoost...")

fused_train = np.hstack([
    X_train_scaled,
    cnn_features_train,
    transformer_features_train,
    ae_scores_train,
    gnn_embeds_train
])
fused_test = np.hstack([
    X_test_scaled,
    cnn_features_test,
    transformer_features_test,
    ae_scores_test,
    gnn_embeds_test
])

print(f"  Fused feature matrix shape: {fused_train.shape}")

hybrid_feature_names = (
    X.columns.tolist() +
    [f"CNN_Emb_{i}"      for i in range(cnn_features_train.shape[1])] +
    [f"Trans_Emb_{i}"    for i in range(transformer_features_train.shape[1])] +
    ["Autoencoder_Anomaly_Score"] +
    [f"GNN_Emb_{i}"      for i in range(gnn_embeds_train.shape[1])]
)

with open("saved_models/hybrid_feature_columns.pkl", "wb") as f:
    pickle.dump(hybrid_feature_names, f)

xgb_hybrid = get_hybrid_xgb()
xgb_hybrid.fit(fused_train, y_train_resampled,
               verbose=False)
joblib.dump(xgb_hybrid, "saved_models/xgb_hybrid.pkl")

# ─────────────────────────────────────────────
# LEVEL 6: PERFORMANCE EVALUATION
# ─────────────────────────────────────────────
print("\n[LEVEL 6] Performance Evaluation — Hybrid Pipeline...")

hybrid_preds  = xgb_hybrid.predict(fused_test)
hybrid_proba  = xgb_hybrid.predict_proba(fused_test)[:,1]

metrics = {
    "accuracy":  round(accuracy_score(y_test, hybrid_preds) * 100, 2),
    "precision": round(precision_score(y_test, hybrid_preds, zero_division=0) * 100, 2),
    "recall":    round(recall_score(y_test, hybrid_preds, zero_division=0) * 100, 2),
    "f1":        round(f1_score(y_test, hybrid_preds, zero_division=0) * 100, 2),
    "auc_roc":   round(roc_auc_score(y_test, hybrid_proba) * 100, 2),
}
joblib.dump(metrics, "saved_models/pipeline_metrics.pkl")

print("\n" + "=" * 65)
print("  MODEL PERFORMANCE SUMMARY")
print("=" * 65)
print(f"  {'Metric':<20} {'Score':>10}")
print(f"  {'-'*30}")
print(f"  {'Accuracy':<20} {metrics['accuracy']:>9.2f}%")
print(f"  {'Precision':<20} {metrics['precision']:>9.2f}%")
print(f"  {'Recall':<20} {metrics['recall']:>9.2f}%")
print(f"  {'F1-Score':<20} {metrics['f1']:>9.2f}%")
print(f"  {'AUC-ROC':<20} {metrics['auc_roc']:>9.2f}%")
print(f"\n  Classification Report:\n")
print(classification_report(y_test, hybrid_preds, target_names=['Legitimate', 'Fraud']))
print("=" * 65)
print(f"\n  ✅ All models saved to saved_models/")
print(f"  ✅ Pipeline Levels 1–6 completed successfully.")
print("=" * 65)
