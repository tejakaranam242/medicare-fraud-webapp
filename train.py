import pandas as pd
import numpy as np
import os
import pickle
import joblib
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from models.autoencoder import create_autoencoder
from models.cnn import create_cnn
from models.transformer import create_transformer
from models.tree_models import get_decision_tree, get_random_forest, get_xgboost, get_hybrid_xgb
from models.gnn import GraphSAGE
import itertools

if not hasattr(np, "bool"):
    np.bool = bool

os.makedirs("saved_models", exist_ok=True)

try:
    df = pd.read_csv("dataset/medicare_5000.csv", encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv("dataset/medicare_5000.csv", encoding="latin1")

# --- LEVEL 2: Data Preprocessing & Feature Engineering ---
# Base 9 fields requested by user
base_fields = ['Claim_Amount', 'Procedure_Code', 'Number_of_Procedures', 'Days_Admitted', 
               'Patient_Age', 'Patient_Gender', 'Hospital_Type', 'Insurance_Type', 'Diagnosis_Code', 'Claim_Day']

# Keep target and base fields
df_filtered = df[base_fields + ['Fraud']].copy()

# Add derived features (Intelligence Layer 2)
# Frequency mapping for categorical interactions
# We count these globally in training and save maps for inference
proc_freq_map = df_filtered['Procedure_Code'].value_counts().to_dict()
diag_freq_map = df_filtered['Diagnosis_Code'].value_counts().to_dict()
avg_claim_diag_map = df_filtered.groupby('Diagnosis_Code')['Claim_Amount'].mean().to_dict()

df_filtered['Procedure_Frequency'] = df_filtered['Procedure_Code'].map(proc_freq_map)
df_filtered['Diag_Frequency'] = df_filtered['Diagnosis_Code'].map(diag_freq_map)
df_filtered['Avg_Claim_By_Diag'] = df_filtered['Diagnosis_Code'].map(avg_claim_diag_map)

# Save these maps
joblib.dump(proc_freq_map, "saved_models/proc_freq_map.pkl")
joblib.dump(diag_freq_map, "saved_models/diag_freq_map.pkl")
joblib.dump(avg_claim_diag_map, "saved_models/avg_claim_diag_map.pkl")

# Label Encode categorical strings for the deep models
cat_cols = ['Hospital_Type', 'Insurance_Type', 'Diagnosis_Code', 'Procedure_Code', 'Claim_Day']
for col in cat_cols:
    le = LabelEncoder()
    df_filtered[col] = le.fit_transform(df_filtered[col].astype(str))
    # Save encoder for inference
    joblib.dump(le, f"saved_models/{col}_encoder.pkl")

df_encoded = pd.get_dummies(df_filtered)
X = df_encoded.drop("Fraud", axis=1)
y = df_encoded["Fraud"]

feature_columns_path = "saved_models/feature_columns.pkl"
if os.path.exists(feature_columns_path):
    os.remove(feature_columns_path)

with open(feature_columns_path, "wb") as f:
    pickle.dump(X.columns.tolist(), f)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "saved_models/scaler.pkl")

classes = np.unique(y_train_resampled)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_resampled)
class_weights = dict(zip(classes, weights))

# --- LEVEL 3: Feature Extraction Models ---
# 1. Base Trees (For comparison display, not part of DL feature ext)
dt_model = get_decision_tree()
dt_model.fit(X_train_resampled, y_train_resampled)
joblib.dump(dt_model, "saved_models/dt_model.pkl")

rf_model = get_random_forest()
rf_model.fit(X_train_resampled, y_train_resampled)
joblib.dump(rf_model, "saved_models/rf_model.pkl")

xgb_model = get_xgboost()
xgb_model.fit(X_train_resampled, y_train_resampled)
joblib.dump(xgb_model, "saved_models/xgb_model.pkl")

# 2. Autoencoder
autoencoder = create_autoencoder(X_train_scaled.shape[1])
autoencoder.fit(X_train_scaled[y_train_resampled == 0], X_train_scaled[y_train_resampled == 0], epochs=20, batch_size=32, validation_split=0.2, verbose=1)
autoencoder.save("saved_models/autoencoder_model.h5")
# Extract Anomaly Scores
train_reconstructed = autoencoder.predict(X_train_scaled)
test_reconstructed = autoencoder.predict(X_test_scaled)
ae_scores_train = np.mean(np.power(X_train_scaled - train_reconstructed, 2), axis=1).reshape(-1, 1)
ae_scores_test = np.mean(np.power(X_test_scaled - test_reconstructed, 2), axis=1).reshape(-1, 1)

# 3. CNN
X_train_cnn = X_train_scaled.reshape(-1, X_train_scaled.shape[1], 1)
X_test_cnn = X_test_scaled.reshape(-1, X_test_scaled.shape[1], 1)

cnn_model = create_cnn((X_train_cnn.shape[1], 1))
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
cnn_model.fit(X_train_cnn, y_train_resampled, validation_split=0.2, epochs=50, batch_size=32, verbose=1, callbacks=[early_stop], class_weight=class_weights)
cnn_model.save("saved_models/cnn_model.h5")
# Deep Feature Extractor
cnn_extractor = tf.keras.Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)
cnn_features_train = cnn_extractor.predict(X_train_cnn)
cnn_features_test = cnn_extractor.predict(X_test_cnn)

# 4. Transformer
transformer_model = create_transformer((X_train_cnn.shape[1], 1))
transformer_model.fit(X_train_cnn, y_train_resampled, validation_split=0.2, epochs=30, batch_size=32, verbose=1)
transformer_model.save("saved_models/transformer_model.h5")
# Deep Feature Extractor
transformer_extractor = tf.keras.Model(inputs=transformer_model.input, outputs=transformer_model.layers[-2].output)
transformer_features_train = transformer_extractor.predict(X_train_cnn)
transformer_features_test = transformer_extractor.predict(X_test_cnn)

# 5. Graph Neural Network (GNN)
# We improve GNN by linking records that share the same Diagnosis_Code or Procedure_Code
try:
    edge_index = []
    # Create edges based on shared categorical codes
    # To avoid O(N^2) complexity, we use grouping
    diag_groups = df_filtered.iloc[X_train.index].groupby('Diagnosis_Code').indices
    proc_groups = df_filtered.iloc[X_train.index].groupby('Procedure_Code').indices
    
    for _, indices in diag_groups.items():
        if len(indices) > 1:
            for i, j in itertools.combinations(indices[:20], 2): # Cap connections to avoid memory explosion
                edge_index.append([i, j])
                edge_index.append([j, i])
                
    for _, indices in proc_groups.items():
        if len(indices) > 1:
            for i, j in itertools.combinations(indices[:20], 2):
                edge_index.append([i, j])
                edge_index.append([j, i])

    if not edge_index:
        # Fallback to sequential if no groups found
        edge_index = [[i, i+1] for i in range(len(X_train_scaled)-1)] + [[i+1, i] for i in range(len(X_train_scaled)-1)]

    edge_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    x_tensor = torch.tensor(X_train_scaled, dtype=torch.float)
    y_tensor = torch.tensor(y_train_resampled.values, dtype=torch.long)
    data = Data(x=x_tensor, edge_index=edge_tensor, y=y_tensor)
    
    gnn_model = GraphSAGE(data.num_features, 64, 2)
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01, weight_decay=5e-4)

    gnn_model.train()
    for __ in range(50): # Increased epochs for better convergence
        optimizer.zero_grad()
        out, _ = gnn_model(data.x, data.edge_index)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
    torch.save(gnn_model.state_dict(), "saved_models/gnn.pt")
    
    gnn_model.eval()
    with torch.no_grad():
        _, gnn_embeds_train = gnn_model(x_tensor, edge_tensor)
        gnn_embeds_train = gnn_embeds_train.numpy()
        
        # Test Embeds (Self-loops for single record inference compatibility)
        test_edge = [[i, i] for i in range(len(X_test_scaled))]
        test_edge = torch.tensor(test_edge, dtype=torch.long).t().contiguous()
        _, gnn_embeds_test = gnn_model(torch.tensor(X_test_scaled, dtype=torch.float), test_edge)
        gnn_embeds_test = gnn_embeds_test.numpy()
except Exception as e:
    print(f"GNN Embedding extraction failed, padding with zeros: {e}")
    traceback.print_exc()
    gnn_embeds_train = np.zeros((X_train_scaled.shape[0], 2))
    gnn_embeds_test = np.zeros((X_test_scaled.shape[0], 2))

# --- LEVEL 4: Classification Layer (XGBoost) ---
# Combine ALL derived representations: [Engineered Tabular] + [CNN Embed] + [Transform Embed] + [Autoencoder Score] + [GNN Embed]
fused_train = np.hstack([X_train_scaled, cnn_features_train, transformer_features_train, ae_scores_train, gnn_embeds_train])
fused_test = np.hstack([X_test_scaled, cnn_features_test, transformer_features_test, ae_scores_test, gnn_embeds_test])

print(f"Final Fused Feature Matrix Shape: {fused_train.shape}")

hybrid_feature_names = X.columns.tolist() + \
                       [f"CNN_Emb_{i}" for i in range(cnn_features_train.shape[1])] + \
                       [f"Trans_Emb_{i}" for i in range(transformer_features_train.shape[1])] + \
                       ["Autoencoder_Anomaly_Score"] + \
                       [f"GNN_Emb_{i}" for i in range(gnn_embeds_train.shape[1])]

with open("saved_models/hybrid_feature_columns.pkl", "wb") as f:
    pickle.dump(hybrid_feature_names, f)

xgb_hybrid = get_hybrid_xgb()
xgb_hybrid.fit(fused_train, y_train_resampled)
joblib.dump(xgb_hybrid, "saved_models/xgb_hybrid.pkl")

# --- LEVEL 5: Performance Evaluation (Proof of Work) ---
hybrid_preds = xgb_hybrid.predict(fused_test)
metrics = {
    "accuracy": round(accuracy_score(y_test, hybrid_preds) * 100, 2),
    "precision": round(precision_score(y_test, hybrid_preds) * 100, 2),
    "recall": round(recall_score(y_test, hybrid_preds) * 100, 2),
    "f1": round(f1_score(y_test, hybrid_preds) * 100, 2)
}
joblib.dump(metrics, "saved_models/pipeline_metrics.pkl")

print(f"✅ Pipeline Level 1-5 Training Completed Successfully.")
print(f"📈 Evaluation Metrics: {metrics}")
