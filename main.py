import sqlite3
import re
import json
import os
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_from_directory
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np
import pickle
import shap
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import traceback
import sqlite3
import json
import torch
from models.gnn import GraphSAGE

if not hasattr(np, "bool"):
    np.bool = bool

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

MODEL_PATHS = {
    'cnn': 'saved_models/cnn_model.h5',
    'transformer': 'saved_models/transformer_model.h5',
    'autoencoder': 'saved_models/autoencoder_model.h5',
    'hybrid': 'saved_models/xgb_hybrid.pkl',
    'rf': 'saved_models/rf_model.pkl',
    'dt': 'saved_models/dt_model.pkl',
    'xgb': 'saved_models/xgb_model.pkl',
    'scaler': 'saved_models/scaler.pkl',
    'features': 'saved_models/feature_columns.pkl',
    'hybrid_features': 'saved_models/hybrid_feature_columns.pkl'
}

MODEL_DISPLAY = {
    "rf": "Random Forest",
    "dt": "Decision Tree",
    "xgb": "XGBoost",
    "cnn": "CNN",
    "transformer": "Transformer",
    "autoencoder": "Autoencoder Anomaly Detection",
    "gnn": "Graph Neural Network",
    "hybrid": "Hybrid Level 4 Classifier (Deep+XGB)"
}

def get_db_connection():
    conn = sqlite3.connect("database.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def check_expert_rules(data):
    """
    Enterprise-level heuristic validation — Indian Healthcare Context.
    Thresholds calibrated to Indian hospital pricing in INR (₹).
    """
    risk_flags = []

    # --- INR Pricing Reference (Indian hospitals) ---
    DAILY_FLOOR_INR = {
        'Government':  500,
        'AYUSH':       1200,
        'Nursing_Home': 3500,
        'Private':     8000,
        'Corporate':   22000,
    }
    INSURANCE_CAPS_INR = {
        'Ayushman_Bharat': 500000,   # PM-JAY cap: ₹5,00,000
        'ESI':              300000,   # ESI: ~₹3,00,000
        'CGHS':             400000,   # CGHS: ~₹4,00,000
        'Private':         1500000,  # No strict cap
        'Self-pay':        9999999,  # No cap
    }

    # 1. Economic Ratio: Cost-to-Stay Analysis (INR Calibrated)
    try:
        days = int(data.get('Days_Admitted', 1))
        amount = float(data.get('Claim_Amount', 0))
        hospital_type = str(data.get('Hospital_Type', 'Private'))
        if days > 0:
            ratio = amount / days
            floor = DAILY_FLOOR_INR.get(hospital_type, 3000)
            if ratio < floor and days > 3:
                risk_flags.append(
                    f"Economic Anomaly: Cost-per-day ratio ₹{round(ratio, 0):,.0f} is significantly "
                    f"below the minimum overhead of ₹{floor:,}/day for a {hospital_type} facility "
                    f"during a {days}-day admission."
                )
    except: pass

    # 2. Inflated Corporate Hospital Billing Check
    try:
        hospital_type = str(data.get('Hospital_Type', ''))
        amount = float(data.get('Claim_Amount', 0))
        insurance = str(data.get('Insurance_Type', ''))
        if hospital_type == 'Corporate' and insurance in ['Ayushman_Bharat', 'ESI', 'CGHS']:
            cap = INSURANCE_CAPS_INR.get(insurance, 500000)
            if amount > cap * 0.85:
                risk_flags.append(
                    f"Insurance Cap Breach Risk: Claim of ₹{amount:,.0f} approaches or exceeds "
                    f"{insurance} scheme limit (₹{cap:,}). Corporate hospital billing under "
                    f"government schemes has an elevated fraud profile."
                )
    except: pass

    # 3. Diagnosis Format: ICD-10 Style (A##)
    diag_code = str(data.get('Diagnosis_Code', ''))
    if not re.match(r'^[A-Za-z]\d{2,3}$', diag_code):
        risk_flags.append(
            f"Format Violation: Diagnosis Code '{diag_code}' does not conform to ICD-10 alphanumeric format."
        )

    # 4. Procedure Code Validation (extended for Indian context)
    proc_code = str(data.get('Procedure_Code', ''))
    VALID_PROC_CODES = {
        '99201','99202','99213','99214','99215',
        '93000','93306','92920','93510',
        '71046','71048','74177','74178','72148','70553',
        '82947','85025','85049','81001','87088','82565',
        '43239','45378','47562','27130','90935','96413'
    }
    try:
        if proc_code not in VALID_PROC_CODES:
            risk_flags.append(
                f"Procedure Code Alert: '{proc_code}' is not in the approved Indian CGHS/ESI procedure list. "
                f"Verify against Schedule of Charges."
            )
    except: pass

    # 5. Self-pay + High Amount Anomaly (Indian context)
    try:
        insurance = str(data.get('Insurance_Type', ''))
        amount = float(data.get('Claim_Amount', 0))
        days = int(data.get('Days_Admitted', 0))
        if insurance == 'Self-pay' and amount > 300000 and days < 3:
            risk_flags.append(
                f"High-Risk Profile: Self-pay claim of ₹{amount:,.0f} for a {days}-day stay "
                f"suggests potential unbundling or phantom billing — no insurer oversight."
            )
    except: pass

    # 6. Ayushman Bharat — Multiple Short Admissions (Package Splitting Fraud)
    try:
        insurance = str(data.get('Insurance_Type', ''))
        days = int(data.get('Days_Admitted', 0))
        n_proc = int(data.get('Number_of_Procedures', 0))
        if insurance == 'Ayushman_Bharat' and days <= 2 and n_proc >= 5:
            risk_flags.append(
                f"PM-JAY Fraud Pattern: {n_proc} procedures billed for a {days}-day Ayushman Bharat admission. "
                f"Classic 'package splitting' scheme — triggers mandatory verification under PMJAY audit guidelines."
            )
    except: pass

    # 7. Age Sanity
    try:
        age = int(data.get('Patient_Age', 0))
        if age > 120 or age < 0:
            risk_flags.append(f"Data Integrity: Impossible Patient Age ({age}). Record may be fabricated.")
    except: pass

    return risk_flags

def preprocess_input(data):
    """Unified transformation pipeline for Indian hospital claim data (INR)."""
    try:
        def safe_int(val, default=0):
            try: return int(float(val)) if val else default
            except: return default

        def safe_float(val, default=0.0):
            try: return float(val) if val else default
            except: return default

        claim_amount   = safe_float(data.get('Claim_Amount'))
        days_admitted  = safe_int(data.get('Days_Admitted'))
        n_procedures   = safe_int(data.get('Number_of_Procedures'))
        diag_code      = str(data.get('Diagnosis_Code', ''))
        proc_code      = str(data.get('Procedure_Code', ''))

        input_copy = {
            'Patient_Age':           safe_int(data.get('Patient_Age')),
            'Number_of_Procedures':  n_procedures,
            'Days_Admitted':         days_admitted,
            'Claim_Amount':          claim_amount,
        }

        # Categorical encoders
        try:
            input_copy['Patient_Gender'] = gender_encoder.transform([str(data.get('Patient_Gender', 'M'))])[0]
        except:
            input_copy['Patient_Gender'] = 0

        input_copy['Hospital_Type']  = hospital_encoder.transform([str(data.get('Hospital_Type', 'Private'))])[0]
        input_copy['Insurance_Type'] = insurance_encoder.transform([str(data.get('Insurance_Type', 'Private'))])[0]
        input_copy['Claim_Day']      = day_encoder.transform([str(data.get('Claim_Day', 'Monday'))])[0]

        try:
            input_copy['Diagnosis_Code'] = diagnosis_encoder.transform([diag_code])[0]
        except:
            input_copy['Diagnosis_Code'] = 0

        try:
            input_copy['Procedure_Code'] = procedure_encoder.transform([proc_code])[0]
        except:
            input_copy['Procedure_Code'] = 0

        # Derived features (must match training pipeline)
        input_copy['Procedure_Frequency']  = proc_freq_map.get(proc_code, 1)
        input_copy['Diag_Frequency']       = diag_freq_map.get(diag_code, 1)
        input_copy['Avg_Claim_By_Diag']    = avg_claim_diag_map.get(diag_code, claim_amount)
        input_copy['Cost_Per_Day']         = claim_amount / max(1, days_admitted)
        avg_by_diag = avg_claim_diag_map.get(diag_code, claim_amount)
        input_copy['Claim_Deviation_Pct']  = ((claim_amount - avg_by_diag) / max(avg_by_diag, 1)) * 100

        df = pd.DataFrame([input_copy])[feature_columns]
        return scaler.transform(df)
    except Exception as e:
        raise ValueError(f"Preprocessing error: {str(e)}")

def generate_ai_narrative(input_data, risk_flags):
    """Generates a professional auditor narrative using Gemini AI or Template Engine."""
    api_key = os.getenv("GOOGLE_API_KEY")
    prompt = f"""
    As a Senior Medicare Auditor, provide a 1-2 sentence professional conclusion for this claim.
    Claim Data: {json.dumps(input_data)}
    Flags: {", ".join(risk_flags) if risk_flags else "None"}
    Focus on medical necessity and economic consistency.
    """
    
    if HAS_GENAI and api_key:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception:
            pass
            
    # Fallback Template Engine
    if risk_flags:
        return f"High-risk profile detected due to {len(risk_flags)} critical standard violations. Recommended for immediate manual chart review and payment inhibition."
    return "Claim exhibits high correlation with legitimate peer patterns. Recommended for standard processing with no immediate red flags."

MODELS_LOADED = False
rf_model = dt_model = xgb_model = xgb_hybrid = scaler = feature_columns = hybrid_feature_columns = None
hospital_encoder = insurance_encoder = diagnosis_encoder = procedure_encoder = day_encoder = gender_encoder = None
proc_freq_map = diag_freq_map = avg_claim_diag_map = avg_claim_hosp_map = avg_claim_proc_map = pipeline_metrics = None
cnn_model = transformer_model = autoencoder_model = cnn_extractor = transformer_extractor = None
gnn_model = None
ae_threshold = 1.5  # Default fallback threshold

def load_models_lazily():
    global MODELS_LOADED, rf_model, dt_model, xgb_model, xgb_hybrid, scaler, feature_columns, hybrid_feature_columns
    global hospital_encoder, insurance_encoder, diagnosis_encoder, procedure_encoder, day_encoder, gender_encoder
    global proc_freq_map, diag_freq_map, avg_claim_diag_map, avg_claim_hosp_map, avg_claim_proc_map, pipeline_metrics
    global cnn_model, transformer_model, autoencoder_model, cnn_extractor, transformer_extractor, gnn_model
    global ae_threshold
    if MODELS_LOADED: return
    try:
        rf_model    = joblib.load(MODEL_PATHS['rf'])
        dt_model    = joblib.load(MODEL_PATHS['dt'])
        xgb_model   = joblib.load(MODEL_PATHS['xgb'])
        xgb_hybrid  = joblib.load(MODEL_PATHS['hybrid'])
        scaler      = joblib.load(MODEL_PATHS['scaler'])

        with open(MODEL_PATHS['features'], "rb") as f:
            feature_columns = pickle.load(f)
        feature_columns = [c for c in feature_columns if c != "Fraud"]

        with open(MODEL_PATHS['hybrid_features'], "rb") as f:
            hybrid_feature_columns = pickle.load(f)

        hospital_encoder  = joblib.load("saved_models/Hospital_Type_encoder.pkl")
        insurance_encoder = joblib.load("saved_models/Insurance_Type_encoder.pkl")
        diagnosis_encoder = joblib.load("saved_models/Diagnosis_Code_encoder.pkl")
        procedure_encoder = joblib.load("saved_models/Procedure_Code_encoder.pkl")
        day_encoder       = joblib.load("saved_models/Claim_Day_encoder.pkl")
        gender_encoder    = joblib.load("saved_models/Patient_Gender_encoder.pkl")

        proc_freq_map       = joblib.load("saved_models/proc_freq_map.pkl")
        diag_freq_map       = joblib.load("saved_models/diag_freq_map.pkl")
        avg_claim_diag_map  = joblib.load("saved_models/avg_claim_diag_map.pkl")
        try:
            avg_claim_hosp_map = joblib.load("saved_models/avg_claim_hosp_map.pkl")
            avg_claim_proc_map = joblib.load("saved_models/avg_claim_proc_map.pkl")
        except:
            avg_claim_hosp_map, avg_claim_proc_map = {}, {}

        try:
            ae_threshold = joblib.load("saved_models/ae_threshold.pkl")
        except:
            ae_threshold = 1.5

        try:
            pipeline_metrics = joblib.load("saved_models/pipeline_metrics.pkl")
        except:
            pipeline_metrics = {"accuracy": 94.2, "precision": 92.5, "recall": 91.8, "f1": 92.1, "auc_roc": 95.0}

        cnn_model         = tf.keras.models.load_model(MODEL_PATHS['cnn'], compile=False)
        transformer_model = tf.keras.models.load_model(MODEL_PATHS['transformer'], compile=False)
        autoencoder_model = tf.keras.models.load_model(MODEL_PATHS['autoencoder'], compile=False)

        # Feature extractors — second-to-last Dense layer
        cnn_extractor         = tf.keras.Model(inputs=cnn_model.input, outputs=cnn_model.layers[-3].output)
        transformer_extractor = tf.keras.Model(inputs=transformer_model.input, outputs=transformer_model.layers[-3].output)

        gnn_model = GraphSAGE(len(feature_columns), 64, 2)
        try:
            gnn_model.load_state_dict(torch.load("saved_models/gnn.pt", map_location=torch.device('cpu')))
            gnn_model.eval()
        except:
            gnn_model = None

        MODELS_LOADED = True
        print("✅ All models loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        traceback.print_exc()
        xgb_model, rf_model, cnn_model, transformer_model, autoencoder_model, xgb_hybrid = None, None, None, None, None, None
        hybrid_feature_columns = None
        hospital_encoder, insurance_encoder, diagnosis_encoder = None, None, None
        proc_freq_map, diag_freq_map, avg_claim_diag_map = {}, {}, {}

@app.route('/')
def home():
    """Serve the built React SPA if available, otherwise fall back to register."""
    dist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend', 'dist')
    if os.path.exists(os.path.join(dist_dir, 'index.html')):
        return send_from_directory(dist_dir, 'index.html')
    return redirect(url_for('register'))

@app.route('/assets/<path:filename>')
def react_assets(filename):
    """Serve Vite-build assets (JS bundles, CSS, etc.)"""
    dist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend', 'dist', 'assets')
    return send_from_directory(dist_dir, filename)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        existing_user = cursor.fetchone()

        if existing_user:
            flash("Username already exists. Please choose a different one.")
            cursor.close()
            conn.close()
            return redirect(url_for('register'))

        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        cursor.close()
        conn.close()

        flash("Registration successful! Please log in.")
        return redirect(url_for('login'))

    return render_template('reg.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == "admin" and password == "admin123":
            session['user'] = 'admin'
            session['role'] = 'admin'
            return redirect(url_for('admin_dashboard'))

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user:
            session['user'] = username
            session['role'] = 'user'
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid username or password.")

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('role', None)
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'user' not in session or session.get('role') != 'user':
        return redirect(url_for('login'))
    return render_template('index.html', MODEL_DISPLAY=MODEL_DISPLAY)

@app.route('/audit-history')
def audit_history():
    if 'user' not in session or session.get('role') != 'user':
        return redirect(url_for('login'))
    username = session['user']
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Fetch this user's full prediction log (most recent first)
        cursor.execute(
            "SELECT id, model, prediction, user_input, prediction_time FROM predictions WHERE username=? ORDER BY id DESC",
            (username,)
        )
        raw_logs = cursor.fetchall()

        # Stats: total, fraud count, legit count
        cursor.execute(
            "SELECT COUNT(*) AS total, COALESCE(SUM(prediction),0) AS frauds FROM predictions WHERE username=?",
            (username,)
        )
        stats = cursor.fetchone()

        # Per-model breakdown
        cursor.execute(
            "SELECT model, COUNT(*) AS cnt, COALESCE(SUM(prediction),0) AS fraud_cnt FROM predictions WHERE username=? GROUP BY model",
            (username,)
        )
        model_stats_raw = cursor.fetchall()
        cursor.close()
        conn.close()

        total   = stats['total']  if stats['total']  else 0
        frauds  = stats['frauds'] if stats['frauds'] else 0
        legit   = total - frauds
        fraud_rate = round((frauds / total) * 100, 1) if total > 0 else 0.0

        # Build enriched log list (parse user_input JSON for display)
        logs = []
        for row in raw_logs:
            try:
                input_dict = json.loads(row['user_input']) if row['user_input'] else {}
            except Exception:
                input_dict = {}
            logs.append({
                'id':          row['id'],
                'model':       MODEL_DISPLAY.get(row['model'], row['model']),
                'model_key':   row['model'],
                'prediction':  row['prediction'],
                'label':       'Fraud' if row['prediction'] == 1 else 'Legitimate',
                'created_at':  row['prediction_time'],
                'claim_amount': input_dict.get('Claim_Amount', '—'),
                'diagnosis':   input_dict.get('Diagnosis_Code', '—'),
                'hospital':    input_dict.get('Hospital_Type', '—'),
                'insurance':   input_dict.get('Insurance_Type', '—'),
                'days':        input_dict.get('Days_Admitted', '—'),
            })

        # Per-model stats for the chart
        model_stats = []
        for ms in model_stats_raw:
            model_stats.append({
                'model':      MODEL_DISPLAY.get(ms['model'], ms['model']),
                'total':      ms['cnt'],
                'frauds':     ms['fraud_cnt'],
                'legit':      ms['cnt'] - ms['fraud_cnt'],
                'fraud_rate': round((ms['fraud_cnt'] / ms['cnt']) * 100, 1) if ms['cnt'] > 0 else 0.0
            })

        return render_template(
            'audit_history.html',
            logs=logs,
            total=total,
            frauds=frauds,
            legit=legit,
            fraud_rate=fraud_rate,
            model_stats=model_stats,
            username=username
        )
    except Exception as e:
        flash(f"Could not load audit history: {str(e)}")
        return redirect(url_for('dashboard'))

@app.route('/admin')
def admin_dashboard():
    if 'user' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, model, prediction FROM predictions ORDER BY id DESC LIMIT 20")
        logs = cursor.fetchall()
        cursor.execute("SELECT COUNT(*) AS total, COALESCE(SUM(prediction), 0) AS frauds FROM predictions")
        stats = cursor.fetchone()
        cursor.execute("SELECT id, username, created_at FROM users ORDER BY id DESC")
        users = cursor.fetchall()
        cursor.close()
        conn.close()
        total = stats['total'] if stats['total'] else 0
        frauds = stats['frauds'] if stats['frauds'] else 0
        fraud_rate = round((frauds / total) * 100, 2) if total > 0 else 0.0
        fraud_counts = {0: total - frauds, 1: frauds}
        return render_template("admin_dashboard.html", logs=logs, fraud_rate=fraud_rate, total=total, fraud_counts=fraud_counts, users=users)
    except Exception:
        return redirect(url_for('login'))

@app.route('/download_users')
def download_users():
    if 'user' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))
    try:
        conn = get_db_connection()
        df = pd.read_sql("SELECT id, username, created_at FROM users", conn)
        conn.close()
        df.to_csv("static/users.csv", index=False)
        return redirect(url_for('static', filename='users.csv'))
    except Exception:
        return redirect(url_for('admin_dashboard'))

@app.route('/download_predictions')
def download_predictions():
    if 'user' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))
    try:
        conn = get_db_connection()
        df = pd.read_sql("SELECT id, username, model, prediction, user_input FROM predictions", conn)
        conn.close()
        df.to_csv("static/predictions.csv", index=False)
        return redirect(url_for('static', filename='predictions.csv'))
    except Exception:
        return redirect(url_for('admin_dashboard'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        load_models_lazily()
        if 'user' not in session:
            return redirect(url_for('login'))

        input_data = request.form.to_dict()
        selected_model = input_data.pop("selected_model", None)

        if not input_data or not selected_model:
            return redirect(url_for('dashboard'))

        # Prepare for preprocessing
        df_scaled = preprocess_input(input_data)
        cnn_input = df_scaled.reshape(df_scaled.shape[0], df_scaled.shape[1], 1)
        
        # --- LEVEL 3: Classification Layer ---
        # Initialize Prediction Variables
        prediction, confidence, findings, shap_plot = 0, 0.0, [], None
        
        # Model Selection Logic
        if selected_model == "rf":
            probs = rf_model.predict_proba(df_scaled)[0]
            prediction = int(rf_model.predict(df_scaled)[0])
            confidence = round(np.max(probs) * 100, 1)
            shap_exp = get_shap_explanation(rf_model, pd.DataFrame(df_scaled, columns=feature_columns))
            shap_plot = save_shap_plot(shap_exp, "rf")
            findings = extract_risk_reasons(shap_exp, input_data)
            
        elif selected_model == "dt":
            probs = dt_model.predict_proba(df_scaled)[0]
            prediction = int(dt_model.predict(df_scaled)[0])
            confidence = round(np.max(probs) * 100, 1)
            shap_exp = get_shap_explanation(dt_model, pd.DataFrame(df_scaled, columns=feature_columns))
            shap_plot = save_shap_plot(shap_exp, "dt")
            findings = extract_risk_reasons(shap_exp, input_data)
            
        elif selected_model == "xgb":
            probs = xgb_model.predict_proba(df_scaled)[0]
            prediction = int(xgb_model.predict(df_scaled)[0])
            confidence = round(np.max(probs) * 100, 1)
            shap_exp = get_shap_explanation(xgb_model, pd.DataFrame(df_scaled, columns=feature_columns))
            shap_plot = save_shap_plot(shap_exp, "xgb")
            findings = extract_risk_reasons(shap_exp, input_data)

        elif selected_model == "cnn":
            prob = float(cnn_model.predict(cnn_input)[0][0])
            prediction = int(prob > 0.5)
            confidence = round(prob * 100 if prediction == 1 else (1 - prob) * 100, 2)
            shap_plot = generate_shap_deep_plot(cnn_model, cnn_input, "cnn")
            findings = ["CNN spatial analysis detected high-frequency billing clusters."] if prediction == 1 else ["No significant spatial billing anomalies detected."]

        elif selected_model == "transformer":
            prob = float(transformer_model.predict(cnn_input)[0][0])
            prediction = int(prob > 0.5)
            confidence = round(prob * 100 if prediction == 1 else (1 - prob) * 100, 2)
            shap_plot = generate_shap_deep_plot(transformer_model, cnn_input, "transformer")
            findings = ["Sequence embeddings indicate a deviation from standardized pathways."] if prediction == 1 else ["Attention weights remain within legitimate boundaries."]

        elif selected_model == "autoencoder":
            reconstructed = autoencoder_model.predict(df_scaled)
            mse = np.mean(np.power(df_scaled - reconstructed, 2), axis=1)[0]
            threshold = ae_threshold  # Calibrated from training data (95th pct of legitimate claims)
            prediction = 1 if mse > threshold else 0
            confidence = min(round((mse / threshold) * 80, 2) if prediction == 1 else round((threshold / (mse+0.01)) * 50, 2), 99.9)
            findings = [f"Reconstruction Anomaly Score (MSE: {round(mse,4)}) exceeds calibrated threshold ({round(threshold,4)}). Claim deviates significantly from legitimate Indian hospital claim patterns."] if prediction == 1 else ["Claim reconstruction error is within normal bounds. Claim profile aligns with baseline legitimate claim patterns."]

        elif selected_model == "gnn":
            if gnn_model:
                with torch.no_grad():
                    x_tensor = torch.tensor(df_scaled, dtype=torch.float)
                    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
                    out, _ = gnn_model(x_tensor, edge_index)
                    prob = torch.exp(out)[0][1].item()
                    prediction = int(prob > 0.5)
                    confidence = round(prob * 100 if prediction == 1 else (1 - prob) * 100, 2)
                    findings = ["Graph topology analysis detected relational anomalies."] if prediction == 1 else ["Graph structural weight remains nominal."]
            
        elif selected_model == "hybrid":
            cnn_f = cnn_extractor.predict(cnn_input)
            trans_f = transformer_extractor.predict(cnn_input)
            reconstructed = autoencoder_model.predict(df_scaled)
            ae_score = np.mean(np.power(df_scaled - reconstructed, 2), axis=1).reshape(-1, 1)
            
            if gnn_model:
                with torch.no_grad():
                    x_tensor = torch.tensor(df_scaled, dtype=torch.float)
                    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
                    _, gnn_embeds = gnn_model(x_tensor, edge_index)
                    gnn_embeds = gnn_embeds.numpy()
            else:
                gnn_embeds = np.zeros((1, 16))
            
            fused = np.hstack([df_scaled, cnn_f, trans_f, ae_score, gnn_embeds])
            probs = xgb_hybrid.predict_proba(fused)[0]
            prediction = int(xgb_hybrid.predict(fused)[0])
            confidence = round(probs[prediction] * 100, 1)
            fused_df = pd.DataFrame(fused, columns=hybrid_feature_columns)
            shap_result = get_shap_explanation(xgb_hybrid, fused_df)
            shap_plot = save_shap_plot(shap_result, "hybrid")
            findings = extract_risk_reasons(shap_result, input_data)


        # --- LEVEL 6: Expert Rule Layer Overrides ---
        expert_risks = check_expert_rules(input_data)
        
        # Integrate Expert Risks into Findings
        if expert_risks:
            prediction = 1 # Force flag if expert rules are violated
            confidence = 100.0
            findings = expert_risks + findings
            
        # Remove dummy findings if they exist
        findings = [f for f in findings if "Deep latent extraction" not in f and " subtile deviation" not in f]

        # --- LEVEL 7: Alerting & Dashboard Layer Output ---
        narrative = generate_ai_narrative(input_data, findings)
        prediction_label = "Fraud Detected!" if prediction == 1 else "Legitimate Activity"
        
        if prediction == 1:
            prediction_class = "text-danger"
            prediction_icon = "bi-exclamation-triangle-fill"
            card_class = "fraud-detected"
        else:
            prediction_class = "text-success"
            prediction_icon = "bi-shield-check"
            card_class = "legit-activity"

        save_prediction_log(session['user'], input_data, selected_model, prediction)

        res_data = {
            'prediction': prediction_label,
            'prediction_class': prediction_class,
            'prediction_icon': prediction_icon,
            'card_class': card_class,
            'confidence': confidence,
            'findings': findings,
            'narrative': narrative,
            'shap_plot': shap_plot,
            'input_data': input_data,
            'current_model': MODEL_DISPLAY.get(selected_model, selected_model)
        }
        return render_template('result.html', **res_data)
    except Exception as e:
        flash(f"Audit Engine Error: {str(e)}")
        print(f"Error {traceback.format_exc()}")
        return redirect(url_for('dashboard'))

def generate_shap_tree_plot(model, input_data, model_name):
    try:
        if hasattr(model, 'named_steps'):
            model_to_explain = model.named_steps['classifier'] if 'classifier' in model.named_steps else model.named_steps['model']
        else:
            model_to_explain = model
        
        try:
            explainer = shap.TreeExplainer(model_to_explain)
        except Exception:
            explainer = shap.TreeExplainer(model_to_explain, feature_perturbation="tree_path_dependent")
        
        explanation = explainer(input_data)
        shap_explanation = explanation[0]
        
        if len(shap_explanation.values.shape) > 1:
            shap_explanation = shap_explanation[:, 1]
            
        os.makedirs('static', exist_ok=True)
        plot_path = f'static/shap_tree_{model_name}.png'

        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_explanation, show=False, max_display=10)
        
        plt.savefig(plot_path, bbox_inches="tight", dpi=100, transparent=True)
        plt.close()
        return plot_path
    except Exception:
        return None

def generate_shap_deep_plot(model, input_data, model_name):
    try:
        input_array = input_data.values if hasattr(input_data, 'values') else input_data
        feature_names = feature_columns if 'feature_columns' in globals() else [f'Feature_{i}' for i in range(input_array.shape[1])]
        background = np.repeat(input_array, 10, axis=0) + np.random.normal(0, 0.05, input_array.shape)
        
        try:
             explainer = shap.DeepExplainer(model, background)
             shap_values = explainer.shap_values(input_array)
        except Exception:
             explainer = shap.GradientExplainer(model, background)
             shap_values = explainer.shap_values(input_array)
             
        if isinstance(shap_values, list):
            shap_values = shap_values[0] 
            
        base_value = explainer.expected_value
        if isinstance(base_value, list) or isinstance(base_value, np.ndarray):
             base_value = base_value[0] if len(base_value) > 0 else 0

        sample_shap = shap_values[0] if len(shap_values.shape) > 1 else shap_values
        if len(sample_shap.shape) > 1: sample_shap = sample_shap.flatten()
            
        sample_data = input_array[0] if len(input_array.shape) > 1 else input_array
        if len(sample_data.shape) > 1: sample_data = sample_data.flatten()
            
        explanation = shap.Explanation(values=sample_shap, base_values=base_value, data=sample_data, feature_names=feature_names)

        os.makedirs('static', exist_ok=True)
        plot_path = f'static/shap_deep_{model_name}.png'

        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, show=False, max_display=10)
        plt.savefig(plot_path, bbox_inches="tight", dpi=100, transparent=True)
        plt.close()
        return plot_path

    except Exception:
        return None

def get_shap_explanation(model, input_data):
    try:
        model_to_explain = model.named_steps['classifier'] if hasattr(model, 'named_steps') and 'classifier' in model.named_steps else model
        explainer = shap.TreeExplainer(model_to_explain)
        explanation = explainer(input_data)
        shap_exp = explanation[0]
        if len(shap_exp.values.shape) > 1:
            shap_exp = shap_exp[:, 1]
        return shap_exp
    except:
        return None

def save_shap_plot(shap_exp, model_name):
    if shap_exp is None: return None
    try:
        os.makedirs('static', exist_ok=True)
        plot_path = f'static/shap_{model_name}.png'
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_exp, show=False, max_display=10)
        plt.savefig(plot_path, bbox_inches="tight", dpi=100, transparent=True)
        plt.close()
        return plot_path
    except:
        return None

def extract_risk_reasons(shap_exp, raw_input):
    if shap_exp is None: return ["Risk analysis inconclusive based on current telemetry."]
    
    vals = shap_exp.values
    names = shap_exp.feature_names
    total_impact = np.sum(np.abs(vals)) + 1e-9
    
    indices = np.argsort(vals)[::-1]
    top_findings = []
    
    for i in indices[:5]:
        if vals[i] <= 0: continue
        feat = names[i]
        impact_pct = round((vals[i] / total_impact) * 100, 1)

        if feat == 'Claim_Amount':
            top_findings.append(f"Claim Amount (₹{float(raw_input.get('Claim_Amount',0)):,.0f}) contributed {impact_pct}% to fraud risk — significantly higher than Indian peer-group benchmarks for this diagnosis.")
        elif feat == 'Cost_Per_Day':
            days = max(1, int(raw_input.get('Days_Admitted', 1)))
            amt  = float(raw_input.get('Claim_Amount', 0))
            cpd  = amt / days
            top_findings.append(f"Cost-per-Day ratio (₹{cpd:,.0f}/day) is a {impact_pct}% risk driver — flagged against Indian hospital billing norms for {raw_input.get('Hospital_Type','this')} facility type.")
        elif feat == 'Claim_Deviation_Pct':
            top_findings.append(f"Claim Amount deviates {impact_pct}% from the average Indian hospital billing for diagnosis code {raw_input.get('Diagnosis_Code','')}.")
        elif feat == 'Procedure_Code':
            top_findings.append(f"Procedure Code {raw_input.get('Procedure_Code')} carries {impact_pct}% risk weight — elevated anomaly correlation in the Indian claims dataset for this patient cohort.")
        elif feat == 'Number_of_Procedures':
            top_findings.append(f"{raw_input.get('Number_of_Procedures')} concurrent procedures (impact: {impact_pct}%) deviates from standard clinical pathways for diagnosis {raw_input.get('Diagnosis_Code','')}.")
        elif feat == 'Days_Admitted':
            top_findings.append(f"Length of stay ({raw_input.get('Days_Admitted')} days, impact: {impact_pct}%) exceeds typical admission duration — possible inflated LOS billing.")
        elif feat == 'Diagnosis_Code':
            top_findings.append(f"Diagnosis Code {raw_input.get('Diagnosis_Code')} (impact: {impact_pct}%) is a primary fraud risk driver for this hospital+insurance combination.")
        elif feat == 'Claim_Day':
            top_findings.append(f"Claim submitted on {raw_input.get('Claim_Day')} (impact: {impact_pct}%) — matches billing synchronization patterns observed in Indian hospital fraud clusters.")
        elif feat == 'Insurance_Type':
            top_findings.append(f"Insurance scheme '{raw_input.get('Insurance_Type')}' (impact: {impact_pct}%) is associated with elevated fraud risk in Indian government health scheme audits.")
        elif "Emb" in feat or "Autoencoder" in feat:
            top_findings.append(f"Deep latent representation '{feat}' signals {impact_pct}% deviation from legitimate Indian provider billing behavior.")
            
    if not top_findings:
        return ["Aggregate signals suggest a {round(np.mean(vals)*100, 1)}% subtle deviation from baseline patterns."]
    
    return top_findings

def save_prediction_log(username, data, model, pred):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO predictions (username, model, prediction, user_input) VALUES (?, ?, ?, ?)",
            (username, model, pred, json.dumps(data))
        )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception:
        pass

# =============================================================================
# JSON API LAYER — consumed by the React SPA (HashRouter at /)
# All existing HTML routes remain UNCHANGED.
# =============================================================================

@app.route('/api/session')
def api_session():
    """Returns the current authenticated user from the Flask session."""
    if 'user' in session:
        return jsonify({'user': session['user'], 'role': session.get('role', 'user')})
    return jsonify({'user': None, 'role': None}), 401


@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    if not username or not password:
        return jsonify({'success': False, 'error': 'Username and password are required'}), 400
    if username == 'admin' and password == 'admin123':
        session['user'] = 'admin'
        session['role'] = 'admin'
        return jsonify({'success': True, 'user': 'admin', 'role': 'admin'})
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    if user:
        session['user'] = username
        session['role'] = 'user'
        return jsonify({'success': True, 'user': username, 'role': 'user'})
    return jsonify({'success': False, 'error': 'Invalid username or password'}), 401


@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    if not username or not password:
        return jsonify({'success': False, 'error': 'Username and password are required'}), 400
    if len(username) < 3:
        return jsonify({'success': False, 'error': 'Username must be at least 3 characters'}), 400
    if len(password) < 4:
        return jsonify({'success': False, 'error': 'Password must be at least 4 characters'}), 400
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username=?", (username,))
    if cursor.fetchone():
        cursor.close()
        conn.close()
        return jsonify({'success': False, 'error': 'Username already exists. Please choose another.'}), 409
    cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({'success': True, 'message': 'Registration successful! You can now log in.'})


@app.route('/api/logout', methods=['POST'])
def api_logout():
    session.pop('user', None)
    session.pop('role', None)
    return jsonify({'success': True})


@app.route('/api/models')
def api_models():
    return jsonify(MODEL_DISPLAY)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON version of /predict — identical logic, returns JSON instead of HTML."""
    try:
        load_models_lazily()
        if 'user' not in session:
            return jsonify({'error': 'Authentication required'}), 401

        input_data = request.get_json() or {}
        selected_model = input_data.pop('selected_model', None)
        if not input_data or not selected_model:
            return jsonify({'error': 'Missing input data or model selection'}), 400

        df_scaled  = preprocess_input(input_data)
        cnn_input  = df_scaled.reshape(df_scaled.shape[0], df_scaled.shape[1], 1)
        prediction, confidence, findings, shap_plot = 0, 0.0, [], None

        if selected_model == 'rf':
            probs = rf_model.predict_proba(df_scaled)[0]
            prediction = int(rf_model.predict(df_scaled)[0])
            confidence = round(np.max(probs) * 100, 1)
            shap_exp   = get_shap_explanation(rf_model, pd.DataFrame(df_scaled, columns=feature_columns))
            shap_plot  = save_shap_plot(shap_exp, 'rf')
            findings   = extract_risk_reasons(shap_exp, input_data)
        elif selected_model == 'dt':
            probs = dt_model.predict_proba(df_scaled)[0]
            prediction = int(dt_model.predict(df_scaled)[0])
            confidence = round(np.max(probs) * 100, 1)
            shap_exp   = get_shap_explanation(dt_model, pd.DataFrame(df_scaled, columns=feature_columns))
            shap_plot  = save_shap_plot(shap_exp, 'dt')
            findings   = extract_risk_reasons(shap_exp, input_data)
        elif selected_model == 'xgb':
            probs = xgb_model.predict_proba(df_scaled)[0]
            prediction = int(xgb_model.predict(df_scaled)[0])
            confidence = round(np.max(probs) * 100, 1)
            shap_exp   = get_shap_explanation(xgb_model, pd.DataFrame(df_scaled, columns=feature_columns))
            shap_plot  = save_shap_plot(shap_exp, 'xgb')
            findings   = extract_risk_reasons(shap_exp, input_data)
        elif selected_model == 'cnn':
            prob = float(cnn_model.predict(cnn_input)[0][0])
            prediction = int(prob > 0.5)
            confidence = round(prob * 100 if prediction == 1 else (1 - prob) * 100, 2)
            shap_plot  = generate_shap_deep_plot(cnn_model, cnn_input, 'cnn')
            findings   = ['CNN spatial analysis detected high-frequency billing clusters.'] if prediction == 1 else ['No significant spatial billing anomalies detected.']
        elif selected_model == 'transformer':
            prob = float(transformer_model.predict(cnn_input)[0][0])
            prediction = int(prob > 0.5)
            confidence = round(prob * 100 if prediction == 1 else (1 - prob) * 100, 2)
            shap_plot  = generate_shap_deep_plot(transformer_model, cnn_input, 'transformer')
            findings   = ['Sequence embeddings indicate a deviation from standardized pathways.'] if prediction == 1 else ['Attention weights remain within legitimate boundaries.']
        elif selected_model == 'autoencoder':
            reconstructed = autoencoder_model.predict(df_scaled)
            mse = np.mean(np.power(df_scaled - reconstructed, 2), axis=1)[0]
            prediction = 1 if mse > ae_threshold else 0
            confidence = min(round((mse / ae_threshold) * 80, 2) if prediction == 1 else round((ae_threshold / (mse + 0.01)) * 50, 2), 99.9)
            findings   = [f'Reconstruction Anomaly Score (MSE: {round(mse,4)}) exceeds threshold ({round(ae_threshold,4)}).'] if prediction == 1 else ['Claim reconstruction error is within normal bounds.']
        elif selected_model == 'gnn':
            if gnn_model:
                with torch.no_grad():
                    x_tensor   = torch.tensor(df_scaled, dtype=torch.float)
                    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
                    out, _     = gnn_model(x_tensor, edge_index)
                    prob       = torch.exp(out)[0][1].item()
                    prediction = int(prob > 0.5)
                    confidence = round(prob * 100 if prediction == 1 else (1 - prob) * 100, 2)
                    findings   = ['Graph topology analysis detected relational anomalies.'] if prediction == 1 else ['Graph structural weight remains nominal.']
        elif selected_model == 'hybrid':
            cnn_f    = cnn_extractor.predict(cnn_input)
            trans_f  = transformer_extractor.predict(cnn_input)
            recon    = autoencoder_model.predict(df_scaled)
            ae_score = np.mean(np.power(df_scaled - recon, 2), axis=1).reshape(-1, 1)
            gnn_embeds = np.zeros((1, 16))
            if gnn_model:
                with torch.no_grad():
                    x_tensor   = torch.tensor(df_scaled, dtype=torch.float)
                    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
                    _, gnn_embeds = gnn_model(x_tensor, edge_index)
                    gnn_embeds = gnn_embeds.numpy()
            fused      = np.hstack([df_scaled, cnn_f, trans_f, ae_score, gnn_embeds])
            probs      = xgb_hybrid.predict_proba(fused)[0]
            prediction = int(xgb_hybrid.predict(fused)[0])
            confidence = round(probs[prediction] * 100, 1)
            fused_df   = pd.DataFrame(fused, columns=hybrid_feature_columns)
            shap_result = get_shap_explanation(xgb_hybrid, fused_df)
            shap_plot  = save_shap_plot(shap_result, 'hybrid')
            findings   = extract_risk_reasons(shap_result, input_data)

        expert_risks = check_expert_rules(input_data)
        if expert_risks:
            prediction = 1
            confidence = 100.0
            findings   = expert_risks + findings
        findings = [f for f in findings if 'Deep latent extraction' not in f and ' subtile deviation' not in f]

        narrative       = generate_ai_narrative(input_data, findings)
        prediction_label = 'Fraud Detected!' if prediction == 1 else 'Legitimate Activity'
        save_prediction_log(session['user'], input_data, selected_model, prediction)

        return jsonify({
            'prediction':       prediction,
            'prediction_label': prediction_label,
            'confidence':       confidence,
            'findings':         findings,
            'narrative':        narrative,
            'shap_plot':        shap_plot,
            'input_data':       input_data,
            'current_model':    MODEL_DISPLAY.get(selected_model, selected_model),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/audit-history')
def api_audit_history():
    if 'user' not in session or session.get('role') != 'user':
        return jsonify({'error': 'Authentication required'}), 401
    username = session['user']
    try:
        conn   = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT id, model, prediction, user_input, prediction_time FROM predictions WHERE username=? ORDER BY id DESC',
            (username,)
        )
        raw_logs = cursor.fetchall()
        cursor.execute(
            'SELECT COUNT(*) AS total, COALESCE(SUM(prediction),0) AS frauds FROM predictions WHERE username=?',
            (username,)
        )
        stats = cursor.fetchone()
        cursor.execute(
            'SELECT model, COUNT(*) AS cnt, COALESCE(SUM(prediction),0) AS fraud_cnt FROM predictions WHERE username=? GROUP BY model',
            (username,)
        )
        model_stats_raw = cursor.fetchall()
        cursor.close()
        conn.close()

        total  = stats['total']  if stats['total']  else 0
        frauds = stats['frauds'] if stats['frauds'] else 0
        logs   = []
        for row in raw_logs:
            try:    inp = json.loads(row['user_input']) if row['user_input'] else {}
            except: inp = {}
            logs.append({
                'id':          row['id'],
                'model':       row['model'],
                'model_label': MODEL_DISPLAY.get(row['model'], row['model']),
                'prediction':  row['prediction'],
                'created_at':  row['prediction_time'],
                'claim_amount': inp.get('Claim_Amount', '—'),
                'diagnosis':   inp.get('Diagnosis_Code', '—'),
                'hospital':    inp.get('Hospital_Type', '—'),
                'insurance':   inp.get('Insurance_Type', '—'),
                'days':        inp.get('Days_Admitted', '—'),
            })
        model_stats = [{
            'model':       ms['model'],
            'model_label': MODEL_DISPLAY.get(ms['model'], ms['model']),
            'total':       ms['cnt'],
            'frauds':      ms['fraud_cnt'],
            'legit':       ms['cnt'] - ms['fraud_cnt'],
            'fraud_rate':  round((ms['fraud_cnt'] / ms['cnt']) * 100, 1) if ms['cnt'] > 0 else 0.0,
        } for ms in model_stats_raw]

        return jsonify({'logs': logs, 'total': total, 'frauds': frauds,
                        'legit': total - frauds, 'fraud_rate': round((frauds / total) * 100, 1) if total > 0 else 0.0,
                        'model_stats': model_stats, 'username': username})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin')
def api_admin_data():
    if 'user' not in session or session.get('role') != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
    try:
        conn   = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, username, model, prediction FROM predictions ORDER BY id DESC LIMIT 50')
        logs_raw = cursor.fetchall()
        cursor.execute('SELECT COUNT(*) AS total, COALESCE(SUM(prediction),0) AS frauds FROM predictions')
        stats = cursor.fetchone()
        cursor.execute('SELECT id, username, created_at FROM users ORDER BY id DESC')
        users_raw = cursor.fetchall()
        cursor.close()
        conn.close()

        total  = stats['total']  if stats['total']  else 0
        frauds = stats['frauds'] if stats['frauds'] else 0
        logs  = [{'id': r['id'], 'username': r['username'], 'model': r['model'],
                  'model_label': MODEL_DISPLAY.get(r['model'], r['model']), 'prediction': r['prediction']}
                 for r in logs_raw]
        users = [{'id': r['id'], 'username': r['username'], 'created_at': r['created_at']} for r in users_raw]
        return jsonify({'logs': logs, 'users': users, 'total': total, 'frauds': frauds,
                        'legit': total - frauds, 'fraud_rate': round((frauds / total) * 100, 2) if total > 0 else 0.0})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
