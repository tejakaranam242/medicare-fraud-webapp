# verify_models.py
import joblib
import pickle
import tensorflow as tf

MODEL_PATHS = {
    'cnn': 'saved_models/cnn_model.h5',
    'transformer': 'saved_models/transformer_model.h5',
    'hybrid': 'saved_models/xgb_hybrid.pkl',
    'rf': 'saved_models/rf_model.pkl',
    'dt': 'saved_models/dt_model.pkl',
    'xgb': 'saved_models/xgb_model.pkl',
    'scaler': 'saved_models/scaler.pkl',
    'features': 'saved_models/feature_columns.pkl'
}

for name, path in MODEL_PATHS.items():
    print(f"🔍 Checking {name} ...")
    try:
        if name in ['cnn', 'transformer']:
            tf.keras.models.load_model(path)
        elif name == 'features':
            with open(path, "rb") as f:
                pickle.load(f)
        else:
            joblib.load(path)
        print(f"✅ {name} loaded successfully!\n")
    except Exception as e:
        print(f"❌ Failed to load {name}: {e}\n")
