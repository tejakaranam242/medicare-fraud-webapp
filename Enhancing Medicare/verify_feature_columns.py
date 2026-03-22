# verify_feature_columns.py
import pickle

FEATURE_PATH = "saved_models/feature_columns.pkl"

try:
    with open(FEATURE_PATH, "rb") as f:   # ✅ always use rb
        feature_columns = pickle.load(f)
    print("✅ Feature columns loaded successfully!")
    print("Number of features:", len(feature_columns))
    print("Sample features:", feature_columns[:10])
except Exception as e:
    print("❌ Error while loading feature_columns.pkl:", e)
