import joblib
import os

folder = "saved_models"

for file in os.listdir(folder):
    if file.endswith(".pkl"):
        path = os.path.join(folder, file)
        try:
            obj = joblib.load(path)
            print(f"✅ OK: {file} -> Loaded successfully ({type(obj)})")
        except Exception as e:
            print(f"❌ CORRUPTED: {file} -> {e}")
