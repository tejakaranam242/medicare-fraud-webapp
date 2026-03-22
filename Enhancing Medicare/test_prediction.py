import requests

# Create session
s = requests.Session()
host = "http://127.0.0.1:5000"

# Register/Login
try:
    print("Attempting to Register...")
    s.post(f"{host}/register", data={"username": "debug_user", "password": "password", "role": "user"})
    
    print("Attempting to Login...")
    login_resp = s.post(f"{host}/login", data={"username": "debug_user", "password": "password"})
    if "dashboard" in login_resp.url or "Login successful" in login_resp.text:
        print("Login Successful")
    else:
        print("Login might have failed")
        
    print("Making Prediction...")
    data = {
        "Claim_Amount": "95000",
        "Procedure_Code": "99215",
        "Diagnosis_Code": "I10",
        "Number_of_Procedures": "5",
        "Days_Admitted": "10",
        "Patient_Age": "75",
        "Patient_Gender": "M",
        "Hospital_Type": "General",
        "Insurance_Type": "Medicare",
        "Claim_Day": "Monday",
        "selected_model": "cnn"
    }
    data["selected_model"] = "cnn"
    try:
        response = s.post(f"{host}/predict", data=data, timeout=30)
    except requests.exceptions.Timeout:
        print("Error: Request Timed Out (SHAP likely still hanging)")
        exit(1)
    
    print(f"Status Code: {response.status_code}")
    response_text = response.text
    if "Fraud" in response_text:
        print("Prediction Found in HTML")
    if "shap_deep_cnn.png" in response_text:
        print("SHAP Plot URL Found: shap_deep_cnn.png")
    elif "fallback" in response_text:
        print("SHAP Fallback Plot Found")
    else:
        print("SHAP Plot NOT Found")
        
    # Print a safe snippet
    print(response_text.encode('ascii', 'ignore').decode('ascii')[:1000])

except Exception as e:
    print(f"Error: {e}")
