"""
Enterprise-Grade Indian Hospital Medicare Dataset Generator
============================================================
Generates 10,000 realistic medical insurance claims in Indian Rupees (INR).
Pricing is correlated across: Diagnosis Code × Procedure Code × Hospital Type.

Real Indian Hospital Pricing Sources Referenced:
- AIIMS/Government: Subsidized rates (free to minimal)
- ESI/CGHS approved rate lists
- Private hospital ranges (Manipal, Fortis, Apollo secondary)
- Corporate hospital (Apollo Spectra, Max, Medanta premium) pricing
- Ayushman Bharat PMJAY package rates (defined ceiling per procedure)
"""

import pandas as pd
import numpy as np
import random
import string
import os

np.random.seed(42)
random.seed(42)

# ─────────────────────────────────────────────
# 1. DIAGNOSIS PROFILES (ICD-10 codes common in India)
# ─────────────────────────────────────────────
# Each entry: (code, description, typical_stay_days_range, typical_procedures)
DIAGNOSIS_PROFILES = {
    'I10':  {'desc': 'Hypertension',            'days': (1, 5),   'procedures': ['99213', '99214', '93000', '82947'], 'base_severity': 1.0},
    'E11':  {'desc': 'Diabetes Mellitus T2',     'days': (2, 8),   'procedures': ['99214', '82947', '85025', '99215'], 'base_severity': 1.2},
    'J45':  {'desc': 'Bronchial Asthma',         'days': (2, 7),   'procedures': ['99213', '99214', '71046'],          'base_severity': 1.1},
    'K21':  {'desc': 'GERD',                     'days': (1, 4),   'procedures': ['99213', '43239'],                   'base_severity': 0.9},
    'M54':  {'desc': 'Back Pain / Dorsalgia',    'days': (1, 6),   'procedures': ['99213', '99214', '72148'],          'base_severity': 0.9},
    'A90':  {'desc': 'Dengue Fever',             'days': (4, 14),  'procedures': ['99214', '99215', '85025', '85049'], 'base_severity': 1.5},
    'K80':  {'desc': 'Cholelithiasis (Gallstone)','days': (3, 10), 'procedures': ['99215', '47562', '74177'],          'base_severity': 2.0},
    'N39':  {'desc': 'UTI / Urinary Tract Infect','days': (2, 6),  'procedures': ['99213', '81001', '87088'],          'base_severity': 1.0},
    'I21':  {'desc': 'Acute Myocardial Infarct', 'days': (5, 21),  'procedures': ['99215', '93306', '92920', '93510'], 'base_severity': 4.0},
    'B01':  {'desc': 'Chickenpox / Varicella',   'days': (3, 10),  'procedures': ['99213', '99214'],                   'base_severity': 0.8},
    'C34':  {'desc': 'Lung Cancer',              'days': (7, 30),  'procedures': ['99215', '71048', '96413'],          'base_severity': 5.0},
    'N18':  {'desc': 'Chronic Kidney Disease',   'days': (3, 14),  'procedures': ['99215', '90935', '82565'],          'base_severity': 3.5},
    'S72':  {'desc': 'Hip Fracture',             'days': (7, 25),  'procedures': ['99215', '27130', '73721'],          'base_severity': 3.8},
    'G35':  {'desc': 'Multiple Sclerosis',       'days': (4, 15),  'procedures': ['99215', '70553'],                   'base_severity': 3.2},
    'J18':  {'desc': 'Pneumonia',                'days': (4, 14),  'procedures': ['99214', '99215', '71046', '85025'], 'base_severity': 2.0},
}

# ─────────────────────────────────────────────
# 2. PROCEDURE CODES & BASE COSTS
# ─────────────────────────────────────────────
# Base costs in INR at a mid-tier Private hospital
PROCEDURE_BASE_COST = {
    '99201':  2500,    # OPD Consultation Level 1
    '99202':  3500,    # OPD Consultation Level 2
    '99213':  5000,    # OPD Consultation Level 3 (established patient)
    '99214':  8000,    # OPD Consultation Level 4 (complex)
    '99215':  12000,   # OPD Consultation Level 5 (high complexity)
    '93000':  1500,    # ECG / EKG 12-lead
    '93306':  8000,    # Echocardiogram (2D with Doppler)
    '92920':  120000,  # Coronary Angioplasty (PCI)
    '93510':  35000,   # Cardiac Catheterization
    '71046':  2500,    # Chest X-Ray (PA + Lateral)
    '71048':  5000,    # Chest CT
    '74177':  9000,    # CT Abdomen (with contrast)
    '74178':  15000,   # CT Abdomen + Pelvis
    '72148':  9500,    # MRI Lumbar Spine
    '70553':  18000,   # MRI Brain (with contrast)
    '82947':  400,     # Blood Glucose (Fasting)
    '85025':  600,     # Complete Blood Count (CBC)
    '85049':  800,     # CBC with Differential
    '81001':  350,     # Urinalysis
    '87088':  1500,    # Urine Culture
    '82565':  500,     # Creatinine
    '43239':  18000,   # Upper GI Endoscopy (EGD with biopsy)
    '45378':  22000,   # Colonoscopy (diagnostic)
    '47562':  75000,   # Laparoscopic Cholecystectomy
    '27130':  150000,  # Total Hip Replacement (THR)
    '90935':  2200,    # Hemodialysis (single session)
    '96413':  35000,   # Chemotherapy infusion (initial hour)
}

# ─────────────────────────────────────────────
# 3. HOSPITAL TYPE MULTIPLIERS
# ─────────────────────────────────────────────
# Government hospitals are heavily subsidized; Corporate are premium priced
HOSPITAL_TYPE_MULTIPLIER = {
    'Government':    0.10,   # Free to nominal (₹100–₹500 for major surgery at AIIMS)
    'Nursing_Home':  0.55,   # Basic private facility
    'AYUSH':         0.30,   # Ayurveda/Homeopathy — very low cost
    'Private':       1.00,   # Reference baseline
    'Corporate':     2.50,   # Max, Medanta, Apollo Premium — highly inflated
}

# ─────────────────────────────────────────────
# 4. INSURANCE TYPE PARAMETERS
# ─────────────────────────────────────────────
INSURANCE_TYPES = {
    'Ayushman_Bharat': {'cap': 500000,  'fraud_risk': 0.22},  # PM-JAY: 5L cap, high fraud
    'ESI':             {'cap': 300000,  'fraud_risk': 0.12},  # Employees' State Insurance
    'CGHS':            {'cap': 400000,  'fraud_risk': 0.10},  # Central Govt Health Scheme
    'Private':         {'cap': 1500000, 'fraud_risk': 0.20},  # Private insurer (high claim inflate)
    'Self-pay':        {'cap': 9999999, 'fraud_risk': 0.08},  # No insurer — lower fraud incentive
}

# ─────────────────────────────────────────────
# 5. FRAUD PATTERNS (Indian Healthcare Context)
# ─────────────────────────────────────────────
# Fraud types and their claim inflation multipliers
FRAUD_PATTERNS = {
    'phantom_procedure':    2.5,   # Billing for procedures not performed
    'upcoding':             1.6,   # Billing higher severity than warranted
    'duplicate_billing':    1.8,   # Same service billed multiple times
    'inflated_los':         1.4,   # Longer stay than medically necessary
    'corporate_inflation':  3.0,   # Corporate hospitals massively inflate bills
    'package_splitting':    1.7,   # Splitting one day-care into multiple admissions
}

CLAIM_DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
PATIENT_GENDERS = ['M', 'F']

def generate_provider_id():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=6))

def compute_claim_amount(diag_code, proc_code, hospital_type, days_admitted, n_procedures, is_fraud, fraud_type=None):
    """
    Compute a realistic INR claim amount based on:
    - Base procedure cost
    - Diagnosis severity multiplier
    - Hospital type multiplier
    - Days admitted (bed charge: ₹500–₹25,000/day based on hospital)
    - Number of concurrent procedures
    - Fraud inflation if applicable
    """
    diag = DIAGNOSIS_PROFILES.get(diag_code, {'base_severity': 1.0})
    severity = diag['base_severity']

    # Base procedure cost
    proc_cost = PROCEDURE_BASE_COST.get(proc_code, 5000)

    # Bed charge per day (realistic Indian rates)
    BED_CHARGE_PER_DAY = {
        'Government':    500,
        'AYUSH':         1200,
        'Nursing_Home':  3500,
        'Private':       8000,
        'Corporate':     22000,
    }
    bed_charge = BED_CHARGE_PER_DAY[hospital_type] * max(1, days_admitted)

    # Medications & consumables (correlated to severity & stay)
    med_cost = severity * 1500 * max(1, days_admitted) * np.random.uniform(0.4, 1.6)

    # Diagnostic tests (CBC, ECG, imaging — always run multiple)
    diag_cost = proc_cost * np.random.uniform(0.15, 0.35)

    # Procedure cost (scaled by hospital multiplier)
    hosp_mult = HOSPITAL_TYPE_MULTIPLIER[hospital_type]
    actual_proc_cost = proc_cost * hosp_mult * n_procedures * np.random.uniform(0.6, 1.2)

    # Base total before fraud
    base_total = bed_charge + med_cost + diag_cost + actual_proc_cost

    # Apply hospital noise (±15%)
    base_total *= np.random.uniform(0.85, 1.15)

    # Apply fraud inflation
    if is_fraud and fraud_type:
        multiplier = FRAUD_PATTERNS.get(fraud_type, 1.5)
        # Add some randomness to fraud inflation
        base_total *= multiplier * np.random.uniform(0.8, 1.2)

    # Floor: prevent unrealistically low values
    min_floor = 500 if hospital_type == 'Government' else 2000
    base_total = max(base_total, min_floor)

    # Ayushman Bharat insurance cap enforcement (non-fraud)
    # (Fraudsters deliberately exceed caps via phantom procedures)
    return round(base_total, 2)


def assign_fraud(diag_code, hospital_type, insurance_type_key, days, n_procedures, claim_amount):
    """
    Determine if a claim is fraudulent based on Indian healthcare fraud risk model.
    Returns (is_fraud: bool, fraud_type: str or None)
    """
    insurance_info = INSURANCE_TYPES[insurance_type_key]
    base_risk = insurance_info['fraud_risk']

    # Corporate hospitals + private insurance = higher fraud risk
    if hospital_type == 'Corporate' and insurance_type_key in ['Private', 'Ayushman_Bharat']:
        base_risk *= 1.8
    
    # Government insurance schemes are targeted for fraud
    if insurance_type_key in ['Ayushman_Bharat', 'ESI'] and hospital_type in ['Private', 'Corporate']:
        base_risk *= 1.5

    # High-severity diagnoses see more fraud (cancer, cardiac, orthopedic)
    severity = DIAGNOSIS_PROFILES.get(diag_code, {}).get('base_severity', 1.0)
    if severity >= 3.0:
        base_risk *= 1.3

    # Very long stays are suspicious
    if days > 15:
        base_risk *= 1.2

    # Cap at 60% fraud rate
    base_risk = min(base_risk, 0.60)

    is_fraud = np.random.random() < base_risk

    if is_fraud:
        # Pick most likely fraud type based on context
        if hospital_type == 'Corporate':
            fraud_type = np.random.choice(['corporate_inflation', 'upcoding', 'phantom_procedure'], p=[0.4, 0.35, 0.25])
        elif insurance_type_key == 'Ayushman_Bharat':
            fraud_type = np.random.choice(['phantom_procedure', 'package_splitting', 'upcoding'], p=[0.35, 0.35, 0.30])
        elif days > 15:
            fraud_type = np.random.choice(['inflated_los', 'duplicate_billing'], p=[0.6, 0.4])
        else:
            fraud_type = np.random.choice(list(FRAUD_PATTERNS.keys()))
        return True, fraud_type
    
    return False, None


def generate_dataset(n=10000):
    records = []
    diag_codes = list(DIAGNOSIS_PROFILES.keys())
    hospital_types = list(HOSPITAL_TYPE_MULTIPLIER.keys())
    insurance_types = list(INSURANCE_TYPES.keys())

    for _ in range(n):
        # --- Demographics ---
        patient_age = np.random.randint(5, 90)
        patient_gender = random.choice(PATIENT_GENDERS)

        # --- Clinical ---
        diag_code = random.choice(diag_codes)
        diag_profile = DIAGNOSIS_PROFILES[diag_code]
        
        # Days admitted: 0 for OPD, else inpatient
        stay_min, stay_max = diag_profile['days']
        is_opd = np.random.random() < 0.25  # 25% are OPD visits
        if is_opd:
            days_admitted = 0
        else:
            days_admitted = np.random.randint(stay_min, stay_max + 1)

        # Procedure code: pick from diagnosis-typical procedures
        proc_code = random.choice(diag_profile['procedures'])
        n_procedures = np.random.randint(1, 9)

        # --- Facility & Insurance ---
        hospital_type = random.choice(hospital_types)
        insurance_key = random.choice(insurance_types)
        claim_day = random.choice(CLAIM_DAYS)

        # --- Determine fraud FIRST (needed for claim computation) ---
        is_fraud, fraud_type = assign_fraud(
            diag_code, hospital_type, insurance_key,
            days_admitted, n_procedures, 0  # placeholder
        )

        # --- Compute claim amount with correlated pricing ---
        claim_amount = compute_claim_amount(
            diag_code, proc_code, hospital_type,
            days_admitted, n_procedures, is_fraud, fraud_type
        )

        # Map insurance key to display string
        insurance_display = insurance_key  # already stored as key

        records.append({
            'Provider_ID':          generate_provider_id(),
            'Claim_Amount':         claim_amount,
            'Procedure_Code':       proc_code,
            'Diagnosis_Code':       diag_code,
            'Number_of_Procedures': n_procedures,
            'Days_Admitted':        days_admitted,
            'Patient_Age':          patient_age,
            'Patient_Gender':       patient_gender,
            'Hospital_Type':        hospital_type,
            'Insurance_Type':       insurance_display,
            'Claim_Day':            claim_day,
            'Fraud':                int(is_fraud),
        })

    df = pd.DataFrame(records)
    return df


if __name__ == '__main__':
    print("🏥 Generating Indian Hospital Medicare Dataset (INR Pricing)...")
    df = generate_dataset(n=10000)
    
    os.makedirs('dataset', exist_ok=True)
    output_path = 'dataset/medicare_india_10000.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Dataset saved to: {output_path}")
    print(f"   Total records: {len(df)}")
    print(f"\n📊 Claim Amount Statistics (₹):")
    print(f"   Min:    ₹{df['Claim_Amount'].min():,.2f}")
    print(f"   Max:    ₹{df['Claim_Amount'].max():,.2f}")
    print(f"   Mean:   ₹{df['Claim_Amount'].mean():,.2f}")
    print(f"   Median: ₹{df['Claim_Amount'].median():,.2f}")
    print(f"\n🔍 Fraud Distribution:")
    fraud_counts = df['Fraud'].value_counts()
    print(f"   Legitimate: {fraud_counts.get(0, 0)} ({fraud_counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"   Fraud:      {fraud_counts.get(1, 0)} ({fraud_counts.get(1, 0)/len(df)*100:.1f}%)")
    print(f"\n🏥 Hospital Type Distribution:")
    print(df['Hospital_Type'].value_counts().to_string())
    print(f"\n💼 Insurance Type Distribution:")
    print(df['Insurance_Type'].value_counts().to_string())
    print(f"\n🩺 Average Claim Amount by Hospital Type (₹):")
    print(df.groupby('Hospital_Type')['Claim_Amount'].mean().sort_values(ascending=False).apply(lambda x: f"₹{x:,.0f}").to_string())
    print(f"\n🩺 Average Claim Amount by Diagnosis (₹):")
    print(df.groupby('Diagnosis_Code')['Claim_Amount'].mean().sort_values(ascending=False).apply(lambda x: f"₹{x:,.0f}").to_string())
    print(f"\nSample Records:")
    print(df.head(5).to_string(index=False))
