import pandas as pd
import numpy as np

for filename in ['dataset/medicare_5000.csv']:
    np.random.seed(42)
    df = pd.read_csv(filename, encoding='latin1')

    z = (
        (df['Claim_Amount'] - 5000) / 2000 * 2.0 + 
        (df['Number_of_Procedures'] - 4) / 2 * 1.5 + 
        (df['Days_Admitted'] - 14) / 5 * 1.0 + 
        (df['Procedure_Code'] == 99215) * 1.5 + 
        (df['Diagnosis_Code'] == 'E11') * 1.0
    )

    z += np.random.normal(0, 1.0, size=len(df))
    probs = 1 / (1 + np.exp(-z))
    threshold = np.percentile(probs, 85)
    df['Fraud'] = (probs > threshold).astype(int)

    print(f"\n--- {filename} New fraud stats ---")
    print(df["Fraud"].value_counts())
    cols = ['Claim_Amount','Number_of_Procedures','Days_Admitted']
    for c in cols:
        fraud_m = df[df['Fraud']==1][c].mean()
        legit_m = df[df['Fraud']==0][c].mean()
        print(f"{c}: Fraud mean={fraud_m:.2f}, Legit mean={legit_m:.2f}")

    df.to_csv(filename, index=False)
