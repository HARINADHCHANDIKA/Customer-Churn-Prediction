"""
predict.py
Load the best saved model and score new customers.
Usage:
    python3 predict.py                     # scores demo customers
    python3 predict.py --csv new_customers.csv   # scores a CSV file
"""

import argparse, pickle, json
import pandas as pd
import numpy as np

MODEL_PATH = '/home/claude/churn_project/models/logistic_regression.pkl'
THRESHOLD  = 0.40   # tuned for higher recall (catch more churners)

# ── Load model bundle ────────────────────────────────────────
with open(MODEL_PATH, 'rb') as f:
    bundle = pickle.load(f)

model        = bundle['model']
preprocessor = bundle['preprocessor']
feat_names   = bundle['feature_names']

# ── Demo customers ───────────────────────────────────────────
DEMO_CUSTOMERS = pd.DataFrame([
    {   # High-risk: month-to-month, fiber, electronic check, new
        'Gender':'Male','SeniorCitizen':0,'Partner':'No','Dependents':'No',
        'TenureMonths':2,'PhoneService':'Yes','MultipleLines':'No',
        'InternetService':'Fiber optic','OnlineSecurity':'No',
        'OnlineBackup':'No','DeviceProtection':'No','TechSupport':'No',
        'StreamingTV':'Yes','StreamingMovies':'Yes','Contract':'Month-to-month',
        'PaperlessBilling':'Yes','PaymentMethod':'Electronic check',
        'MonthlyCharges':95.40,'TotalCharges':190.80,
    },
    {   # Low-risk: 2-yr contract, DSL, long tenure
        'Gender':'Female','SeniorCitizen':0,'Partner':'Yes','Dependents':'Yes',
        'TenureMonths':60,'PhoneService':'Yes','MultipleLines':'Yes',
        'InternetService':'DSL','OnlineSecurity':'Yes',
        'OnlineBackup':'Yes','DeviceProtection':'Yes','TechSupport':'Yes',
        'StreamingTV':'No','StreamingMovies':'No','Contract':'Two year',
        'PaperlessBilling':'No','PaymentMethod':'Bank transfer (automatic)',
        'MonthlyCharges':65.20,'TotalCharges':3912.00,
    },
    {   # Medium-risk: 1-yr contract, fiber, no security add-ons
        'Gender':'Male','SeniorCitizen':1,'Partner':'No','Dependents':'No',
        'TenureMonths':18,'PhoneService':'Yes','MultipleLines':'No',
        'InternetService':'Fiber optic','OnlineSecurity':'No',
        'OnlineBackup':'Yes','DeviceProtection':'No','TechSupport':'No',
        'StreamingTV':'Yes','StreamingMovies':'No','Contract':'One year',
        'PaperlessBilling':'Yes','PaymentMethod':'Credit card (automatic)',
        'MonthlyCharges':78.50,'TotalCharges':1413.00,
    },
])

def predict(df_input: pd.DataFrame, threshold: float = THRESHOLD) -> pd.DataFrame:
    """Score customers and return risk tiers with recommended actions."""
    X_t   = preprocessor.transform(df_input)
    probs = model.predict_proba(X_t)[:, 1]
    preds = (probs >= threshold).astype(int)

    def risk_tier(p):
        if p >= 0.70: return '🔴 CRITICAL'
        if p >= 0.45: return '🟠 HIGH'
        if p >= 0.25: return '🟡 MEDIUM'
        return '🟢  LOW'

    def action(p):
        if p >= 0.70:
            return 'Immediate outreach — offer retention discount + account review'
        if p >= 0.45:
            return 'Proactive call — upsell TechSupport / OnlineSecurity bundle'
        if p >= 0.25:
            return 'Targeted email — loyalty reward or contract upgrade offer'
        return 'Standard engagement — monitor quarterly'

    return pd.DataFrame({
        'ChurnProbability': probs.round(4),
        'ChurnPredicted':   preds,
        'RiskTier':         [risk_tier(p) for p in probs],
        'RecommendedAction':[action(p) for p in probs],
    })

# ── Main ─────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Churn prediction inference')
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to CSV of new customers (same schema)')
    parser.add_argument('--threshold', type=float, default=THRESHOLD)
    args = parser.parse_args()

    if args.csv:
        df_input = pd.read_csv(args.csv)
        if 'CustomerID' in df_input.columns:
            ids = df_input['CustomerID']
            df_input = df_input.drop('CustomerID', axis=1)
        else:
            ids = [f'CUST-{i+1:04d}' for i in range(len(df_input))]
    else:
        df_input = DEMO_CUSTOMERS
        ids = ['DEMO-001 (High Risk)', 'DEMO-002 (Low Risk)', 'DEMO-003 (Medium Risk)']

    results = predict(df_input, threshold=args.threshold)
    results.insert(0, 'CustomerID', ids)

    print("\n" + "═"*90)
    print("  CUSTOMER CHURN PREDICTION — INFERENCE RESULTS")
    print(f"  Model: Logistic Regression  |  Threshold: {args.threshold}  |  "
          f"{len(results)} customer(s)")
    print("═"*90)

    for _, row in results.iterrows():
        print(f"\n  Customer : {row['CustomerID']}")
        print(f"  Churn Prob: {row['ChurnProbability']:.1%}   {row['RiskTier']}")
        print(f"  Predicted : {'⚠ CHURN' if row['ChurnPredicted'] else '✓ RETAIN'}")
        print(f"  Action    : {row['RecommendedAction']}")
        print("  " + "─"*85)

    # Summary
    n_churn = results['ChurnPredicted'].sum()
    print(f"\n  Summary: {n_churn}/{len(results)} customers flagged for churn "
          f"({n_churn/len(results):.0%})")
    print("═"*90 + "\n")
