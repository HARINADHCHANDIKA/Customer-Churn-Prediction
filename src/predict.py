import joblib
import pandas as pd

model = joblib.load('../model/churn_model.pkl')

def predict(data):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return prediction[0]
