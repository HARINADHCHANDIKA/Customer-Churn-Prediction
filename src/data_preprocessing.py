import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    df.drop('customerID', axis=1, inplace=True)

    # Convert target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Encode categorical
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    return X, y
