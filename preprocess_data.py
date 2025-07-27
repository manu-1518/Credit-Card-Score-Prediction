from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

def preprocess(df, is_train=True):
    df = df.copy()

    # Separate target variable early if training
    if is_train:
        if "Target" not in df.columns:
            raise ValueError("Training data must contain a 'Target' column.")
        y = df["Target"]
        df = df.drop("Target", axis=1)
    else:
        y = None

    # Encode categorical features
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Scale features
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

    if is_train:
        return train_test_split(df_scaled, y, test_size=0.2, random_state=42)
    else:
        return df_scaled
