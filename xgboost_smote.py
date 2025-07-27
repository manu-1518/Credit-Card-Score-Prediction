import pandas as pd
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

def train_xgboost_with_smote(X_train_encoded, y_train, random_state=42):
    # Step 1: Resample using SMOTE
    sm = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = sm.fit_resample(X_train_encoded, y_train)

    print("Class counts after SMOTE:")
    print(pd.Series(y_train_resampled).value_counts())

    # Step 2: Train XGBoost on resampled data
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)
    model.fit(X_train_resampled, y_train_resampled)
    
    return model
