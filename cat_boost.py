from catboost import CatBoostClassifier

def train_catboost(X_train, y_train):
    model = CatBoostClassifier(verbose=False, early_stopping_rounds=50, eval_metric='F1')
    model.fit(X_train, y_train)
    return model
