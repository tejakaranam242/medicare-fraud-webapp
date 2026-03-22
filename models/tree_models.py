from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def get_decision_tree():
    return DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=42)

def get_random_forest():
    return RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42)

def get_xgboost():
    return XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )

def get_hybrid_xgb():
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
