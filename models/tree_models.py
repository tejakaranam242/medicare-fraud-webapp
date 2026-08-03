from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def get_decision_tree():
    """Decision Tree optimized for financial fraud detection."""
    return DecisionTreeClassifier(
        max_depth=12,
        min_samples_leaf=3,
        min_samples_split=10,
        class_weight='balanced',
        criterion='gini',
        random_state=42
    )

def get_random_forest():
    """Random Forest with balanced class weights and improved hyperparams."""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        min_samples_split=10,
        max_features='sqrt',
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )

def get_xgboost():
    """
    XGBoost standalone classifier.
    scale_pos_weight handles class imbalance automatically.
    """
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.5,
        scale_pos_weight=3,    # Adjust for ~25% fraud rate in dataset
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1
    )

def get_hybrid_xgb():
    """
    Hybrid XGBoost — operates on fused deep feature embeddings.
    Tuned for high-dimensional fused feature space.
    """
    return XGBClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=3,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1
    )
