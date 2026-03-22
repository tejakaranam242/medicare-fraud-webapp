from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

class TabNetWrapper:
    def __init__(self, input_dim, output_dim=2):
        self.model = TabNetClassifier(
            n_d=8, n_a=8, n_steps=3,
            gamma=1.3, n_independent=2, n_shared=2,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size":10, "gamma":0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR, 
            mask_type='entmax', # "sparsemax"
            verbose=1
        )

    def train(self, X_train, y_train, X_valid, y_valid, max_epochs=20):
        self.model.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_valid, y_valid)],
            eval_name=['valid'],
            eval_metric=['accuracy', 'auc'],
            max_epochs=max_epochs, 
            patience=5,
            batch_size=256, 
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False
        )
        
    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, path):
        self.model.save_model(path)
        
    def load(self, path):
        self.model.load_model(path)
