import os
import pickle
from typing import Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

from models.Model import Model


class RandomForestModel(Model):

    def __init__(self, features, label):
        self.Ytrain = None
        self.Ytest = None
        self.Xtrain = None
        self.Xtest = None

        self.model = self.build_base_model()
        self.scaler = StandardScaler()

        self.split(features, label)

    def split(self, features, label):
        self.Xtrain, self.Xtest, self.Ytrain, self.Ytest = train_test_split(
            features, label, test_size=0.2, stratify=label, random_state=42
        )
        self.Xtrain = self.scaler.fit_transform(self.Xtrain)
        self.Xtest = self.scaler.transform(self.Xtest)

    def build_base_model(self):
        return RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt"
        )

    def train(self):
        self.model.fit(self.Xtrain, self.Ytrain)

        y_pred_proba = self.model.predict_proba(self.Xtest)[:, 1]
        y_pred = self.model.predict(self.Xtest)

        print(f"AUC-ROC Score: {roc_auc_score(self.Ytest, y_pred_proba):.4f}")
        print("Classification Report:\n", classification_report(self.Ytest, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(self.Ytest, y_pred))

        return self.model, self.Xtest, self.Ytest, self.scaler

    def tune(self, paramGrid: Dict):
        gridSearch = GridSearchCV(
            estimator=self.model,
            param_grid=paramGrid,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=2,
            return_train_score=True
        )
        gridSearch.fit(self.Xtrain, self.Ytrain)
        print(f"[Tuning] Best ROC AUC: {gridSearch.best_score_:.4f}")
        print(f"[Tuning] Best parameters: {gridSearch.best_params_}")
        self.model = gridSearch.best_estimator_
        return self.model

    def dump_model(self, dump_to):
        os.makedirs(dump_to, exist_ok=True)

        model_path = os.path.join(dump_to, "rf_model.pkl")
        scaler_path = os.path.join(dump_to, "rf_scaler.pkl")


        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
