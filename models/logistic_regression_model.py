import os

import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.Model import Model


class LogisticRegressionModel(Model):

    def __init__(self, class_weight="balanced", solver="liblinear", random_state=42):
        self.model = LogisticRegression(
            class_weight=class_weight, solver=solver, random_state=random_state
        )
        self.scaler = StandardScaler()

    def train(self, features, label):
        # === Train-test split ===
        X_train, X_test, y_train, y_test = train_test_split(
            features, label, test_size=0.2, stratify=label, random_state=42
        )

        # === Feature scaling ===
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # === Train logistic regression ===
        self.model.fit(X_train_scaled, y_train)

        return self.model, X_test_scaled, y_test, self.scaler

    def dump_model(self,dump_to):

        os.makedirs(dump_to, exist_ok=True)

        model_path = os.path.join(dump_to, "logistic_model.pkl")
        scaler_path = os.path.join(dump_to, "logistic_scaler.pkl")

        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
