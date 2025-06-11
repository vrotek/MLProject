import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.Model import Model

class RandomForestModel(Model):

    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
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

        # === Train Random Forest ===
        self.model.fit(X_train_scaled, y_train)

        return self.model, X_test_scaled, y_test, self.scaler

    def dump_model(self, dump_to):
        os.makedirs(dump_to, exist_ok=True)

        model_path = os.path.join(dump_to, "rf_model.pkl")
        scaler_path = os.path.join(dump_to, "rf_scaler.pkl")

        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
