import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

from models.Model import Model

class RandomForestModel(Model):
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()

    def train(self, X, y, enable_tuning=True):
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        if enable_tuning==True:
            print("Tuning hyperparameters...")
            param_grid = {
                "n_estimators": [50, 100, 150],
                "max_depth": [None, 10, 20],
            }
            grid_search = GridSearchCV(
                estimator=self.model,
                param_grid=param_grid,
                cv=3,
                n_jobs=-1,
                verbose=2
            )
            grid_search.fit(X_train_scaled, y_train)
            self.model = grid_search.best_estimator_
            print("Best parameters found:", grid_search.best_params_)
        else:
            self.model.fit(X_train_scaled, y_train)

        return self.model, X_test_scaled, y_test, self.scaler

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def dump_model(self, model_path, scaler_path):
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
