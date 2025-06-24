import os

import pickle
from typing import Dict

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

from models.Model import Model


class LogisticRegressionModel(Model):

    def __init__(self, features, label, class_weight="balanced", solver="liblinear", random_state=42):

        self.Ytest = None #Label for testing
        self.Ytrain = None #Label for training
        self.Xtest = None #features for testing
        self.Xtrain = None #features for training

        self.model = self.build_base_model(class_weight, solver, random_state);
        self.scaler = StandardScaler()

        self.split(features, label)

    def split(self, features, label):
        # === Train-test split ===
        X_train, X_test, self.Ytrain, self.Ytest = train_test_split(
            features, label, test_size=0.2, stratify=label, random_state=42
        )

        # === Feature scaling ===
        self.Xtrain = self.scaler.fit_transform(X_train)
        self.Xtest = self.scaler.transform(X_test)

    def train(self):
        # === Train logistic regression ===
        self.model.fit(self.Xtrain, self.Ytrain)

        return self.model, self.Xtest, self.Ytest, self.scaler

    def tune(self, paramGrid: Dict) -> LogisticRegression:
        """
           Tunes the logistic regression model using grid search.

           Args:
               paramGrid (Dict): Dictionary of hyperparameters to search over.

           Returns:
               LogisticRegression: The best estimator found during tuning.
        """
        baseModel = LogisticRegression(class_weight='balanced', max_iter=500, random_state=42)

        # Wrap in GridSearch
        gridSearch = GridSearchCV(
            estimator=baseModel,
            param_grid=paramGrid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=2,
            return_train_score=True,
        )

        # Fit on training data
        gridSearch.fit(self.Xtrain, self.Ytrain)

        # Store best model and params
        print(f"[Tuning] Best ROC AUC: {gridSearch.best_score_:.4f}")
        print(f"[Tuning] Best parameters: {gridSearch.best_params_}")
        self.model = gridSearch.best_estimator_
        return self.model

    def build_base_model(self,class_weight,solver,random_state):
        return LogisticRegression(
            class_weight=class_weight, solver=solver, random_state=random_state
        )

    def dump_model(self, dump_to):

        os.makedirs(dump_to, exist_ok=True)

        model_path = os.path.join(dump_to, "logistic_model.pkl")
        scaler_path = os.path.join(dump_to, "logistic_scaler.pkl")

        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
