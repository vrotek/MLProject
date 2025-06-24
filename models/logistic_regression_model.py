import os

import pickle
from typing import Dict

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

from models.Model import Model
from models.gridsearchCV_tuning_result_vo import GridSearchCVTuningResult


class LogisticRegressionModel(Model):

    def __init__(self, features, label):

        self.Ytest = None #Label for testing
        self.Ytrain = None #Label for training
        self.Xtest = None #features for testing
        self.Xtrain = None #features for training

        self.model = self.build_base_model()
        self.scaler = self.set_scalar()

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

    def tune(self, paramGrid: Dict) -> GridSearchCVTuningResult:
        """
           Tunes the logistic regression model using grid search.

           Args:
               paramGrid (Dict): Dictionary of hyperparameters to search over.

           Returns:
               LogisticRegression: The best estimator found during tuning.
        """
        baseModel = self.build_base_model()

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

        self.model = gridSearch.best_estimator_

        return GridSearchCVTuningResult(
        best_score=gridSearch.best_score_,
        best_params=gridSearch.best_params_,
        cv_results=gridSearch.cv_results_
    )

    def build_base_model(self, class_weight='balanced', solver='liblinear',random_state=42):
        """
        factory method for building Sklean LinReg Model
        :param class_weight:
            If the data is imbalanced, consider setting class_weight='balanced’
            or defining custom weights to improve the minority class performance.
        :param solver:

        :param random_state:

        :return: an instance of a preconfigured LinReg model from sklearn
        """
        return LogisticRegression(
            class_weight=class_weight, solver=solver, random_state=random_state
        )

    def set_scalar(self):
        """
         Logistic Regression is sensitive to feature magnitude,
         so apply standardization (e.g., StandardScaler) in a pipeline to help the optimizer converge efficiently.
        :return: an instance of the StandardScaler
        """
        return StandardScaler()

    def dump_model(self, dump_to):

        os.makedirs(dump_to, exist_ok=True)

        model_path = os.path.join(dump_to, "logistic_model.pkl")
        scaler_path = os.path.join(dump_to, "logistic_scaler.pkl")

        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
