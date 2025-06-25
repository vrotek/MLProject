import os

import pickle
from typing import Dict, List

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

from models.Model import Model
from models.gridsearchCV_tuning_result_vo import GridSearchCVTuningResult


class LogisticRegressionModel(Model):
    tuned_models: Dict[str, LogisticRegression]
    tuning_results: Dict[str, GridSearchCVTuningResult]

    def __init__(self, features, label):

        self.Ytest = None #Label for testing
        self.Ytrain = None #Label for training
        self.Xtest = None #features for testing
        self.Xtrain = None #features for training

        self.selected_model = self.build_base_model()
        self.scaler = self.set_scalar()

        self.split(features, label)

        self.tuned_models: Dict[str, LogisticRegression] = {}
        self.tuning_results: Dict[str, GridSearchCVTuningResult] = {}

    def split(self, features, label):
        X_train, X_test, self.Ytrain, self.Ytest = train_test_split(
            features, label, test_size=0.2, stratify=label, random_state=42
        )

        # === Feature scaling ===
        self.Xtrain = self.scaler.fit_transform(X_train)
        self.Xtest = self.scaler.transform(X_test)

    def train(self):
        self.selected_model.fit(self.Xtrain, self.Ytrain)
        return self.selected_model

    def tune(self, paramGrid: Dict, scoring_metrics: List[str]) -> List[GridSearchCVTuningResult]:
        """
        Tunes the logistic regression model using GridSearchCV for multiple scoring metrics.

        Args:
            paramGrid (Dict): Dictionary of hyperparameters to search over.
            scoring_metrics (List[str]): A list of scoring metric names (e.g. ['roc_auc', 'average_precision'])

        Returns:
            List[GridSearchCVTuningResult]: A list of tuning results for each scoring metric.
        """
        if not paramGrid:
            raise ValueError("Parameter grid cannot be empty.")
        if not scoring_metrics:
            raise ValueError("You must specify at least one scoring metric.")

        results = []

        for scoring in scoring_metrics:

            baseModel = self.build_base_model()

            gridSearch = GridSearchCV(
                estimator=baseModel,
                param_grid=paramGrid,
                cv=5,
                scoring=scoring,
                n_jobs=-1,
                verbose=2,
                return_train_score=True,
            )

            gridSearch.fit(self.Xtrain, self.Ytrain)


            best_model = gridSearch.best_estimator_
            self.tuned_models[scoring] = best_model

            self.tuning_results[scoring] = GridSearchCVTuningResult(
                test_case=scoring,
                best_score=gridSearch.best_score_,
                best_params=gridSearch.best_params_,
                cv_results=gridSearch.cv_results_
            )

            results.append(self.tuning_results[scoring])

        self.selected_model = self.tuned_models[scoring_metrics[0]]

        return results

    def build_base_model(self, class_weight='balanced', solver='liblinear', random_state=42):
        """
        factory method for building Sklean LinReg Model
        :param class_weight:
            If the data is imbalanced, consider setting class_weight='balanced’
            or defining custom weights to improve the minority class performance.
        :param solver:
            liblinear: Generally the best for this type of use case
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

    def select_model(self, key: str):
        """
        Sets the current model (`self.model`) to one of the tuned models
        identified by the given scoring key.

        Args:
            key (str): The scoring key used during tuning (e.g., 'roc_auc', 'f1')

        Raises:
            ValueError: If the key is not found in tuned_models
        """
        if key not in self.tuned_models:
            raise ValueError(
                f"No model found for scoring key: '{key}'. Available keys: {list(self.tuned_models.keys())}")

        self.selected_model = self.tuned_models[key]

    def dump_model(self, dump_to, suffix=""):

        os.makedirs(dump_to, exist_ok=True)

        model_path = os.path.join(dump_to, f"logistic_model{suffix}.pkl")
        scaler_path = os.path.join(dump_to, f"logistic_scaler{suffix}.pkl")

        with open(model_path, "wb") as f:
            pickle.dump(self.selected_model, f)
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
