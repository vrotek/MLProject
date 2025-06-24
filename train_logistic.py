from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import joblib

from models.gridsearchCV_tuning_result_vo import GridSearchCVTuningResult
from models.logistic_regression_parameter_grid_vo import LRParameterGrid
from util.dataset import Dataset
from models.logistic_regression_model import LogisticRegressionModel

# Fixed Grid params used for tuning
LR_GRID_PARAMS = LRParameterGrid(
    C_values=[0.01, 0.1, 1, 10, 100],
    penalties=['l1', 'l2'],
    solvers=['liblinear']
)

# scoring metric you want to use in the tuning run
SCORING_METRICS = ['roc_auc', 'average_precision', 'f1']

# Activate this flag if you want tune your model on this run
TUNING_RUN = True

tuning_results = []

dataset = Dataset("transactions.csv.zip")
dataset.extract_to(".")
df = dataset.load_data()

X, y = dataset.split(df, "Class")

linRegModel = LogisticRegressionModel(X, y)


def tune_model():
    return linRegModel.tune(LR_GRID_PARAMS.to_grid(),SCORING_METRICS)


if (TUNING_RUN):
    tuning_results = tune_model()
    for result in tuning_results:
        print(result)
        result.export_csv(f"tuning_results/lr_results_{result.test_case}.csv")


def train_model():
    return linRegModel.train()


model, X_test_scaled, y_test, scaler = train_model()


def dump_model():
    linRegModel.dump_model("model_dumps")
    print("Model and scaler saved as 'logistic_model.joblib' and 'logistic_scaler.joblib'")


def evaluate_model():
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)

    print(f"AUC-ROC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


def run():
    evaluate_model()
    dump_model()


run()
