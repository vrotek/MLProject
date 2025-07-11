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
SCORING_METRICS = ['roc_auc', # Good for: Ranking frauds in general, robust to imbalance
                   'average_precision', # Good for: Catching rare positives (fraud), better for very imbalanced datasets
                   'f1' # Good for: When both false positives and false negatives matter
]

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

# use this function if you dont want to tune but just train the default linReg Model
def train_model():
    return linRegModel.train()

if (TUNING_RUN):
    tuning_results = tune_model()
    for result in tuning_results:
        print(result)
        result.export_csv(f"tuning_results/lr_results_{result.test_case}.csv")
else:
    train_model()


def dump_model(model: LogisticRegressionModel,suffix: str):
    model.dump_model("model_dumps",suffix)
    print("Model and scaler saved as 'logistic_model.joblib' and 'logistic_scaler.joblib'")


def evaluate_model(model: LogisticRegressionModel):
    y_pred_proba = model.selected_model.predict_proba(model.Xtest)[:, 1]
    y_pred = model.selected_model.predict(model.Xtest)

    print(f"AUC-ROC Score: {roc_auc_score(model.Ytest, y_pred_proba):.4f}")
    print("Classification Report:\n", classification_report(model.Ytest, y_pred))
    print("Confusion Matrix Keys: \n")
    print("[[TN,FP], Legitimate transactions correctly predicted, Legitimate transactions incorrectly flagged as fraud\n"
          "[FN,TP]] Fraud cases missed, Fraud cases correctly identified\n")


    print("Confusion Matrix :\n", confusion_matrix(model.Ytest, y_pred))


def run():
    if(TUNING_RUN):
        linRegModel.select_model("roc_auc")
        evaluate_model(linRegModel)
        dump_model(linRegModel, "_roc_auc")
        linRegModel.select_model("average_precision")
        evaluate_model(linRegModel)
        dump_model(linRegModel, "_average_precision")
        linRegModel.select_model("f1")
        evaluate_model(linRegModel)
        dump_model(linRegModel, "f1")
    else:
        evaluate_model(linRegModel)
        dump_model(linRegModel, "_default")



run()
