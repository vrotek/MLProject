from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import joblib

from models.logistic_regression_parameter_grid_vo import LRParameterGrid
from util.dataset import Dataset
from models.logistic_regression_model import LogisticRegressionModel

dataset = Dataset("transactions.csv.zip")
dataset.extract_to(".")
df = dataset.load_data()

# === Prepare features and labels ===
X, y = dataset.split(df, "Class")

linRegModel = LogisticRegressionModel(X, y)

lrGridParams = LRParameterGrid(
    C_values=[0.01, 0.1, 1, 10, 100],
    penalties=['l1', 'l2'],
    solvers=['liblinear']
)
print("Tuning model...")
linRegModel.tune(lrGridParams.to_grid())

model, X_test_scaled, y_test, scaler = linRegModel.train()

# === Evaluate model ===
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred = model.predict(X_test_scaled)

print(f"AUC-ROC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# === Save model and scaler ===
linRegModel.dump_model("model_dumps")

print("✅ Model and scaler saved as 'logistic_model.joblib' and 'logistic_scaler.joblib'")