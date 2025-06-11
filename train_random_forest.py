from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import joblib

from util.dataset import Dataset
from models.random_forest_model import RandomForestModel

# === Load and prepare data ===
dataset = Dataset("transactions.csv.zip")
dataset.extract_to(".")
df = dataset.load_data()

X, y = dataset.split(df, target_column="Class")

# === Train model ===
rfModel = RandomForestModel()
model, X_test_scaled, y_test, scaler = rfModel.train(X, y)

# === Evaluate ===
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred = model.predict(X_test_scaled)

print(f"AUC-ROC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# === Save model and scaler ===
rfModel.dump_model("model_dumps")

print("✅ Model and scaler saved as 'rf_model.pkl' and 'rf_scaler.pkl'")
