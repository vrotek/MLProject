from util.dataset import Dataset
from models.random_forest_model import RandomForestModel
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np

# === Load and prepare data ===
print("Loading data...")
dataset = Dataset("transactions.csv.zip")
dataset.extract_to("data")
df = dataset.load_data()
X, y = dataset.split(df, target_column="Class") 

# === Train model ===
print("Initializing model...")
rfModel = RandomForestModel()

print("Training...")
model, X_test_scaled, y_test, scaler = rfModel.train(X, y, enable_tuning=False) 

# === Evaluate ===
print("Evaluating...")
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred = model.predict(X_test_scaled)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("AUC-ROC Score:", roc_auc_score(y_test, y_pred_proba))

# === Save model and scaler ===
from joblib import dump
dump(model, "model_dumps/rf_model.pkl")
dump(scaler, "model_dumps/rf_scaler.pkl")

print("Model and scaler saved as 'rf_model.pkl' and 'rf_scaler.pkl'")