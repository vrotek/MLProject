from util.dataset import Dataset
from models.random_forest_model import RandomForestModel

# === Load and prepare data ===
dataset = Dataset("transactions.csv.zip")
dataset.extract_to(".")
df = dataset.load_data()

X, y = dataset.split(df, target_column="Class")

# === Create and train model ===
rfModel = RandomForestModel(X, y)
model, X_test_scaled, y_test, scaler = rfModel.train()

# === Save model and scaler ===
rfModel.dump_model("model_dumps")

print("✅ Model and scaler saved as 'rf_model.pkl' and 'rf_scaler.pkl'")
