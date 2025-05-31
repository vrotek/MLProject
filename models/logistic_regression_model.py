from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class LogisticRegressionModel:

    def train(self, features, label):
        # === Train-test split ===
        X_train, X_test, y_train, y_test = train_test_split(
            features, label, test_size=0.2, stratify=label, random_state=42
        )

        # === Feature scaling ===
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # === Train logistic regression ===
        model = LogisticRegression(
            class_weight="balanced", solver="liblinear", random_state=42
        )
        model.fit(X_train_scaled, y_train)

        return model, X_test_scaled, y_test, scaler
