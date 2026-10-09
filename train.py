import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

def train_and_save_model(data_path="diabetes.csv", output_path="model.pkl"):
    print(f"Loading dataset from '{data_path}'...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"'{data_path}' not found.")

    data = pd.read_csv(data_path)
    print(f"Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns.")

    # 1. Handle Missing (Zero) Values for biological features that cannot be 0
    features_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    median_values = {}

    print("Calculating medians for non-zero imputation...")
    for feature in features_with_zeros:
        median_val = float(data[data[feature] != 0][feature].median())
        median_values[feature] = median_val
        data[feature] = data[feature].replace(0, median_val)

    print(f"Stored medians: {median_values}")

    # 2. Features and Target
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Model Training
    print("Training XGBoost classifier...")
    model = XGBClassifier(
        objective='binary:logistic',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # 5. Evaluate
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # 6. Save Bundle
    artifacts = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'features_with_zeros': features_with_zeros,
        'median_values': median_values
    }

    with open(output_path, 'wb') as f:
        pickle.dump(artifacts, f)
    print(f"Model bundle saved to '{output_path}'.")

if __name__ == "__main__":
    train_and_save_model()
