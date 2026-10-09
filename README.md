# Diabetes Risk Prediction System

A full-stack Flask machine learning web application that predicts a patient's risk of diabetes using an **XGBoost Classifier** trained on the Pima Indians Diabetes Dataset (`diabetes.csv`).

---

## 📁 Project Structure

```text
Prime Diagnostics/
├── venv/                   # Python Virtual Environment
├── templates/
│   └── index.html          # Frontend Tailwind CSS web interface
├── app.py                  # Flask web server & prediction API
├── train.py                # Model training, imputation, scaling & serialization
├── test_app.py             # Automated unit tests
├── requirements.txt        # Python package dependencies
├── diabetes.csv            # Patient dataset
├── model.pkl               # Serialized model, scaler, and imputation medians
└── README.md               # Setup & usage guide
```

---

## 🚀 Getting Started

### 1. Virtual Environment Activation

The virtual environment `venv` is already set up in this directory.

- **On Windows (PowerShell):**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
  *(If script execution is restricted, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

- **On Windows (Command Prompt):**
  ```cmd
  venv\Scripts\activate.bat
  ```

### 2. (Optional) Install Dependencies
If you ever need to re-install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Train / Retrain Model
To re-train the XGBoost model on `diabetes.csv`:
```bash
python train.py
```
This performs non-zero median imputation for biological features (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`), scales the features using `StandardScaler`, trains the XGBoost model, and saves the artifacts to `model.pkl`.

### 4. Run the Flask Web Application
```bash
python app.py
```
Open your browser and navigate to:
[http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🧪 Run Tests
```bash
python test_app.py
```

---

## 📡 API Reference

### `POST /predict`
- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  {
    "pregnancies": 2,
    "glucose": 130,
    "bloodPressure": 75,
    "skinThickness": 25,
    "insulin": 100,
    "bmi": 28.4,
    "diabetesPedigreeFunction": 0.45,
    "age": 42
  }
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "prediction": 0,
    "probability": 28.15,
    "risk_level": "Low"
  }
  ```
