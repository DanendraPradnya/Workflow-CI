import os
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

def train_model_autolog():
  # MLflow Set Experiment name
  mlflow.set_experiment("Credit_Card_Fraud_Detection_Autolog")
  mlflow.sklearn.autolog()

  #Loading Data
  preprocessed_path ="credit_card_fraud_processed.csv"
  
  if not os.path.exists(preprocessed_path):
    print("Data processed tidak ditemukan! Jalankan preprocessing terlebih dahulu.")
    return

  df = pd.read_csv(preprocessed_path)
  X = df.drop(columns=['is_fraud'])
  y = df['is_fraud']

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

  # SMOTE
  print(f"Jumlah sampel sebelum SMOTE: {np.bincount(y_train)}")
  
  smote = SMOTE(random_state=42)
  X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
  
  print(f"Jumlah sampel setelah SMOTE: {np.bincount(y_train_res)}")

  # Mulai MLflow Run
  with mlflow.start_run(run_name="RF_Fraud_Autologging") as run:
    print(f"Running experiment: {run.info.run_name}")
        
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    
    model.fit(X_train_res, y_train_res)

    # Simpan Model Lokal untuk Artifact GitHub
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/best_model.pkl")

    # Simpan Run ID ke file
    with open("run_id.txt", "w") as f:
        f.write(mlflow.active_run().info.run_id)
    
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    print(classification_report(y_test, y_pred))
    mlflow.log_metric("accuracy", accuracy)

if __name__ == "__main__":
  train_model_autolog()