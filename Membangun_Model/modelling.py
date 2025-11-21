# modelling.py
import os
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from preprocessing import load_preprocessed_data

# --- 1. Load Data ---
X_train, X_test, y_train, y_test = load_preprocessed_data()

# --- 2. Setup MLflow Experiment ---
mlflow.set_tracking_uri("http://127.0.0.1:5000")  
mlflow.set_experiment("SMSML_XGBoost_Tuning")

# --- 3. Definisikan daftar hyperparameter yang akan diuji ---
learning_rate_list = [0.05, 0.1]
max_depth_list = [3, 5]
n_estimators_list = [100, 200]

best_acc = 0
best_params = {}

# --- 4. Loop tuning sederhana ---
for lr in learning_rate_list:
    for depth in max_depth_list:
        for n_est in n_estimators_list:
            run_name = f"XGB_lr{lr}_depth{depth}_n{n_est}"
            with mlflow.start_run(run_name=run_name):
                print(f"\n🚀 Training: {run_name}")

                # --- 4.1 Inisialisasi model XGBoost ---
                model = XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='logloss',
                    use_label_encoder=False,
                    learning_rate=lr,
                    max_depth=depth,
                    n_estimators=n_est,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                )

                # --- 4.2 Training model ---
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_train, y_train), (X_test, y_test)],
                    verbose=False
                )

                # --- 4.3 Evaluasi ---
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                print(f"Accuracy: {acc:.4f}")

                # Log model ke MLflow
                mlflow.sklearn.log_model(model, "model")

                # --- 4.5 Logging grafik logloss ---
                results = model.evals_result()
                epochs = len(results['validation_0']['logloss'])
                x_axis = range(0, epochs)

                plt.figure()
                plt.plot(x_axis, results['validation_0']['logloss'], label='Train')
                plt.plot(x_axis, results['validation_1']['logloss'], label='Test')
                plt.title('XGBoost Logloss per Epoch')
                plt.xlabel('Epoch')
                plt.ylabel('Logloss')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()

                plot_path = f"logloss_{run_name}.png"
                plt.savefig(plot_path)
                plt.close()

                mlflow.log_artifact(plot_path)  # simpan grafik ke MLflow
                os.remove(plot_path)  # hapus lokal agar rapi

                # --- 4.6 Simpan model terbaik ---
                if acc > best_acc:
                    best_acc = acc
                    best_params = {
                        "learning_rate": lr,
                        "max_depth": depth,
                        "n_estimators": n_est
                    }
                    os.makedirs("model", exist_ok=True)
                    model.save_model("model/xgb_best_model.json")

# --- 5. Hasil Akhir ---
print("\n🎯 Best Model Parameters:")
print(best_params)
print(f"✅ Best Accuracy: {best_acc:.4f}")
