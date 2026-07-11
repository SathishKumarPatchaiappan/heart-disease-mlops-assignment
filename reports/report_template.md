# End-to-End MLOps Pipeline for Heart Disease Risk Prediction

**Course:** AIMLCZG523 Machine Learning Operations  
**Assignment:** Assignment 01  
**Student:** SathishKumar P  
**BITS ID:** 2024AD05172  
**Repository:** https://github.com/SathishKumarPatchaiappan/heart-disease-mlops-assignment  

---

## 1. Executive Summary

This project implements an end-to-end MLOps pipeline for predicting heart disease risk using the UCI Heart Disease dataset. The objective was to design, develop, track, package, containerize, deploy, and monitor a machine learning model using modern MLOps practices.

The solution includes data acquisition, exploratory data analysis, preprocessing, model development, MLflow experiment tracking, automated testing, GitHub Actions CI/CD, FastAPI model serving, Docker containerization, Kubernetes deployment, and Prometheus-style monitoring.

Two classification models were developed and compared:

- Logistic Regression
- Random Forest

The final selected model was **Logistic Regression**, as it achieved the best ROC-AUC score.

| Model | ROC-AUC | F1-score |
|---|---:|---:|
| Logistic Regression | 0.966 | 0.881 |
| Random Forest | 0.948 | 0.881 |

The final model was packaged as a reusable Scikit-learn pipeline and deployed as a FastAPI-based prediction service. The API was successfully tested locally, inside Docker, and through Kubernetes using Docker Desktop Kubernetes with port forwarding.

---

## 2. Problem Statement and Dataset

The goal of this project is to build a machine learning classifier that predicts the presence or absence of heart disease based on patient clinical attributes. The dataset used is the **Heart Disease UCI Dataset**, specifically the Cleveland processed dataset.

The dataset contains patient health-related features such as:

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- Resting ECG results
- Maximum heart rate achieved
- Exercise-induced angina
- Oldpeak
- Slope
- Number of major vessels
- Thalassemia status

The original target variable contains values from `0` to `4`. For this project, the target was converted into a binary classification problem:

- `0` = No heart disease
- `1` = Heart disease present

This binary transformation makes the model suitable for a risk prediction API.

---

## 3. Data Acquisition, Cleaning, and EDA

The dataset was stored in the repository under:

```text
data/raw/processed.cleveland.data
```

A cleaned version was generated and saved as:

```text
data/processed/heart_disease_clean.csv
```

The dataset contains:

- 303 records
- 13 input features
- 1 target column

Missing-value analysis showed that the `ca` and `thal` columns contained missing values. These were handled through the preprocessing pipeline using appropriate imputation strategies.

The exploratory data analysis included:

- Missing-value visualization
- Class distribution plot
- Numeric feature histograms
- Correlation heatmap
- Feature relationship analysis
- Target-wise comparison of important features

Key EDA findings:

- The dataset is reasonably balanced between heart disease and non-heart disease cases.
- Features such as chest pain type, maximum heart rate, oldpeak, exercise-induced angina, and thalassemia showed useful relationships with the target.
- Some features had missing values, confirming the need for a reproducible preprocessing pipeline.
- Correlation analysis helped understand relationships between numeric variables and the target.

Relevant figures generated:

```text
reports/figures/class_distribution.png
reports/figures/missing_values.png
reports/figures/numeric_histograms.png
reports/figures/correlation_heatmap.png
reports/figures/thalach_by_target.png
```

---

## 4. Feature Engineering and Model Development

A reusable preprocessing pipeline was developed using Scikit-learn.

The pipeline includes:

- Median imputation for numerical features
- Most-frequent imputation for categorical features
- Standard scaling for numerical variables
- One-hot encoding for categorical variables
- Integrated model training through Scikit-learn Pipeline

The data was split into training and testing sets using a stratified split to preserve the target class distribution.

Two classification models were trained:

1. Logistic Regression
2. Random Forest Classifier

Hyperparameter tuning was performed using GridSearchCV. Cross-validation was used to estimate generalization performance during training.

The following metrics were used:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

---

## 5. Model Evaluation and Selection

The model comparison results are stored in:

```text
reports/model_results.csv
```

Final model comparison:

| Model | ROC-AUC | F1-score |
|---|---:|---:|
| Logistic Regression | 0.966 | 0.881 |
| Random Forest | 0.948 | 0.881 |

Although both models achieved the same F1-score, Logistic Regression achieved the higher ROC-AUC score. Therefore, Logistic Regression was selected as the final model.

The selected final model was saved as:

```text
models/heart_disease_pipeline.joblib
```

Model metadata was saved as:

```text
models/heart_disease_pipeline.metadata.json
```

Evaluation artifacts include:

```text
reports/figures/logistic_regression_confusion_matrix.png
reports/figures/logistic_regression_roc_curve.png
reports/figures/random_forest_confusion_matrix.png
reports/figures/random_forest_roc_curve.png
```

---

## 6. Experiment Tracking with MLflow

MLflow was integrated to track all model experiments. The experiment name used was:

```text
heart-disease-classification
```

For each model run, the following items were logged:

- Model name
- Hyperparameters
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- ROC curve
- Model artifact

MLflow was run locally using:

```powershell
python -m mlflow ui --backend-store-uri .\mlruns --port 5000
```

Screenshots captured:

```text
05_mlflow_experiment_runs.png
06_mlflow_run_metrics.png
07_mlflow_confusion_matrix.png
08_mlflow_roc_curve.png
09_mlflow_model_artifacts.png
10_mlflow_parameters.png
```

These screenshots demonstrate that both Logistic Regression and Random Forest experiments were tracked successfully with metrics and artifacts.

---

## 7. Reproducibility and Model Packaging

Reproducibility was ensured through:

- A clean `requirements.txt`
- Fixed random seeds
- Reusable Scikit-learn Pipeline
- Script-based training
- Saved model artifact
- Metadata file
- Repository-based project structure

The final model pipeline includes both preprocessing and classification logic. This ensures the same transformations are applied during both training and inference.

The main model file is:

```text
models/heart_disease_pipeline.joblib
```

The environment can be recreated using:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The model can be retrained using:

```powershell
python -m src.train
```

---

## 8. Automated Testing and CI/CD

Automated testing was implemented using Pytest.

The test suite includes:

- Data loading tests
- Data cleaning tests
- Model artifact loading test
- API health endpoint test
- API prediction endpoint test

Tests were run locally using:

```powershell
python -m pytest -v
```

Result:

```text
5 passed
```

Code quality was checked using Ruff:

```powershell
python -m ruff check .
```

Result:

```text
All checks passed!
```

GitHub Actions was configured to automate:

- Dependency installation
- Linting
- Unit testing
- Training validation
- Docker build validation

Screenshots captured:

```text
01_github_actions_passed.png
02_local_pytest_passed.png
03_ruff_checks_passed.png
```

The successful GitHub Actions run confirms that the CI/CD pipeline works correctly.

---

## 9. API and Docker Container

The model was served using FastAPI.

The API exposes the following endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Root endpoint |
| `/health` | GET | Health check |
| `/predict` | POST | Heart disease prediction |
| `/metrics` | GET | Prometheus-style metrics |

Example prediction request:

```json
{
  "age": 63,
  "sex": 1,
  "cp": 1,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 2,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 3,
  "ca": 0,
  "thal": 6
}
```

Example response:

```json
{
  "prediction": 0,
  "risk": "low",
  "confidence": 0.5446,
  "model_version": "1.0.0"
}
```

The Docker image was built using:

```powershell
docker build -t heart-disease-api:latest .
```

The container was run using:

```powershell
docker run -p 8001:8000 heart-disease-api:latest
```

Docker evidence screenshots:

```text
15_docker_build_success.png
16_docker_container_swagger.png
17_docker_prediction_success.png
```

These screenshots prove that the container builds successfully, runs locally, and serves predictions through the `/predict` endpoint.

---

## 10. Kubernetes Deployment

The Dockerized API was deployed using Docker Desktop Kubernetes.

The Kubernetes configuration files are stored in:

```text
deployment/
```

The deployment includes:

- ConfigMap
- Deployment
- Service

The Docker image was tagged for Kubernetes:

```powershell
docker tag heart-disease-api:latest heart-disease-api:1.0.0
```

The Kubernetes manifests were applied using:

```powershell
kubectl apply -f deployment/
```

The deployment was verified using:

```powershell
kubectl get pods
kubectl get deployments
kubectl get svc
```

The Kubernetes deployment successfully created:

- 2 running pods
- Deployment ready status: `2/2`
- NodePort service
- Port-forwarded API access

Since direct NodePort access was not reachable from localhost, port forwarding was used:

```powershell
kubectl port-forward service/heart-disease-api-service 8080:80
```

The API was then accessed at:

```text
http://127.0.0.1:8080/docs
```

Kubernetes screenshots captured:

```text
18_kubernetes_node_ready.png
19_kubernetes_apply_success.png
20_kubernetes_pods_services_ready.png
21_kubernetes_swagger_ui.png
22_kubernetes_prediction_success.png
23_kubernetes_port_forward.png
```

These screenshots confirm that the API was deployed and tested successfully in Kubernetes.

---

## 11. Monitoring and Logging

Monitoring was implemented using Prometheus-compatible metrics exposed through the `/metrics` endpoint.

The metrics include:

- API request count
- Endpoint-level request count
- HTTP status code count
- API request latency
- Prediction count by class
- Python process metrics

Example metrics include:

```text
heart_api_requests_total
heart_api_request_duration_seconds
heart_api_predictions_total
```

The metrics endpoint showed successful monitoring for:

- `/health`
- `/docs`
- `/predict`
- `/metrics`

The API also logs incoming requests and prediction outcomes. Docker logs and Kubernetes logs can be used to verify runtime behaviour.

Monitoring screenshots:

```text
14_prometheus_metrics_endpoint.png
24_kubernetes_metrics_endpoint.png
```

This demonstrates basic production monitoring and logging suitable for a deployed machine learning API.

---

## 12. Architecture and Workflow

The architecture follows a complete MLOps lifecycle:

1. Raw dataset is stored in the repository.
2. Data cleaning and preprocessing scripts prepare the dataset.
3. EDA scripts generate visual insights.
4. Training script trains multiple models.
5. MLflow tracks experiments, metrics, and artifacts.
6. Best model is saved as a reusable pipeline.
7. FastAPI loads the model and exposes prediction endpoints.
8. Pytest validates data, model, and API behaviour.
9. GitHub Actions automates linting, testing, and validation.
10. Docker packages the API and model.
11. Kubernetes deploys the containerized API.
12. Prometheus-style metrics monitor request and prediction behaviour.

Architecture documentation is available at:

```text
docs/architecture.md
```

---

## 13. Limitations, Ethics, and Future Work

This project is intended for academic and educational purposes only. It should not be used as a real clinical diagnosis system without medical validation.

Limitations:

- The dataset is small with only 303 records.
- The dataset is historical and may not represent current populations.
- Clinical bias may exist in the original data.
- The model does not include advanced calibration.
- Data drift monitoring is basic.
- Security controls such as authentication and rate limiting are not implemented.

Ethical considerations:

- The model should support decision-making, not replace doctors.
- Incorrect predictions may have serious consequences in a real healthcare environment.
- Sensitive medical data must be protected with privacy and security controls.

Future improvements:

- Use a larger and more recent dataset.
- Add data drift and model drift monitoring.
- Add authentication to the API.
- Add model calibration.
- Add automated retraining.
- Deploy to managed cloud Kubernetes such as AKS, EKS, or GKE.
- Add Grafana dashboard screenshots.
- Add CI/CD deployment automation.

---

## 14. Setup and Reproduction Instructions

### Clone repository

```powershell
git clone https://github.com/SathishKumarPatchaiappan/heart-disease-mlops-assignment.git
cd heart-disease-mlops-assignment
```

### Create Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Run tests

```powershell
python -m pytest -v
```

### Run linting

```powershell
python -m ruff check .
```

### Train model

```powershell
python -m src.train
```

### Start MLflow UI

```powershell
python -m mlflow ui --backend-store-uri .\mlruns --port 5000
```

Open:

```text
http://127.0.0.1:5000
```

### Run FastAPI locally

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/docs
```

### Build Docker image

```powershell
docker build -t heart-disease-api:latest .
```

### Run Docker container

```powershell
docker run -p 8001:8000 heart-disease-api:latest
```

Open:

```text
http://127.0.0.1:8001/docs
```

### Deploy to Kubernetes

```powershell
docker tag heart-disease-api:latest heart-disease-api:1.0.0
kubectl apply -f deployment/
kubectl get pods
kubectl get deployments
kubectl get svc
```

### Port-forward Kubernetes service

```powershell
kubectl port-forward service/heart-disease-api-service 8080:80
```

Open:

```text
http://127.0.0.1:8080/docs
```

### View metrics

```text
http://127.0.0.1:8080/metrics
```

---

## 15. Conclusion

This project successfully demonstrates an end-to-end MLOps workflow for heart disease risk prediction. It includes dataset preparation, EDA, feature engineering, model development, MLflow experiment tracking, reproducible model packaging, automated testing, CI/CD, FastAPI serving, Docker containerization, Kubernetes deployment, and monitoring.

The final model was selected based on evaluation metrics, packaged as a reusable pipeline, and deployed as a monitored API. The GitHub repository includes all required code, configuration files, test cases, deployment manifests, screenshots, and documentation needed to reproduce the project.

Repository:

```text
https://github.com/SathishKumarPatchaiappan/heart-disease-mlops-assignment
```