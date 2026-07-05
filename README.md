# Heart Disease MLOps Assignment

End-to-end machine-learning operations project for heart-disease risk classification using the UCI Cleveland dataset.

## What this repository demonstrates

- Reproducible data loading, cleaning, and exploratory analysis
- Reusable Scikit-learn preprocessing pipeline
- Logistic Regression and Random Forest model comparison
- GridSearchCV with stratified cross-validation
- Accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, and ROC curves
- MLflow experiment tracking and model artifact logging
- Joblib model packaging
- FastAPI inference service with prediction confidence
- Prometheus-compatible metrics and structured logging
- Pytest unit and API tests
- GitHub Actions lint, test, train, artifact, and Docker-build stages
- Docker containerization
- Kubernetes deployment with ConfigMap, probes, replicas, and NodePort service
- Prometheus and Grafana local monitoring stack

## Repository structure

```text
.github/workflows/ci.yml      CI pipeline
app/                          FastAPI service
configs/config.yaml           Experiment and training settings
data/raw/                     UCI Cleveland source data
data/processed/               Generated cleaned CSV
deployment/                   Kubernetes manifests
docs/                         Architecture documentation
models/                       Generated model artifact
monitoring/                   Prometheus and Grafana configuration
notebooks/                    EDA and modelling notebook
reports/                      Results, figures, screenshots, report template
src/                          Data, features, EDA, evaluation, training code
tests/                        Automated tests
Dockerfile                    API image definition
requirements.txt              Reproducible Python environment
sample_request.json           Example API payload
```

## 1. Local setup

Python 3.11 is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Prepare data and create EDA figures

```bash
python -m src.data
python -m src.eda
```

The cleaned CSV is written to `data/processed/heart_disease_clean.csv`. EDA figures are written to `reports/figures/`.

## 3. Train models with MLflow

Open one terminal and start the tracking UI:

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Open another terminal and train:

```bash
python -m src.train
```

Then open `http://127.0.0.1:5000` and capture evidence of experiments, metrics, parameters, plots, and models.

A shorter CI smoke test is available:

```bash
python -m src.train --quick --no-mlflow
```

## Validated baseline results

Using a fixed stratified 80/20 split and five-fold GridSearchCV, the packaged baseline produced:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.885 | 0.839 | 0.929 | 0.881 | 0.966 |
| Random Forest | 0.885 | 0.839 | 0.929 | 0.881 | 0.948 |

Logistic Regression was selected because it achieved the stronger ROC-AUC on the held-out test split while retaining a simpler, more interpretable decision function. Re-run the full experiment locally and use your own MLflow and workflow screenshots in the report.

## 4. Run automated tests and linting

```bash
ruff check .
pytest
```

## 5. Run the FastAPI service

Ensure training has created `models/heart_disease_pipeline.joblib`, then run:

```bash
uvicorn app.main:app --reload
```

Open:

- Swagger UI: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`
- Metrics: `http://127.0.0.1:8000/metrics`

Test prediction:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  --data @sample_request.json
```

Example response:

```json
{
  "prediction": 0,
  "risk": "low",
  "confidence": 0.8343,
  "model_version": "1.0.0"
}
```

## 6. Docker

```bash
docker build -t heart-disease-api:1.0.0 .
docker run --rm -p 8000:8000 heart-disease-api:1.0.0
```

Test `/health`, `/docs`, `/predict`, and `/metrics` while the container is running.

## 7. Minikube deployment

```bash
minikube start
```

Use Minikube's Docker daemon, build the image, and apply manifests:

```bash
eval $(minikube docker-env)
docker build -t heart-disease-api:1.0.0 .
kubectl apply -f deployment/configmap.yaml
kubectl apply -f deployment/deployment.yaml
kubectl apply -f deployment/service.yaml
kubectl get pods
kubectl get services
minikube service heart-disease-api-service --url
```

For Windows PowerShell, run:

```powershell
minikube docker-env --shell powershell | Invoke-Expression
docker build -t heart-disease-api:1.0.0 .
```

## 8. Prometheus and Grafana

After a trained model exists:

```bash
cd monitoring
docker compose up --build
```

Open:

- API: `http://127.0.0.1:8000`
- Prometheus: `http://127.0.0.1:9090`
- Grafana: `http://127.0.0.1:3000` (default login `admin` / `admin`)

Useful Prometheus metrics:

- `heart_api_requests_total`
- `heart_api_request_duration_seconds`
- `heart_api_predictions_total`

## 9. Submission evidence

Use `reports/screenshots/README.md` as the evidence checklist and `reports/report_template.md` as the report outline. Record a short video showing data preparation, training and MLflow, tests and CI, API, Docker, Kubernetes, and monitoring.

## Responsible-use note

This project is an educational demonstration. It is not a medical device and must not be used to diagnose or treat patients.
