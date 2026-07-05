# Architecture

```mermaid
flowchart LR
    A[UCI Heart Disease Data] --> B[Data Validation and Cleaning]
    B --> C[EDA and Feature Engineering]
    C --> D[Scikit-learn Pipelines]
    D --> E[GridSearchCV and Cross-validation]
    E --> F[MLflow Tracking]
    E --> G[Packaged Joblib Model]
    G --> H[FastAPI Prediction Service]
    H --> I[Docker Image]
    I --> J[GitHub Actions CI]
    I --> K[Kubernetes Deployment]
    H --> L[Prometheus Metrics]
    L --> M[Grafana Dashboard]
```

## Design decisions

- The preprocessing transformer and estimator are saved as one pipeline to prevent training-serving skew.
- Logistic Regression provides a transparent baseline; Random Forest represents a nonlinear ensemble.
- ROC-AUC is used as the tuning score because both ranking quality and class discrimination matter.
- The API exposes health, prediction, metrics, and OpenAPI documentation endpoints.
- Kubernetes probes and resource limits provide basic production-readiness controls.
