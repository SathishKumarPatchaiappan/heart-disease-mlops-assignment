"""Train, tune, compare, track, and package heart-disease classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from src.data import FEATURE_COLUMNS, TARGET_COLUMN, clean_data, load_raw_data
from src.evaluate import classification_metrics, save_evaluation_plots
from src.features import build_pipeline


def load_config(path: str | Path = "configs/config.yaml") -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def model_candidates(config: dict, random_state: int, quick: bool = False):
    lr_grid = {
        "model__C": [1.0] if quick else config["models"]["logistic_regression"]["C"],
        "model__solver": ["liblinear"] if quick else config["models"]["logistic_regression"]["solver"],
        "model__class_weight": [None] if quick else config["models"]["logistic_regression"]["class_weight"],
    }
    rf_grid = {
        "model__n_estimators": [50] if quick else config["models"]["random_forest"]["n_estimators"],
        "model__max_depth": [5] if quick else config["models"]["random_forest"]["max_depth"],
        "model__min_samples_split": [2]
        if quick
        else config["models"]["random_forest"]["min_samples_split"],
        "model__class_weight": [None] if quick else config["models"]["random_forest"]["class_weight"],
    }
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, random_state=random_state),
            lr_grid,
        ),
        "random_forest": (
            RandomForestClassifier(random_state=random_state),
            rf_grid,
        ),
    }


def _log_to_mlflow(
    experiment_name: str,
    tracking_uri: str,
    run_name: str,
    search: GridSearchCV,
    metrics: dict,
    artifact_paths: list[Path],
) -> None:
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError as exc:
        raise RuntimeError("MLflow is not installed. Run pip install -r requirements.txt") from exc

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({key.replace("model__", ""): value for key, value in search.best_params_.items()})
        mlflow.log_metric("best_cv_roc_auc", float(search.best_score_))
        mlflow.log_metrics(metrics)
        for artifact_path in artifact_paths:
            mlflow.log_artifact(str(artifact_path), artifact_path="evaluation")
        mlflow.sklearn.log_model(search.best_estimator_, artifact_path="model")


def train(config_path: str = "configs/config.yaml", quick: bool = False, use_mlflow: bool = True):
    config = load_config(config_path)
    random_state = int(config["project"]["random_state"])
    data_cfg = config["data"]
    train_cfg = config["training"]

    df = clean_data(load_raw_data(data_cfg["raw_path"]))
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(data_cfg["test_size"]),
        stratify=y,
        random_state=random_state,
    )

    cv_folds = 3 if quick else int(train_cfg["cv_folds"])
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    results: list[dict] = []
    fitted: dict[str, GridSearchCV] = {}

    for name, (estimator, grid) in model_candidates(config, random_state, quick).items():
        search = GridSearchCV(
            estimator=build_pipeline(estimator),
            param_grid=grid,
            scoring=train_cfg["scoring"],
            cv=cv,
            n_jobs=int(train_cfg["n_jobs"]),
            refit=True,
        )
        search.fit(X_train, y_train)
        predictions = search.predict(X_test)
        probabilities = search.predict_proba(X_test)[:, 1]
        metrics = classification_metrics(y_test, predictions, probabilities)
        metrics.update(
            {
                "model": name,
                "best_cv_roc_auc": float(search.best_score_),
                "best_params": json.dumps(search.best_params_, default=str),
            }
        )
        plot_paths = save_evaluation_plots(
            search.best_estimator_, X_test, y_test, "reports/figures", name
        )
        if use_mlflow:
            _log_to_mlflow(
                train_cfg["experiment_name"],
                train_cfg["tracking_uri"],
                name,
                search,
                {k: v for k, v in metrics.items() if isinstance(v, float)},
                plot_paths,
            )
        fitted[name] = search
        results.append(metrics)
        print(f"{name}: ROC-AUC={metrics['roc_auc']:.3f}, F1={metrics['f1_score']:.3f}")

    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    results_path = Path(train_cfg["results_output_path"])
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)

    best_name = str(results_df.iloc[0]["model"])
    best_pipeline = fitted[best_name].best_estimator_
    model_path = Path(train_cfg["model_output_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_path)

    metadata = {
        "selected_model": best_name,
        "selection_metric": "test_roc_auc",
        "test_metrics": {
            k: float(results_df.iloc[0][k])
            for k in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        },
        "feature_columns": FEATURE_COLUMNS,
        "model_path": str(model_path),
        "random_state": random_state,
    }
    metadata_path = model_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Selected model: {best_name}")
    print(f"Saved pipeline to {model_path}")
    return best_pipeline, results_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train heart-disease classifiers")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--quick", action="store_true", help="Use a compact grid for CI smoke tests")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    args = parser.parse_args()
    train(args.config, quick=args.quick, use_mlflow=not args.no_mlflow)


if __name__ == "__main__":
    main()
