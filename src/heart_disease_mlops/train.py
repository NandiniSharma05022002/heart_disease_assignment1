"""Train, evaluate, track, and package the heart-disease classifiers."""
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from .pipeline import make_logistic_regression, make_random_forest
from .schema import FEATURES

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "heart_disease.csv"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "reports" / "figures"
MLFLOW_DB = ROOT / "mlflow.db"
SEED = 42
EXPERIMENT_NAME = "heart-disease-classification"


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA)
    return df[FEATURES], df["target"]


def _log_classifier_params(model) -> None:
    """Log primitive classifier parameters rather than the classifier object itself."""
    classifier = model.named_steps["classifier"]
    params = classifier.get_params()
    mlflow.log_params({f"classifier_{key}": value for key, value in params.items()})


def evaluate_model(name, model, param_grid, X_train, X_test, y_train, y_test):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = GridSearchCV(
        model,
        param_grid=param_grid,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "roc_auc": "roc_auc",
        },
        refit="roc_auc",
        cv=cv,
        n_jobs=1,
    )
    search.fit(X_train, y_train)
    best_index = search.best_index_
    model = search.best_estimator_
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    metrics = {
        "cv_accuracy_mean": float(search.cv_results_["mean_test_accuracy"][best_index]),
        "cv_precision_mean": float(search.cv_results_["mean_test_precision"][best_index]),
        "cv_recall_mean": float(search.cv_results_["mean_test_recall"][best_index]),
        "cv_roc_auc_mean": float(search.best_score_),
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_precision": float(precision_score(y_test, pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, prob)),
    }

    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params(
            {
                "model": name,
                "random_seed": SEED,
                "test_size": 0.2,
                "cv_folds": 5,
            }
        )
        mlflow.log_params({f"tuned_{key}": value for key, value in search.best_params_.items()})
        _log_classifier_params(model)
        mlflow.log_metrics(metrics)

        FIGURES.mkdir(parents=True, exist_ok=True)

        cm = confusion_matrix(y_test, pred)
        ConfusionMatrixDisplay(confusion_matrix=cm).plot()
        plt.title(f"{name} — Confusion Matrix")
        cm_path = FIGURES / f"{name.lower().replace(' ', '_')}_confusion_matrix.png"
        plt.savefig(cm_path, dpi=160, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(str(cm_path), artifact_path="plots")

        RocCurveDisplay.from_predictions(y_test, prob)
        plt.title(f"{name} — ROC Curve")
        roc_path = FIGURES / f"{name.lower().replace(' ', '_')}_roc_curve.png"
        plt.savefig(roc_path, dpi=160, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(str(roc_path), artifact_path="plots")

        # MLflow's current sklearn flavor may use skops for model serialization.
        # These are the only additional types required by the models in this project.
        trusted_types = ["numpy.dtype"]
        if name == "Random Forest":
            trusted_types.append("sklearn.tree._tree.Tree")

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            skops_trusted_types=trusted_types,
        )

        return model, metrics, run.info.run_id, search.best_params_


def save_standalone_model(model, path: Path) -> None:
    """Save the selected sklearn Pipeline as a standalone pickle artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError(
            "Processed data missing. Run scripts/download_data.py and scripts/prepare_data.py."
        )

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=SEED,
    )

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.resolve()}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    candidates = [
        (
            "Logistic Regression",
            make_logistic_regression(),
            {"classifier__C": [0.1, 1.0, 10.0]},
        ),
        (
            "Random Forest",
            make_random_forest(),
            {"classifier__min_samples_leaf": [1, 2, 4]},
        ),
    ]

    results = []
    trained = []
    for name, model, param_grid in candidates:
        trained_model, metrics, run_id, best_params = evaluate_model(
            name, model, param_grid, X_train, X_test, y_train, y_test
        )
        results.append({"model": name, "run_id": run_id, "best_params": best_params, **metrics})
        trained.append((name, trained_model, metrics))

    results_df = pd.DataFrame(results).sort_values("cv_roc_auc_mean", ascending=False)
    best_name = str(results_df.iloc[0]["model"])
    best_model = next(model for name, model, _ in trained if name == best_name)

    best_model_path = ARTIFACTS / "model.pkl"
    save_standalone_model(best_model, best_model_path)

    best_metrics = results_df.iloc[0].to_dict()
    metadata = {
        "model": best_name,
        "features": FEATURES,
        "metrics": best_metrics,
        "random_seed": SEED,
        "threshold": 0.5,
        "selection_metric": "cv_roc_auc_mean",
        "serialization": "pickle",
    }
    (ARTIFACTS / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=float),
        encoding="utf-8",
    )
    results_df.to_csv(ARTIFACTS / "model_comparison.csv", index=False)

    print(results_df.to_string(index=False))
    print(f"Saved best model ({best_name}) to {best_model_path}")
    print(f"MLflow tracking database: {MLFLOW_DB}")


if __name__ == "__main__":
    main()
