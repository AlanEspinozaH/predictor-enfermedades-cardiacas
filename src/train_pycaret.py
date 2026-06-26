"""Train a non-deployed PyCaret candidate using leakage-safe data partitions."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pycaret.classification import (
    add_metric,
    compare_models,
    create_model,
    finalize_model,
    interpret_model,
    predict_model,
    pull,
    save_model,
    setup,
    tune_model,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from src.artifact_registry import (
        PROJECT_ROOT,
        ArtifactManifestError,
        repository_relative_path,
        sha256_file,
    )
    from src.data_pipeline import (
        load_training_provenance,
        prepare_modeling_cohort,
        provenance_path_for,
        read_tabular_data,
        split_development_and_test,
        validate_modeling_cohort,
    )
    from src.feature_contract import (
        FEATURE_SCHEMA_VERSION,
        MODEL_CATEGORICAL_FEATURES,
        MODEL_INPUT_FEATURES,
        MODEL_NUMERIC_FEATURES,
        TARGET_COLUMN,
    )
except ModuleNotFoundError:
    from artifact_registry import (
        PROJECT_ROOT,
        ArtifactManifestError,
        repository_relative_path,
        sha256_file,
    )
    from data_pipeline import (
        load_training_provenance,
        prepare_modeling_cohort,
        provenance_path_for,
        read_tabular_data,
        split_development_and_test,
        validate_modeling_cohort,
    )
    from feature_contract import (
        FEATURE_SCHEMA_VERSION,
        MODEL_CATEGORICAL_FEATURES,
        MODEL_INPUT_FEATURES,
        MODEL_NUMERIC_FEATURES,
        TARGET_COLUMN,
    )


DEFAULT_DATA_PATH = (
    PROJECT_ROOT / "data" / "02_processed" / "nhanes_heart_attack_modeling_raw.parquet"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models"


def _prediction_label_column(predictions: pd.DataFrame) -> str:
    if "prediction_label" in predictions.columns:
        return "prediction_label"
    candidates = [column for column in predictions.columns if "label" in column.lower()]
    if not candidates:
        raise ValueError("PyCaret did not produce a prediction label column.")
    return candidates[0]


def _positive_class_score(
    predictions: pd.DataFrame,
    label_column: str,
) -> pd.Series | None:
    exact_candidates = (
        "prediction_score_1",
        "prediction_score_True",
        "Score_1",
        "Score_True",
    )
    for column in exact_candidates:
        if column in predictions.columns:
            return pd.to_numeric(predictions[column], errors="coerce")

    for column in predictions.columns:
        normalized = column.lower()
        if "score" in normalized and normalized.endswith("_1"):
            return pd.to_numeric(predictions[column], errors="coerce")

    if "prediction_score" in predictions.columns:
        confidence = pd.to_numeric(predictions["prediction_score"], errors="coerce")
        label = pd.to_numeric(predictions[label_column], errors="coerce")
        return confidence.where(label == 1, 1.0 - confidence)
    return None


def _metrics_at_threshold(
    y_true: pd.Series,
    y_score: pd.Series,
    threshold: float,
) -> dict[str, object]:
    scores = pd.to_numeric(y_score, errors="coerce")
    if scores.isna().any():
        raise ValueError("Positive-class scores contain missing or non-numeric values.")

    truth = pd.to_numeric(y_true, errors="raise").astype(int)
    predicted = (scores >= threshold).astype(int)
    matrix = confusion_matrix(truth, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    metrics: dict[str, object] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "positive_count": int(truth.sum()),
        "row_count": int(len(truth)),
    }
    if truth.nunique() == 2:
        metrics.update(
            {
                "roc_auc": float(roc_auc_score(truth, scores)),
                "pr_auc": float(average_precision_score(truth, scores)),
                "brier_score": float(brier_score_loss(truth, scores)),
            }
        )
    return metrics


def _select_threshold(
    y_true: pd.Series,
    y_score: pd.Series,
    *,
    minimum_precision: float = 0.40,
) -> tuple[float, dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for threshold in np.arange(0.05, 0.951, 0.01):
        metrics = _metrics_at_threshold(y_true, y_score, float(threshold))
        candidates.append(metrics)

    precision_safe = [
        result
        for result in candidates
        if float(result["precision"]) >= minimum_precision
    ]
    if precision_safe:
        selected = max(
            precision_safe,
            key=lambda result: (float(result["recall"]), float(result["f1"])),
        )
        selected["selection_rule"] = (
            f"maximum recall with precision >= {minimum_precision:.2f}"
        )
    else:
        selected = max(candidates, key=lambda result: float(result["f1"]))
        selected["selection_rule"] = "maximum F1; precision constraint unmet"
    return float(selected["threshold"]), selected


def _data_reference(path: Path) -> str:
    try:
        return repository_relative_path(path)
    except ArtifactManifestError:
        return str(path)


def train_baseline(
    data_path: str | Path,
    output_dir: str | Path,
    *,
    strategy: str = "SMOTE",
    test_size: float = 0.20,
    evaluate_protected_test: bool = False,
) -> Path:
    """Train and save a frozen candidate without automatic deployment.

    The protected test is not scored by default. This permits comparing
    development-only strategies without repeatedly consulting the same test set.
    Set ``evaluate_protected_test=True`` only for the selected frozen strategy.
    """

    selected_data_path = Path(data_path).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    if not selected_data_path.is_file():
        raise FileNotFoundError(
            f"Training data not found: {selected_data_path}. Run "
            "data/02_processed/carga.py first."
        )

    provenance = load_training_provenance(selected_data_path)
    provenance_path = provenance_path_for(selected_data_path)
    print(f"Verified training provenance: {provenance_path}")
    print(f"Loading data from {selected_data_path}...")
    source_data = read_tabular_data(selected_data_path)
    cohort = prepare_modeling_cohort(source_data)
    validate_modeling_cohort(cohort)

    split = split_development_and_test(
        cohort,
        test_size=test_size,
        random_state=42,
    )
    development = split.development.loc[
        :, [*MODEL_INPUT_FEATURES, TARGET_COLUMN]
    ].copy()
    independent_test = split.independent_test.loc[
        :, [*MODEL_INPUT_FEATURES, TARGET_COLUMN]
    ].copy()

    print(f"Data split: {json.dumps(split.metadata, indent=2)}")
    print(
        f"Development target distribution:\n{development[TARGET_COLUMN].value_counts()}"
    )

    positive_count = int(development[TARGET_COLUMN].sum())
    negative_count = int(len(development) - positive_count)
    scale_weight = negative_count / positive_count if positive_count else 1.0
    print(f"Development scale_pos_weight: {scale_weight:.2f}")

    setup_args = {
        "data": development,
        "target": TARGET_COLUMN,
        "numeric_features": list(MODEL_NUMERIC_FEATURES),
        "categorical_features": list(MODEL_CATEGORICAL_FEATURES),
        "numeric_imputation": "median",
        "categorical_imputation": "mode",
        "normalize": True,
        "normalize_method": "minmax",
        "train_size": 0.80,
        "data_split_stratify": True,
        "fold_strategy": "stratifiedkfold",
        "fold": 5,
        "session_id": 42,
        "fix_imbalance": strategy == "SMOTE",
        "verbose": False,
    }
    print(
        "Setting up PyCaret. Imputers are fitted inside the development "
        "pipeline; the independent test set is not supplied to setup()."
    )
    setup(**setup_args)

    try:
        add_metric("Precision", "Precision", precision_score)
        add_metric("F1", "F1", f1_score)
    except ValueError:
        # Metrics may already be registered in an interactive session.
        pass

    include_models = None
    if strategy == "SCALE_POS_WEIGHT":
        include_models = ["xgboost", "lightgbm", "rf", "gbc", "et"]

    print("Comparing models on development folds...")
    top_models = compare_models(
        sort="Recall",
        n_select=5,
        include=include_models,
    )
    if not isinstance(top_models, list):
        top_models = [top_models]

    if strategy == "SCALE_POS_WEIGHT":
        print("Adding weighted XGBoost and LightGBM candidates...")
        xgb = create_model(
            "xgboost",
            scale_pos_weight=scale_weight,
            verbose=False,
        )
        lgbm = create_model(
            "lightgbm",
            scale_pos_weight=scale_weight,
            verbose=False,
        )
        top_models = [xgb, lgbm, *top_models]

    results = pull()
    available_columns = [
        column
        for column in ("Accuracy", "AUC", "Recall", "Precision", "F1")
        if column in results.columns
    ]
    if available_columns:
        print(results[available_columns])

    selected_model = None
    best_recall = -1.0
    for model in top_models:
        validation_predictions = predict_model(model, verbose=False)
        label_column = _prediction_label_column(validation_predictions)
        truth = validation_predictions[TARGET_COLUMN]
        predicted = validation_predictions[label_column]
        precision = precision_score(truth, predicted, zero_division=0)
        recall = recall_score(truth, predicted, zero_division=0)
        print(
            f"Model {type(model).__name__}: recall={recall:.4f}, "
            f"precision={precision:.4f}"
        )
        if precision >= 0.40 and recall > best_recall:
            selected_model = model
            best_recall = recall

    if selected_model is None:
        print(
            "No model met precision >= 0.40 on the internal validation set; "
            "using the highest-recall comparison model."
        )
        selected_model = top_models[0]

    try:
        tuned_model = tune_model(
            selected_model,
            optimize="Recall",
            n_iter=20,
            verbose=False,
        )
        tuned_validation = predict_model(tuned_model, verbose=False)
        tuned_label = _prediction_label_column(tuned_validation)
        tuned_precision = precision_score(
            tuned_validation[TARGET_COLUMN],
            tuned_validation[tuned_label],
            zero_division=0,
        )
        if tuned_precision < 0.40:
            print(
                f"Tuned precision {tuned_precision:.4f} is below 0.40; "
                "reverting to the selected untuned model."
            )
            tuned_model = selected_model
    except Exception as exc:
        print(f"Tuning failed ({exc}); using the selected model.")
        tuned_model = selected_model

    internal_predictions = predict_model(
        tuned_model,
        raw_score=True,
        verbose=False,
    )
    internal_label = _prediction_label_column(internal_predictions)
    internal_score = _positive_class_score(internal_predictions, internal_label)
    if internal_score is None:
        best_threshold = 0.50
        internal_threshold_metrics = {
            "threshold": best_threshold,
            "selection_rule": "default because positive-class score was unavailable",
        }
    else:
        best_threshold, internal_threshold_metrics = _select_threshold(
            internal_predictions[TARGET_COLUMN],
            internal_score,
        )
    print(
        "Threshold selected only from the internal development validation set: "
        f"{best_threshold:.2f}"
    )

    # Freeze the candidate on all development data before touching the protected
    # test set.  The evaluated object must be exactly the object that is saved;
    # otherwise the recorded metrics would describe a different fitted model.
    print("Finalizing the frozen candidate on development data only...")
    final_model = finalize_model(tuned_model)

    independent_metrics: dict[str, object] | None = None
    if evaluate_protected_test:
        print("Evaluating the frozen candidate once on the protected test set...")
        independent_predictions = predict_model(
            final_model,
            data=independent_test,
            raw_score=True,
            verbose=False,
        )
        independent_label = _prediction_label_column(independent_predictions)
        independent_score = _positive_class_score(
            independent_predictions,
            independent_label,
        )
        if independent_score is None:
            raise RuntimeError(
                "Protected-test evaluation requires positive-class probabilities, "
                "but PyCaret did not produce them."
            )
        independent_metrics = _metrics_at_threshold(
            independent_predictions[TARGET_COLUMN],
            independent_score,
            best_threshold,
        )
        print(json.dumps(independent_metrics, indent=2))
    else:
        print(
            "Protected test not evaluated. Compare candidate strategies using only "
            "development evidence, then rerun the selected strategy once with "
            "--evaluate-protected-test."
        )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate_dir = output_root / "candidates" / run_id
    candidate_dir.mkdir(parents=True, exist_ok=False)

    try:
        interpret_model(final_model, plot="summary", save=True)
        shap_source = Path("Summary Plot.png")
        if shap_source.is_file():
            shutil.move(
                str(shap_source),
                candidate_dir / "shap_summary_plot.png",
            )
    except Exception as exc:
        print(f"SHAP generation failed: {exc}")

    reference_schema: dict[str, dict[str, float | None]] = {}
    for column in MODEL_NUMERIC_FEATURES:
        description = pd.to_numeric(development[column], errors="coerce").describe()
        reference_schema[column] = {
            "mean": float(description["mean"])
            if pd.notna(description["mean"])
            else None,
            "std": float(description["std"]) if pd.notna(description["std"]) else None,
            "min": float(description["min"]) if pd.notna(description["min"]) else None,
            "max": float(description["max"]) if pd.notna(description["max"]) else None,
            "missing_count": int(development[column].isna().sum()),
        }
    reference_schema_path = candidate_dir / "training_reference_schema.json"
    reference_schema_path.write_text(
        json.dumps(reference_schema, indent=4),
        encoding="utf-8",
    )

    pipeline_stem = candidate_dir / "pipeline"
    save_model(final_model, str(pipeline_stem))
    pipeline_path = pipeline_stem.with_suffix(".pkl")

    feature_config_path = PROJECT_ROOT / "models" / "model_config.json"
    if not feature_config_path.is_file():
        raise FileNotFoundError(
            f"Canonical feature configuration not found: {feature_config_path}"
        )

    candidate_manifest = {
        "manifest_version": "1.1.0",
        "model_id": f"candidate-{run_id}",
        "status": "candidate_not_deployed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "path": repository_relative_path(pipeline_path),
            "format": "pycaret_pickle",
            "sha256": sha256_file(pipeline_path),
        },
        "feature_config": {
            "path": repository_relative_path(feature_config_path),
            "schema_version": FEATURE_SCHEMA_VERSION,
            "sha256": sha256_file(feature_config_path),
        },
        "decision_threshold": {
            "value": float(best_threshold),
            "source": "internal_development_validation",
            "selection_metrics": internal_threshold_metrics,
            "validation_status": (
                "evaluated_once_on_protected_internal_test"
                if evaluate_protected_test
                else "not_evaluated_on_protected_test"
            ),
        },
        "training": {
            "strategy": strategy,
            "data_path": _data_reference(selected_data_path),
            "data_sha256": sha256_file(selected_data_path),
            "provenance_path": _data_reference(provenance_path),
            "provenance_sha256": sha256_file(provenance_path),
            "provenance": dict(provenance),
            "development_rows": int(len(development)),
            "session_id": 42,
            "imputation": {
                "numeric": "median fitted within PyCaret development pipeline",
                "categorical": "mode fitted within PyCaret development pipeline",
                "pre_split_imputation": False,
            },
        },
        "data_split": dict(split.metadata),
        "protected_test_metrics": independent_metrics,
        "protected_test_evaluated": evaluate_protected_test,
        "deployment": {
            "deployed": False,
            "instruction": (
                "Review target provenance, protected-test metrics (when explicitly "
                "enabled), calibration, subgroup performance, and candidate artifacts "
                "before explicitly updating models/model_manifest.json."
            ),
        },
    }
    candidate_manifest_path = candidate_dir / "candidate_manifest.json"
    candidate_manifest_path.write_text(
        json.dumps(candidate_manifest, indent=4),
        encoding="utf-8",
    )

    print(f"Candidate pipeline saved to {pipeline_path}")
    print(f"Candidate manifest saved to {candidate_manifest_path}")
    print("The deployed model and deployed manifest were not modified.")
    return candidate_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a leakage-safe, non-deployed PyCaret candidate."
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--strategy",
        choices=("SMOTE", "SCALE_POS_WEIGHT"),
        default="SMOTE",
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument(
        "--evaluate-protected-test",
        action="store_true",
        help=(
            "Score the frozen candidate on the protected test. Use only for the "
            "selected strategy, not while comparing candidates."
        ),
    )
    args = parser.parse_args()
    train_baseline(
        args.data,
        args.output_dir,
        strategy=args.strategy,
        test_size=args.test_size,
        evaluate_protected_test=args.evaluate_protected_test,
    )


if __name__ == "__main__":
    main()
