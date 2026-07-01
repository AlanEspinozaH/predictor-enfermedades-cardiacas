"""Train a non-deployed PyCaret candidate using leakage-safe data partitions."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
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
        sha256_file,
    )
    from src.candidate_registry import (
        atomic_write_json,
        build_selection_report,
        feature_contract_evidence,
        publish_candidate_directory,
        select_candidate_evidence,
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
        sha256_file,
    )
    from candidate_registry import (
        atomic_write_json,
        build_selection_report,
        feature_contract_evidence,
        publish_candidate_directory,
        select_candidate_evidence,
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


def _software_versions() -> dict[str, str]:
    """Return reproducibility versions without serializing runtime objects."""

    versions = {"python": sys.version.split()[0]}
    for distribution in ("pycaret", "pandas", "numpy", "scikit-learn"):
        try:
            versions[distribution] = version(distribution)
        except PackageNotFoundError:
            versions[distribution] = "not_installed"
    return versions


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

    candidate_metrics: list[dict[str, object]] = []
    for index, model in enumerate(top_models):
        validation_predictions = predict_model(model, verbose=False)
        label_column = _prediction_label_column(validation_predictions)
        truth = validation_predictions[TARGET_COLUMN]
        predicted = validation_predictions[label_column]
        precision = float(precision_score(truth, predicted, zero_division=0))
        recall = float(recall_score(truth, predicted, zero_division=0))
        candidate_metrics.append(
            {
                "candidate_id": f"development-candidate-{index + 1}",
                "algorithm": type(model).__name__,
                "development_metrics": {
                    "precision": precision,
                    "recall": recall,
                },
                "status": "considered",
            }
        )
        print(
            f"Model {type(model).__name__}: recall={recall:.4f}, "
            f"precision={precision:.4f}"
        )
    (
        selected_candidate_index,
        candidate_models_considered,
        model_selection_rule,
    ) = select_candidate_evidence(candidate_metrics, minimum_precision=0.40)
    selected_model = top_models[selected_candidate_index]
    if not any(
        float(candidate["development_metrics"]["precision"]) >= 0.40
        for candidate in candidate_models_considered
    ):
        print(
            "No model met precision >= 0.40 on the internal validation set; "
            "using the measured highest-recall candidate across all candidates."
        )

    tuning_status = "accepted"
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
            tuning_status = "rejected_precision_below_0.40"
    except Exception as exc:
        print(f"Tuning failed ({exc}); using the selected model.")
        tuned_model = selected_model
        tuning_status = f"failed:{type(exc).__name__}"

    internal_predictions = predict_model(
        tuned_model,
        raw_score=True,
        verbose=False,
    )
    internal_label = _prediction_label_column(internal_predictions)
    internal_score = _positive_class_score(internal_predictions, internal_label)
    if internal_score is None:
        raise RuntimeError(
            "Threshold selection requires positive-class scores; no implicit "
            "default is permitted."
        )
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

    created_at = datetime.now(timezone.utc).isoformat()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidates_root = output_root / "candidates"
    candidate_dir = candidates_root / run_id
    staging_dir = candidates_root / f".{run_id}.tmp"
    candidates_root.mkdir(parents=True, exist_ok=True)
    if candidate_dir.exists() or staging_dir.exists():
        raise FileExistsError(f"Candidate run already exists: {run_id}")
    staging_dir.mkdir()

    try:
        try:
            interpret_model(final_model, plot="summary", save=True)
            shap_source = Path("Summary Plot.png")
            if shap_source.is_file():
                shutil.move(
                    str(shap_source),
                    staging_dir / "shap_summary_plot.png",
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
                "std": float(description["std"])
                if pd.notna(description["std"])
                else None,
                "min": float(description["min"])
                if pd.notna(description["min"])
                else None,
                "max": float(description["max"])
                if pd.notna(description["max"])
                else None,
                "missing_count": int(development[column].isna().sum()),
            }
        atomic_write_json(
            staging_dir / "training_reference_schema.json",
            reference_schema,
        )

        pipeline_stem = staging_dir / "pipeline"
        save_model(final_model, str(pipeline_stem))
        pipeline_path = pipeline_stem.with_suffix(".pkl")
        if not pipeline_path.is_file():
            raise FileNotFoundError(
                f"PyCaret did not save the pipeline: {pipeline_path}"
            )

        canonical_feature_config = PROJECT_ROOT / "models" / "model_config.json"
        if not canonical_feature_config.is_file():
            raise FileNotFoundError(
                f"Canonical feature configuration not found: {canonical_feature_config}"
            )
        feature_config = json.loads(
            canonical_feature_config.read_text(encoding="utf-8")
        )
        feature_config_path = atomic_write_json(
            staging_dir / "model_config.json",
            feature_config,
        )

        dataset_hash = sha256_file(selected_data_path)
        candidate_data_provenance = {
            "schema_version": 1,
            "dataset": {
                "identifier": selected_data_path.name,
                "sha256": dataset_hash,
                "row_count": int(len(source_data)),
                "columns": [str(column) for column in source_data.columns],
            },
            "source": str(
                provenance.get("source", "verified_training_provenance_sidecar")
            ),
            "cycles": provenance.get("cycles", provenance.get("source_cycles", [])),
            "independent_from_training": False,
            "target": {
                "column": TARGET_COLUMN,
                "source": "MCQ160E",
                "mapping": {"1": 1, "2": 0},
                "excluded": ["other", "missing"],
                "definition": (
                    "MCQ160E: 1 maps to 1, 2 maps to 0, and other or "
                    "missing responses are excluded."
                ),
            },
            "training_provenance": dict(provenance),
        }
        candidate_provenance_path = atomic_write_json(
            staging_dir / "data.provenance.json",
            candidate_data_provenance,
        )

        selected_evidence = candidate_models_considered[selected_candidate_index]
        selection_report = build_selection_report(
            run_id=run_id,
            created_at=created_at,
            strategy=strategy,
            random_seed=42,
            candidate_models_considered=candidate_models_considered,
            selection_split={
                "name": "internal_development_validation",
                "protected_test_excluded_from_selection": True,
                "data_split": dict(split.metadata),
            },
            primary_metric="recall",
            selection_rule=model_selection_rule,
            threshold_rule=str(internal_threshold_metrics["selection_rule"]),
            selected_model={
                "candidate_id": selected_evidence["candidate_id"],
                "algorithm": type(tuned_model).__name__,
                "tuning_status": tuning_status,
            },
            selected_threshold=best_threshold,
            development_metrics=internal_threshold_metrics,
            protected_test_consulted=evaluate_protected_test,
            feature_contract=feature_contract_evidence(MODEL_INPUT_FEATURES),
            software_versions=_software_versions(),
            data_provenance_reference={
                "path": candidate_provenance_path.name,
                "sha256": sha256_file(candidate_provenance_path),
            },
        )
        selection_report_path = atomic_write_json(
            staging_dir / "selection_report.json",
            selection_report,
        )

        candidate_manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "model_id": f"candidate-{run_id}",
            "status": "candidate_not_deployed",
            "strategy": strategy,
            "created_at": created_at,
            "model": {
                "path": pipeline_path.name,
                "format": "pycaret_pickle",
                "sha256": sha256_file(pipeline_path),
            },
            "feature_config": {
                "path": feature_config_path.name,
                "schema_version": FEATURE_SCHEMA_VERSION,
                "sha256": sha256_file(feature_config_path),
            },
            "decision_threshold": {
                "value": float(best_threshold),
                "source": "development_selection",
            },
            "data_provenance": {
                "path": candidate_provenance_path.name,
                "sha256": sha256_file(candidate_provenance_path),
            },
            "selection_evidence": {
                "path": selection_report_path.name,
                "sha256": sha256_file(selection_report_path),
            },
            "protected_test": {
                "consulted": evaluate_protected_test,
                "metrics": independent_metrics,
            },
            "deployment": {
                "deployed": False,
                "instruction": (
                    "Review candidate evidence before explicitly updating the "
                    "deployed manifest."
                ),
            },
        }
        atomic_write_json(
            staging_dir / "candidate_manifest.json",
            candidate_manifest,
        )
        publish_candidate_directory(staging_dir, candidate_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        Path("Summary Plot.png").unlink(missing_ok=True)
        raise

    pipeline_path = candidate_dir / "pipeline.pkl"
    candidate_manifest_path = candidate_dir / "candidate_manifest.json"
    print(f"Candidate pipeline saved to {pipeline_path}")
    print(f"Candidate manifest saved to {candidate_manifest_path}")
    print("The deployed model and deployed manifest were not modified.")
    return candidate_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a leakage-safe, non-deployed PyCaret candidate and publish a "
            "versioned manifest plus selection evidence. This is not clinical "
            "validation."
        ),
        epilog=(
            "Candidate outputs include hashed model, feature configuration, data "
            "provenance, and selection_report.json components. The deployed "
            "manifest is never changed automatically."
        ),
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA_PATH),
        help="Training cohort with a verified provenance sidecar",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Root where an atomic candidates/<run_id> directory is published",
    )
    parser.add_argument(
        "--strategy",
        choices=("SMOTE", "SCALE_POS_WEIGHT"),
        default="SMOTE",
        help="Development-only candidate training strategy",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Protected split fraction; never used for model or threshold selection",
    )
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
