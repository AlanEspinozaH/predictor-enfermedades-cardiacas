"""Run traceable technical evaluation for a deployed model or candidate.

This command produces structured evaluation evidence and tabular scores.  It is
not clinical validation and its positive-class score is not clinical risk.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from src.adapters import PyCaretAdapter
    from src.artifact_registry import (
        ArtifactManifestError,
        deployed_estimator,
        load_deployed_artifacts,
        load_validated_pycaret_pipeline,
        sha256_file,
        validate_deployed_pipeline_contract,
    )
    from src.candidate_registry import (
        CandidateManifestError,
        atomic_write_validation_outputs,
        feature_contract_evidence,
        load_candidate_artifacts,
        names_sha256,
        strict_threshold,
    )
    from src.data_pipeline import (
        DataProvenanceError,
        load_external_data_provenance,
        normalize_feature_columns,
        read_tabular_data,
        validate_external_provenance_frame,
    )
    from src.evaluation import binary_classification_metrics
    from src.feature_contract import MODEL_INPUT_FEATURES, TARGET_COLUMN
except ModuleNotFoundError:  # Support direct execution of scripts in src/.
    from adapters import PyCaretAdapter
    from artifact_registry import (
        ArtifactManifestError,
        deployed_estimator,
        load_deployed_artifacts,
        load_validated_pycaret_pipeline,
        sha256_file,
        validate_deployed_pipeline_contract,
    )
    from candidate_registry import (
        CandidateManifestError,
        atomic_write_validation_outputs,
        feature_contract_evidence,
        load_candidate_artifacts,
        names_sha256,
        strict_threshold,
    )
    from data_pipeline import (
        DataProvenanceError,
        load_external_data_provenance,
        normalize_feature_columns,
        read_tabular_data,
        validate_external_provenance_frame,
    )
    from evaluation import binary_classification_metrics
    from feature_contract import MODEL_INPUT_FEATURES, TARGET_COLUMN


@dataclass(frozen=True)
class ValidationModel:
    """Verified model identity and threshold policy for one evaluation."""

    mode: str
    model_id: str
    model_path: Path
    decision_threshold: Mapping[str, Any]
    artifact_integrity: Mapping[str, Any]


def _decision_threshold(
    manifest_value: float,
    manifest_source: str,
    override: float | None,
) -> dict[str, Any]:
    selected_manifest_value = strict_threshold(manifest_value)
    if override is None:
        return {
            "value": selected_manifest_value,
            "source": manifest_source,
            "overridden": False,
        }
    return {
        "value": strict_threshold(override),
        "source": "command_line_override",
        "manifest_value": selected_manifest_value,
        "overridden": True,
    }


def resolve_validation_model(
    *,
    candidate_manifest: str | Path | None = None,
    threshold: float | None = None,
    model_path: str | Path | None = None,
) -> ValidationModel:
    """Resolve either the canonical deployment or a complete candidate manifest."""

    if model_path is not None:
        raise CandidateManifestError(
            "invalid_manifest",
            "Los candidatos deben validarse mediante --candidate-manifest para "
            "conservar identidad, hashes, contrato y procedencia.",
        )

    if candidate_manifest is None:
        artifacts = load_deployed_artifacts(verify_hashes=True)
        config_section = artifacts.manifest["feature_config"]
        decision = _decision_threshold(
            artifacts.decision_threshold,
            "deployed_manifest",
            threshold,
        )
        return ValidationModel(
            mode="deployed",
            model_id=artifacts.model_id,
            model_path=artifacts.model_path,
            decision_threshold=decision,
            artifact_integrity={
                "status": "verified",
                "model": {
                    "sha256": artifacts.model_sha256,
                    "verified": True,
                },
                "feature_config": {
                    "sha256": config_section["sha256"],
                    "verified": True,
                },
            },
        )

    candidate = load_candidate_artifacts(candidate_manifest)
    decision = _decision_threshold(
        candidate.decision_threshold,
        "candidate_manifest",
        threshold,
    )
    if not decision["overridden"]:
        decision["declared_source"] = candidate.threshold_source
    return ValidationModel(
        mode="candidate",
        model_id=candidate.model_id,
        model_path=candidate.model_path,
        decision_threshold=decision,
        artifact_integrity={
            "status": "verified",
            "candidate_manifest_sha256": sha256_file(candidate.manifest_path),
            "model": {"sha256": candidate.model_sha256, "verified": True},
            "feature_config": {
                "sha256": candidate.feature_config_sha256,
                "verified": True,
            },
            "data_provenance": {
                "sha256": candidate.data_provenance_sha256,
                "candidate_provenance_sidecar_verified": True,
                "training_dataset_digest_recorded": True,
                "training_dataset_file_reverified": False,
            },
            "selection_evidence": {
                "sha256": candidate.selection_report_sha256,
                "verified": True,
            },
        },
    )


def _pipeline_contract(pipeline: Any) -> dict[str, Any]:
    external_features = validate_deployed_pipeline_contract(
        pipeline,
        expected_features=MODEL_INPUT_FEATURES,
        target_column=TARGET_COLUMN,
    )
    estimator = deployed_estimator(pipeline)
    transformed = getattr(estimator, "feature_names_in_", None)
    if transformed is None:
        raise ValueError(
            "feature_contract_mismatch: estimator does not expose transformed names"
        )
    transformed_names = tuple(str(name) for name in transformed)
    if TARGET_COLUMN in transformed_names:
        raise ValueError(
            "feature_contract_mismatch: target appears in transformed features"
        )
    evidence = feature_contract_evidence(external_features)
    evidence.update(
        {
            "status": "verified",
            "transformed_feature_count": len(transformed_names),
            "transformed_feature_names_sha256": names_sha256(transformed_names),
            "target_in_transformed_features": False,
        }
    )
    return evidence


def validate_external(
    data_path: str | Path,
    output_path: str | Path = "external_validation.json",
    *,
    predictions_path: str | Path = "external_predictions.csv",
    candidate_manifest: str | Path | None = None,
    threshold: float | None = None,
    provenance_path: str | Path | None = None,
    model_path: str | Path | None = None,
    pipeline_loader: Callable[[str], Any] | None = None,
) -> Path:
    """Write atomic JSON evidence and a minimal CSV of per-row scores."""

    selected_model = resolve_validation_model(
        candidate_manifest=candidate_manifest,
        threshold=threshold,
        model_path=model_path,
    )
    provenance = load_external_data_provenance(data_path, provenance_path)

    raw_data = read_tabular_data(data_path)
    data = normalize_feature_columns(raw_data)
    target_present = TARGET_COLUMN in data.columns
    target_definition_verified = validate_external_provenance_frame(
        provenance,
        raw_data,
        target_present=target_present,
    )

    missing = [
        feature for feature in MODEL_INPUT_FEATURES if feature not in data.columns
    ]
    if missing:
        raise ValueError(
            f"feature_contract_mismatch: external dataset is missing {missing}"
        )
    features = data.loc[:, MODEL_INPUT_FEATURES].copy()

    pipeline = load_validated_pycaret_pipeline(
        selected_model.model_path,
        expected_features=MODEL_INPUT_FEATURES,
        loader=pipeline_loader,
    )
    feature_contract = _pipeline_contract(pipeline)
    adapter = PyCaretAdapter(pipeline)
    scores = adapter.predict_proba(features)
    selected_threshold = float(selected_model.decision_threshold["value"])
    labels = (scores >= selected_threshold).astype(int)

    warnings: list[str] = []
    validation_scope = provenance.validation_scope
    provenance_status = provenance.provenance_status
    if provenance.metadata is None:
        warnings.append(
            "Dataset provenance was not verified; scope is external_unverified."
        )
    independence_declared = (
        provenance.metadata.get("independent_from_training")
        if provenance.metadata is not None
        else None
    )
    if independence_declared is True:
        warnings.append(
            "Dataset independence is declared in cryptographically linked "
            "provenance; it was not independently audited by this program."
        )

    metrics: dict[str, Any] | None = None
    if target_present:
        target = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
        valid = target.isin([0, 1])
        if not valid.all():
            raise ValueError(
                f"{TARGET_COLUMN} must contain only explicit binary labels 0 and 1"
            )
        metrics = binary_classification_metrics(
            target.astype(int),
            scores,
            selected_threshold,
        )
        if not target_definition_verified:
            validation_scope = "external_unverified"
            provenance_status = "unverified"
            warnings.append(
                "Metrics were computed, but target provenance was not verified."
            )
    else:
        warnings.append("Canonical target was not supplied; metrics were skipped.")

    predictions = pd.DataFrame(
        {
            "row_id": range(len(data)),
            "positive_class_score": scores,
            "decision_threshold": selected_threshold,
            "predicted_class": labels,
        }
    )
    dataset_section = (
        provenance.metadata.get("dataset", {})
        if provenance.metadata is not None
        else {}
    )
    result_without_predictions = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "validation_scope": validation_scope,
        "provenance_status": provenance_status,
        "independence_evidence": {
            "declared": independence_declared,
            "cryptographically_linked": provenance.metadata is not None,
            "independently_audited": False,
        },
        "model_identity": {
            "mode": selected_model.mode,
            "model_id": selected_model.model_id,
        },
        "artifact_integrity": dict(selected_model.artifact_integrity),
        "feature_contract": feature_contract,
        "decision_threshold": dict(selected_model.decision_threshold),
        "dataset_identity": {
            "file_name": Path(data_path).name,
            "sha256": provenance.dataset_sha256,
            "row_count": int(len(data)),
            "declared_identity": dataset_section.get(
                "identifier", dataset_section.get("path")
            ),
            "provenance_sidecar": (
                provenance.path.name if provenance.path is not None else None
            ),
        },
        "target_status": {
            "present": target_present,
            "column": TARGET_COLUMN if target_present else None,
            "definition_verified": target_definition_verified,
        },
        "metrics": metrics,
        "warnings": warnings,
    }

    def result_factory(
        predictions_name: str, predictions_sha256: str
    ) -> dict[str, Any]:
        result = dict(result_without_predictions)
        result["predictions"] = {
            "path": predictions_name,
            "sha256": predictions_sha256,
            "row_count": int(len(predictions)),
        }
        return result

    selected_output, selected_predictions, result = atomic_write_validation_outputs(
        output_path,
        predictions_path,
        predictions,
        result_factory,
        expected_columns=(
            "row_id",
            "positive_class_score",
            "decision_threshold",
            "predicted_class",
        ),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Validation metadata saved to: {selected_output}")
    print(f"Predictions saved to: {selected_predictions}")
    return selected_output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate non-clinical technical evaluation evidence for the deployed "
            "model or a cryptographically traced candidate."
        ),
        epilog=(
            "Without --candidate-manifest, the verified deployed manifest is used. "
            "Candidate mode requires a complete versioned manifest. Outputs are an "
            "atomic JSON evidence record and a minimal CSV of scores."
        ),
    )
    parser.add_argument(
        "--data-path", required=True, help="Local CSV or Parquet cohort"
    )
    parser.add_argument(
        "--output-path",
        default="external_validation.json",
        help="Primary structured JSON result (published after the CSV)",
    )
    parser.add_argument(
        "--predictions-path",
        default="external_predictions.csv",
        help="Minimal row_id, score, threshold, and predicted-class CSV",
    )
    parser.add_argument(
        "--candidate-manifest",
        help="Versioned candidate manifest; omit to validate the deployed model",
    )
    parser.add_argument(
        "--model-path",
        help="Deprecated and rejected; use --candidate-manifest",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Explicit 0 < threshold < 1 override recorded in the JSON result",
    )
    parser.add_argument(
        "--provenance-path",
        help=(
            "Optional versioned dataset-provenance JSON; otherwise a sidecar next "
            "to the cohort is used when present"
        ),
    )
    return parser


def main() -> None:
    parser = _build_parser()
    arguments = parser.parse_args()
    try:
        validate_external(
            data_path=arguments.data_path,
            output_path=arguments.output_path,
            predictions_path=arguments.predictions_path,
            candidate_manifest=arguments.candidate_manifest,
            threshold=arguments.threshold,
            provenance_path=arguments.provenance_path,
            model_path=arguments.model_path,
        )
    except (
        ArtifactManifestError,
        CandidateManifestError,
        DataProvenanceError,
        FileNotFoundError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        raise SystemExit(f"External evaluation failed: {exc}") from exc


if __name__ == "__main__":
    main()
