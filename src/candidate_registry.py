"""Versioned candidate manifests and atomic evidence writers.

Candidate artifacts are never deployed implicitly.  This module verifies every
component declared by a candidate manifest before a caller may load its pickle.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from src.artifact_registry import sha256_file
    from src.feature_contract import (
        MODEL_INPUT_FEATURES,
        TARGET_COLUMN,
        feature_names_from_config,
        validate_feature_names,
    )
except ModuleNotFoundError:  # Support direct execution of scripts in src/.
    from artifact_registry import sha256_file
    from feature_contract import (
        MODEL_INPUT_FEATURES,
        TARGET_COLUMN,
        feature_names_from_config,
        validate_feature_names,
    )


CANDIDATE_SCHEMA_VERSION = 1
SELECTION_REPORT_SCHEMA_VERSION = 1
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


class CandidateManifestError(RuntimeError):
    """Raised with a stable category when candidate evidence is invalid."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class CandidateArtifacts:
    """Resolved candidate components whose hashes and contracts were verified."""

    manifest: Mapping[str, Any]
    manifest_path: Path
    run_id: str
    model_id: str
    strategy: str
    model_path: Path
    model_sha256: str
    feature_config_path: Path
    feature_config_sha256: str
    feature_config: Mapping[str, Any]
    data_provenance_path: Path
    data_provenance_sha256: str
    data_provenance: Mapping[str, Any]
    selection_report_path: Path
    selection_report_sha256: str
    selection_report: Mapping[str, Any]
    decision_threshold: float
    threshold_source: str


def strict_threshold(value: Any) -> float:
    """Return a numeric decision threshold strictly inside the unit interval."""

    if type(value) not in (int, float):
        raise CandidateManifestError(
            "invalid_threshold", "decision threshold must be numeric"
        )
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise CandidateManifestError(
            "invalid_threshold", "decision threshold must satisfy 0 < value < 1"
        )
    return threshold


def names_sha256(names: Sequence[str]) -> str:
    """Hash an ordered feature-name sequence deterministically."""

    payload = json.dumps(
        [str(name) for name in names],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def feature_contract_evidence(
    names: Sequence[str] = MODEL_INPUT_FEATURES,
) -> dict[str, Any]:
    """Return serializable identity for an ordered external feature contract."""

    normalized = tuple(str(name) for name in names)
    return {
        "external_feature_count": len(normalized),
        "external_feature_names_sha256": names_sha256(normalized),
        "target_column": TARGET_COLUMN,
        "target_in_external_features": TARGET_COLUMN in normalized,
    }


def select_candidate_evidence(
    candidates: Sequence[Mapping[str, Any]],
    *,
    minimum_precision: float = 0.40,
) -> tuple[int, list[dict[str, Any]], str]:
    """Select by development recall with a stable, explicit fallback policy."""

    if not candidates:
        raise ValueError("At least one development candidate is required")
    if not 0.0 <= minimum_precision <= 1.0:
        raise ValueError("minimum_precision must be between 0 and 1")

    annotated: list[dict[str, Any]] = []
    precisions: list[float] = []
    recalls: list[float] = []
    for candidate in candidates:
        evidence = dict(candidate)
        metrics = evidence.get("development_metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError("Each candidate requires development_metrics")
        try:
            precision = float(metrics["precision"])
            recall = float(metrics["recall"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Candidate precision and recall must be numeric") from exc
        if not math.isfinite(precision) or not math.isfinite(recall):
            raise ValueError("Candidate precision and recall must be finite")
        if not 0.0 <= precision <= 1.0 or not 0.0 <= recall <= 1.0:
            raise ValueError("Candidate precision and recall must be in [0, 1]")
        evidence["development_metrics"] = dict(metrics)
        annotated.append(evidence)
        precisions.append(precision)
        recalls.append(recall)

    eligible = [
        index
        for index, precision in enumerate(precisions)
        if precision >= minimum_precision
    ]
    pool = eligible if eligible else list(range(len(annotated)))
    selected_index = max(pool, key=lambda index: recalls[index])
    if eligible:
        selection_rule = (
            "highest development recall among candidates with precision "
            f">= {minimum_precision:.2f}; ties keep reproducible input order"
        )
    else:
        selection_rule = (
            "highest development recall across all candidates because none met "
            f"precision >= {minimum_precision:.2f}; ties keep reproducible input order"
        )

    selected_recall = recalls[selected_index]
    for index, evidence in enumerate(annotated):
        if index == selected_index:
            evidence["status"] = "selected_for_tuning"
            evidence["selection_reason"] = selection_rule
            evidence.pop("rejection_reason", None)
            continue

        evidence["status"] = "rejected"
        evidence.pop("selection_reason", None)
        if eligible and precisions[index] < minimum_precision:
            reason = f"precision below required minimum {minimum_precision:.2f}"
        elif recalls[index] < selected_recall:
            reason = "lower development recall than the selected candidate"
        else:
            reason = (
                "tied on development recall; an earlier candidate in reproducible "
                "input order was selected"
            )
        evidence["rejection_reason"] = reason

    return selected_index, annotated, selection_rule


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateManifestError(
            "invalid_manifest", f"field '{field_name}' must be an object"
        )
    return value


def _require_string(mapping: Mapping[str, Any], field_name: str) -> str:
    value = mapping.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise CandidateManifestError(
            "invalid_manifest",
            f"field '{field_name}' must be a non-empty string",
        )
    return value


def _require_sha256(mapping: Mapping[str, Any], field_name: str) -> str:
    value = _require_string(mapping, field_name)
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise CandidateManifestError(
            "invalid_manifest", f"field '{field_name}' must be a SHA-256 digest"
        )
    return value.lower()


def _resolve_component(
    manifest_path: Path,
    section: Mapping[str, Any],
    section_name: str,
) -> Path:
    relative_text = _require_string(section, "path")
    relative = Path(relative_text)
    if relative.is_absolute():
        raise CandidateManifestError(
            "unsafe_path",
            f"candidate component '{section_name}' must use a relative path",
        )

    root = manifest_path.parent.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise CandidateManifestError(
            "unsafe_path",
            f"candidate component '{section_name}' escapes the manifest directory",
        ) from exc
    if not resolved.is_file():
        raise CandidateManifestError(
            "missing_component",
            f"candidate component '{section_name}' was not found: {resolved}",
        )
    return resolved


def _verify_component(path: Path, expected: str, section_name: str) -> None:
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise CandidateManifestError(
            "hash_mismatch",
            f"candidate component '{section_name}' expected {expected}, found {actual}",
        )


def _read_json_component(path: Path, section_name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateManifestError(
            "invalid_manifest", f"component '{section_name}' is not valid JSON"
        ) from exc
    return _require_mapping(value, section_name)


def load_candidate_artifacts(manifest_path: str | Path) -> CandidateArtifacts:
    """Resolve and verify a complete candidate without loading its pickle."""

    selected_manifest = Path(manifest_path).expanduser().resolve()
    if not selected_manifest.is_file():
        raise CandidateManifestError(
            "missing_component", f"candidate manifest not found: {selected_manifest}"
        )
    try:
        manifest_value = json.loads(selected_manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateManifestError(
            "invalid_manifest",
            f"candidate manifest is not valid JSON: {selected_manifest}",
        ) from exc
    manifest = _require_mapping(manifest_value, "candidate_manifest")

    schema_version = manifest.get("schema_version")
    if type(schema_version) is not int or schema_version != CANDIDATE_SCHEMA_VERSION:
        raise CandidateManifestError(
            "unsupported_schema",
            f"candidate schema_version must be {CANDIDATE_SCHEMA_VERSION}",
        )

    run_id = _require_string(manifest, "run_id")
    model_id = _require_string(manifest, "model_id")
    strategy = _require_string(manifest, "strategy")
    _require_string(manifest, "created_at")
    status = _require_string(manifest, "status")
    if status != "candidate_not_deployed":
        raise CandidateManifestError(
            "invalid_manifest", "candidate status must be 'candidate_not_deployed'"
        )

    sections: dict[str, Mapping[str, Any]] = {}
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for name in (
        "model",
        "feature_config",
        "data_provenance",
        "selection_evidence",
    ):
        section = _require_mapping(manifest.get(name), name)
        sections[name] = section
        hashes[name] = _require_sha256(section, "sha256")
        paths[name] = _resolve_component(selected_manifest, section, name)
        _verify_component(paths[name], hashes[name], name)

    if paths["model"].suffix.lower() != ".pkl":
        raise CandidateManifestError(
            "invalid_manifest", "candidate model component must be a .pkl file"
        )

    threshold_section = _require_mapping(
        manifest.get("decision_threshold"), "decision_threshold"
    )
    if "value" not in threshold_section:
        raise CandidateManifestError(
            "invalid_threshold", "candidate manifest has no decision threshold"
        )
    threshold = strict_threshold(threshold_section["value"])
    threshold_source = _require_string(threshold_section, "source")

    feature_config = _read_json_component(paths["feature_config"], "feature_config")
    declared_config_schema = sections["feature_config"].get("schema_version")
    actual_config_schema = feature_config.get("schema_version")
    if (
        not isinstance(declared_config_schema, str)
        or declared_config_schema != actual_config_schema
    ):
        raise CandidateManifestError(
            "feature_contract_mismatch",
            "candidate feature configuration schema does not match its manifest",
        )
    try:
        configured_features = feature_names_from_config(feature_config)
        validate_feature_names(configured_features)
    except (TypeError, ValueError) as exc:
        raise CandidateManifestError("feature_contract_mismatch", str(exc)) from exc
    if TARGET_COLUMN in configured_features:
        raise CandidateManifestError(
            "feature_contract_mismatch", "target appears in candidate input features"
        )
    if feature_config.get("target") != TARGET_COLUMN:
        raise CandidateManifestError(
            "feature_contract_mismatch",
            f"feature configuration target must be {TARGET_COLUMN}",
        )

    data_provenance = _read_json_component(paths["data_provenance"], "data_provenance")
    if data_provenance.get("schema_version") != 1:
        raise CandidateManifestError(
            "unsupported_schema", "candidate data provenance schema_version must be 1"
        )
    provenance_dataset = _require_mapping(
        data_provenance.get("dataset"), "data_provenance.dataset"
    )
    dataset_identity = provenance_dataset.get(
        "identifier", provenance_dataset.get("path")
    )
    if not isinstance(dataset_identity, str) or not dataset_identity.strip():
        raise CandidateManifestError(
            "invalid_manifest",
            "candidate data provenance requires a dataset identifier or path",
        )
    _require_sha256(provenance_dataset, "sha256")
    _require_string(data_provenance, "source")
    provenance_target = _require_mapping(
        data_provenance.get("target"), "data_provenance.target"
    )
    if (
        provenance_target.get("column") != TARGET_COLUMN
        or provenance_target.get("source") != "MCQ160E"
        or provenance_target.get("mapping") != {"1": 1, "2": 0}
    ):
        raise CandidateManifestError(
            "invalid_manifest",
            "candidate data provenance must declare the canonical MCQ160E target",
        )

    selection_report = _read_json_component(
        paths["selection_evidence"], "selection_evidence"
    )
    if selection_report.get("schema_version") != SELECTION_REPORT_SCHEMA_VERSION:
        raise CandidateManifestError(
            "unsupported_schema",
            "selection report schema_version is unsupported",
        )
    if selection_report.get("run_id") != run_id:
        raise CandidateManifestError(
            "invalid_manifest", "selection report run_id does not match the candidate"
        )
    required_selection_fields = {
        "created_at": str,
        "strategy": str,
        "random_seed": int,
        "candidate_models_considered": list,
        "selection_split": Mapping,
        "primary_metric": str,
        "selection_rule": str,
        "threshold_rule": str,
        "selected_model": Mapping,
        "development_metrics": Mapping,
        "protected_test_consulted": bool,
        "feature_contract": Mapping,
        "software_versions": Mapping,
        "data_provenance_reference": Mapping,
    }
    for field_name, expected_type in required_selection_fields.items():
        value = selection_report.get(field_name)
        valid_type = isinstance(value, expected_type)
        if field_name == "random_seed":
            valid_type = type(value) is int
        elif field_name == "protected_test_consulted":
            valid_type = type(value) is bool
        if not valid_type:
            raise CandidateManifestError(
                "invalid_manifest",
                f"selection report field '{field_name}' has an invalid type",
            )
    considered = selection_report["candidate_models_considered"]
    if not considered or not all(isinstance(item, Mapping) for item in considered):
        raise CandidateManifestError(
            "invalid_manifest",
            "selection report must contain every considered candidate",
        )
    for item in considered:
        for field_name in ("candidate_id", "algorithm", "status"):
            _require_string(item, field_name)
        _require_mapping(item.get("development_metrics"), "development_metrics")
        if not isinstance(
            item.get("selection_reason", item.get("rejection_reason")), str
        ):
            raise CandidateManifestError(
                "invalid_manifest",
                "each considered candidate requires a selection or rejection reason",
            )
    if selection_report["strategy"] != strategy:
        raise CandidateManifestError(
            "invalid_manifest", "selection report strategy does not match the candidate"
        )
    report_threshold = strict_threshold(selection_report.get("selected_threshold"))
    if report_threshold != threshold:
        raise CandidateManifestError(
            "invalid_manifest",
            "selection report threshold does not match the candidate manifest",
        )
    provenance_reference = selection_report["data_provenance_reference"]
    if (
        provenance_reference.get("path") != sections["data_provenance"].get("path")
        or str(provenance_reference.get("sha256", "")).lower()
        != hashes["data_provenance"]
    ):
        raise CandidateManifestError(
            "invalid_manifest",
            "selection report does not reference the declared data provenance",
        )

    return CandidateArtifacts(
        manifest=manifest,
        manifest_path=selected_manifest,
        run_id=run_id,
        model_id=model_id,
        strategy=strategy,
        model_path=paths["model"],
        model_sha256=hashes["model"],
        feature_config_path=paths["feature_config"],
        feature_config_sha256=hashes["feature_config"],
        feature_config=feature_config,
        data_provenance_path=paths["data_provenance"],
        data_provenance_sha256=hashes["data_provenance"],
        data_provenance=data_provenance,
        selection_report_path=paths["selection_evidence"],
        selection_report_sha256=hashes["selection_evidence"],
        selection_report=selection_report,
        decision_threshold=threshold,
        threshold_source=threshold_source,
    )


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    )
    path = Path(handle.name)
    handle.close()
    return path


def atomic_write_json(destination: str | Path, value: Mapping[str, Any]) -> Path:
    """Write and parse-check JSON before atomically replacing the destination."""

    selected = Path(destination).expanduser().resolve()
    temporary = _temporary_path(selected)
    try:
        temporary.write_text(
            json.dumps(dict(value), indent=4, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        parsed = json.loads(temporary.read_text(encoding="utf-8"))
        if not isinstance(parsed, Mapping):
            raise TypeError("JSON output must contain an object")
        os.replace(temporary, selected)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return selected


def atomic_write_csv(
    destination: str | Path,
    frame: pd.DataFrame,
    *,
    expected_columns: Sequence[str] | None = None,
) -> Path:
    """Write and parse-check CSV before atomically replacing the destination."""

    selected = Path(destination).expanduser().resolve()
    temporary = _temporary_path(selected)
    try:
        frame.to_csv(temporary, index=False)
        parsed = pd.read_csv(temporary)
        if expected_columns is not None and tuple(parsed.columns) != tuple(
            expected_columns
        ):
            raise ValueError("CSV output columns changed during serialization")
        os.replace(temporary, selected)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return selected


def publish_candidate_directory(
    staging_directory: str | Path,
    final_directory: str | Path,
) -> Path:
    """Publish a complete verified candidate and remove failed staging output."""

    staging = Path(staging_directory).expanduser().resolve()
    final = Path(final_directory).expanduser().resolve()
    try:
        load_candidate_artifacts(staging / "candidate_manifest.json")
        if final.exists():
            raise FileExistsError(f"Candidate destination already exists: {final}")
        staging.replace(final)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return final


def atomic_write_validation_outputs(
    result_destination: str | Path,
    predictions_destination: str | Path,
    predictions: pd.DataFrame,
    result_factory: Callable[[str, str], Mapping[str, Any]],
    *,
    expected_columns: Sequence[str],
) -> tuple[Path, Path, Mapping[str, Any]]:
    """Stage related CSV/JSON outputs and publish the JSON last.

    The result factory receives the final CSV name and the digest of its staged
    content.  Failures before publication leave both prior final files intact;
    a failure publishing the JSON rolls the CSV back to its prior state.
    """

    result_path = Path(result_destination).expanduser().resolve()
    predictions_path = Path(predictions_destination).expanduser().resolve()
    staged_result = _temporary_path(result_path)
    staged_predictions = _temporary_path(predictions_path)
    predictions_backup = _temporary_path(predictions_path)
    predictions_backup.unlink(missing_ok=True)
    previous_predictions_moved = False
    new_predictions_published = False

    try:
        atomic_write_csv(
            staged_predictions,
            predictions,
            expected_columns=expected_columns,
        )
        result = result_factory(
            predictions_path.name,
            sha256_file(staged_predictions),
        )
        atomic_write_json(staged_result, result)

        if predictions_path.exists():
            os.replace(predictions_path, predictions_backup)
            previous_predictions_moved = True
        os.replace(staged_predictions, predictions_path)
        new_predictions_published = True
        os.replace(staged_result, result_path)

        try:
            predictions_backup.unlink(missing_ok=True)
        except OSError:
            pass
        return result_path, predictions_path, result
    except Exception:
        if new_predictions_published:
            predictions_path.unlink(missing_ok=True)
        if previous_predictions_moved and predictions_backup.exists():
            os.replace(predictions_backup, predictions_path)
        raise
    finally:
        staged_result.unlink(missing_ok=True)
        staged_predictions.unlink(missing_ok=True)
        if not previous_predictions_moved:
            predictions_backup.unlink(missing_ok=True)


def build_selection_report(
    *,
    run_id: str,
    created_at: str,
    strategy: str,
    random_seed: int,
    candidate_models_considered: Sequence[Mapping[str, Any]],
    selection_split: Mapping[str, Any],
    primary_metric: str,
    selection_rule: str,
    threshold_rule: str,
    selected_model: Mapping[str, Any],
    selected_threshold: float,
    development_metrics: Mapping[str, Any],
    protected_test_consulted: bool = False,
    feature_contract: Mapping[str, Any],
    software_versions: Mapping[str, str],
    data_provenance_reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Build serializable model-selection evidence without model objects."""

    if not candidate_models_considered:
        raise ValueError("candidate_models_considered cannot be empty")
    return {
        "schema_version": SELECTION_REPORT_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": created_at,
        "strategy": strategy,
        "random_seed": int(random_seed),
        "candidate_models_considered": [
            dict(candidate) for candidate in candidate_models_considered
        ],
        "selection_split": dict(selection_split),
        "primary_metric": primary_metric,
        "selection_rule": selection_rule,
        "threshold_rule": threshold_rule,
        "selected_model": dict(selected_model),
        "selected_threshold": strict_threshold(selected_threshold),
        "development_metrics": dict(development_metrics),
        "protected_test_consulted": bool(protected_test_consulted),
        "feature_contract": dict(feature_contract),
        "software_versions": dict(software_versions),
        "data_provenance_reference": dict(data_provenance_reference),
    }
