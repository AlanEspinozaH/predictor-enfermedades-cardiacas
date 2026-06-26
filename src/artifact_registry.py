"""Canonical paths and integrity checks for deployed model artifacts.

The repository contains one explicitly deployed PyCaret pipeline. Runtime
components must resolve that artifact through ``models/model_manifest.json``
instead of selecting whichever ``.pkl`` happens to exist in ``models/``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from src.feature_contract import (
        MODEL_INPUT_FEATURES,
        TARGET_COLUMN,
        validate_feature_names,
    )
except ModuleNotFoundError:  # Support direct execution of scripts in src/.
    from feature_contract import (
        MODEL_INPUT_FEATURES,
        TARGET_COLUMN,
        validate_feature_names,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "models" / "model_manifest.json"


class ArtifactManifestError(RuntimeError):
    """Raised when the deployed artifact manifest is missing or inconsistent."""


@dataclass(frozen=True)
class DeployedArtifacts:
    """Resolved and validated artifacts used by runtime components."""

    manifest: Mapping[str, Any]
    manifest_path: Path
    model_id: str
    model_path: Path
    feature_config_path: Path
    training_data_path: Path | None
    decision_threshold: float
    model_sha256: str


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it fully in memory."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_repository_path(relative_path: str, *, must_exist: bool = True) -> Path:
    """Resolve a manifest path and prevent escaping the repository root."""

    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise ArtifactManifestError(
            f"Artifact paths must be repository-relative: {relative_path}"
        )

    resolved = (PROJECT_ROOT / candidate).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ArtifactManifestError(
            f"Artifact path escapes the repository root: {relative_path}"
        ) from exc

    if must_exist and not resolved.is_file():
        raise ArtifactManifestError(f"Artifact file not found: {resolved}")
    return resolved


def repository_relative_path(path: Path) -> str:
    """Return a portable POSIX path relative to the repository root."""

    resolved = path.resolve()
    try:
        relative = resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ArtifactManifestError(
            f"Path is outside the repository and cannot be registered: {path}"
        ) from exc
    return relative.as_posix()


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactManifestError(f"Manifest field '{field_name}' must be an object")
    return value


def _require_string(mapping: Mapping[str, Any], field_name: str) -> str:
    value = mapping.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ArtifactManifestError(
            f"Manifest field '{field_name}' must be a non-empty string"
        )
    return value


def _validate_hash(path: Path, expected_hash: str, label: str) -> None:
    actual_hash = sha256_file(path)
    if actual_hash.lower() != expected_hash.lower():
        raise ArtifactManifestError(
            f"{label} integrity check failed for {path}. "
            f"Expected {expected_hash}, found {actual_hash}."
        )


def load_deployed_artifacts(
    manifest_path: Path | None = None,
    *,
    verify_hashes: bool = True,
    require_training_data: bool = False,
) -> DeployedArtifacts:
    """Load the canonical deployed artifact set and verify its integrity."""

    selected_manifest = (manifest_path or DEFAULT_MANIFEST_PATH).resolve()
    if not selected_manifest.is_file():
        raise ArtifactManifestError(
            f"Deployed model manifest not found: {selected_manifest}"
        )

    try:
        manifest = json.loads(selected_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactManifestError(
            f"Could not read deployed model manifest: {selected_manifest}"
        ) from exc

    if not isinstance(manifest, Mapping):
        raise ArtifactManifestError("The deployed model manifest must be a JSON object")

    model_id = _require_string(manifest, "model_id")
    model_section = _require_mapping(manifest.get("model"), "model")
    config_section = _require_mapping(manifest.get("feature_config"), "feature_config")
    threshold_section = _require_mapping(
        manifest.get("decision_threshold"), "decision_threshold"
    )

    model_path = resolve_repository_path(_require_string(model_section, "path"))
    if model_path.suffix.lower() != ".pkl":
        raise ArtifactManifestError(
            f"The deployed PyCaret model must be a .pkl file: {model_path}"
        )
    model_hash = _require_string(model_section, "sha256")

    feature_config_path = resolve_repository_path(
        _require_string(config_section, "path")
    )
    config_hash = _require_string(config_section, "sha256")

    try:
        threshold = float(threshold_section["value"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactManifestError(
            "Manifest decision_threshold.value must be numeric"
        ) from exc
    if not 0.0 <= threshold <= 1.0:
        raise ArtifactManifestError(
            f"Decision threshold must be between 0 and 1, found {threshold}"
        )

    try:
        feature_config = json.loads(feature_config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactManifestError(
            f"Could not read feature configuration: {feature_config_path}"
        ) from exc

    declared_schema = config_section.get("schema_version")
    actual_schema = feature_config.get("schema_version")
    if declared_schema != actual_schema:
        raise ArtifactManifestError(
            "Feature schema version mismatch between model manifest and "
            f"configuration: {declared_schema!r} != {actual_schema!r}"
        )

    training_data_path: Path | None = None
    training_data_section = manifest.get("training_data")
    if training_data_section is not None:
        training_data = _require_mapping(training_data_section, "training_data")
        training_data_path = resolve_repository_path(
            _require_string(training_data, "path"),
            must_exist=require_training_data,
        )
        if require_training_data and not training_data_path.is_file():
            raise ArtifactManifestError(
                f"Training data file not found: {training_data_path}"
            )

    if require_training_data and training_data_path is None:
        raise ArtifactManifestError(
            "The deployed manifest does not declare a training_data artifact"
        )

    if verify_hashes:
        _validate_hash(model_path, model_hash, "Model")
        _validate_hash(feature_config_path, config_hash, "Feature configuration")
        if require_training_data and training_data_path is not None:
            training_data = _require_mapping(training_data_section, "training_data")
            data_hash = _require_string(training_data, "sha256")
            _validate_hash(training_data_path, data_hash, "Training data")

    return DeployedArtifacts(
        manifest=manifest,
        manifest_path=selected_manifest,
        model_id=model_id,
        model_path=model_path,
        feature_config_path=feature_config_path,
        training_data_path=training_data_path,
        decision_threshold=threshold,
        model_sha256=model_hash,
    )


def pycaret_model_stem(model_path: Path) -> str:
    """Return the file stem expected by ``pycaret.load_model``."""

    if model_path.suffix.lower() != ".pkl":
        raise ArtifactManifestError(
            f"Expected a .pkl PyCaret artifact, found: {model_path}"
        )
    return str(model_path.with_suffix(""))


def _string_feature_names(value: Any) -> tuple[str, ...] | None:
    """Normalize an sklearn/PyCaret ``feature_names_in_`` attribute."""

    if value is None:
        return None
    return tuple(str(name) for name in value)


def pipeline_metadata_features(pipeline: Any) -> tuple[str, ...] | None:
    """Return the external feature metadata exposed by a loaded pipeline.

    PyCaret may retain the training target in ``pipeline.feature_names_in_`` even
    though prediction correctly consumes only the model input columns. This
    function intentionally returns the metadata unchanged; normalization of that
    known PyCaret artifact is performed only by
    :func:`validate_deployed_pipeline_contract`.
    """

    direct = _string_feature_names(getattr(pipeline, "feature_names_in_", None))
    if direct is not None:
        return direct

    steps = getattr(pipeline, "steps", None)
    if steps:
        return _string_feature_names(getattr(steps[0][1], "feature_names_in_", None))
    return None


def deployed_estimator(pipeline: Any) -> Any:
    """Return the final estimator used for prediction by a PyCaret pipeline."""

    named_steps = getattr(pipeline, "named_steps", None)
    if isinstance(named_steps, Mapping) and "actual_estimator" in named_steps:
        return named_steps["actual_estimator"]

    steps = getattr(pipeline, "steps", None)
    if steps:
        return steps[-1][1]

    return pipeline


def validate_deployed_pipeline_contract(
    pipeline: Any,
    *,
    expected_features: Sequence[str] = MODEL_INPUT_FEATURES,
    target_column: str = TARGET_COLUMN,
) -> tuple[str, ...]:
    """Validate pipeline metadata without treating PyCaret target metadata as input.

    Only the canonical target may be removed from external pipeline metadata.
    Every other unexpected, missing, duplicated, or reordered column remains a
    contract error. Independently, the effective final estimator is inspected to
    ensure that the target was not used as a transformed predictor.

    Returns the normalized external feature order after the optional target
    removal. A pipeline that exposes no input metadata is rejected because the
    deployed artifact contract could not be verified.
    """

    metadata_features = pipeline_metadata_features(pipeline)
    if metadata_features is None:
        raise ValueError(
            "The loaded pipeline does not expose feature_names_in_; "
            "the deployed feature contract cannot be verified."
        )

    normalized_metadata = list(metadata_features)
    if target_column in normalized_metadata:
        normalized_metadata.remove(target_column)

    validate_feature_names(normalized_metadata, expected=expected_features)

    estimator = deployed_estimator(pipeline)
    estimator_features = _string_feature_names(
        getattr(estimator, "feature_names_in_", None)
    )
    if estimator_features is None:
        raise ValueError(
            "The deployed estimator does not expose feature_names_in_; "
            "target leakage cannot be checked."
        )
    if target_column in estimator_features:
        raise ValueError("Target leakage detected in deployed estimator")

    return tuple(normalized_metadata)


def load_validated_pycaret_pipeline(
    model_path: Path,
    *,
    expected_features: Sequence[str] = MODEL_INPUT_FEATURES,
    loader: Callable[[str], Any] | None = None,
) -> Any:
    """Load the real PyCaret artifact and validate its deployed feature contract.

    ``loader`` exists only to make unit tests independent from the optional
    PyCaret runtime. Production and integration tests use PyCaret's real
    ``classification.load_model`` function.
    """

    if loader is None:
        from pycaret.classification import load_model

        loader = load_model

    pipeline = loader(pycaret_model_stem(model_path))
    validate_deployed_pipeline_contract(
        pipeline,
        expected_features=expected_features,
        target_column=TARGET_COLUMN,
    )
    return pipeline
