import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.artifact_registry import sha256_file
from src.candidate_registry import CandidateManifestError
from src.data_pipeline import (
    DataProvenanceError,
    load_external_data_provenance,
    validate_external_provenance_frame,
)
from src.feature_contract import MODEL_INPUT_FEATURES, TARGET_COLUMN
from src.validate_external import resolve_validation_model, validate_external


class _FakeEstimator:
    def __init__(self):
        self.feature_names_in_ = np.asarray(MODEL_INPUT_FEATURES)
        self.classes_ = np.asarray([0, 1])


class _FakePipeline:
    def __init__(self):
        self.feature_names_in_ = np.asarray([*MODEL_INPUT_FEATURES, TARGET_COLUMN])
        self.named_steps = {"actual_estimator": _FakeEstimator()}
        self.classes_ = np.asarray([0, 1])

    def predict_proba(self, data):
        scores = np.linspace(0.1, 0.9, len(data))
        return np.column_stack([1.0 - scores, scores])


def _fake_loader(_stem: str):
    return _FakePipeline()


def _canonical_frame(*, with_target: bool) -> pd.DataFrame:
    rows = []
    for index in range(4):
        row = {feature: float(index + 1) for feature in MODEL_INPUT_FEATURES}
        row.update(
            {
                "Sex": 1 if index % 2 == 0 else 2,
                "Race": index + 1,
                "Education": index + 1,
                "Smoking": index % 2,
                "PhysicalActivity": (index + 1) % 2,
                "HealthInsurance": 1,
                "Alcohol": 0,
            }
        )
        if with_target:
            row[TARGET_COLUMN] = index % 2
        rows.append(row)
    return pd.DataFrame(rows)


def _write_valid_provenance(data_path: Path, frame: pd.DataFrame) -> Path:
    sidecar = data_path.with_suffix(".provenance.json")
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": {
                    "identifier": "synthetic-independent-cohort",
                    "sha256": sha256_file(data_path),
                    "row_count": len(frame),
                    "columns": list(frame.columns),
                },
                "source": "synthetic unit-test fixture",
                "period": "not applicable",
                "independent_from_training": True,
                "feature_contract": {"input_features": list(MODEL_INPUT_FEATURES)},
                "target": {
                    "column": TARGET_COLUMN,
                    "source": "MCQ160E",
                    "mapping": {"1": 1, "2": 0},
                    "excluded": ["other", "missing"],
                    "definition": (
                        "MCQ160E: 1 maps to 1, 2 maps to 0; other and missing "
                        "responses are excluded."
                    ),
                },
            }
        ),
        encoding="utf-8",
    )
    return sidecar


def test_deployed_mode_preserves_manifest_threshold_020():
    selected = resolve_validation_model()

    assert selected.mode == "deployed"
    assert selected.decision_threshold == {
        "value": pytest.approx(0.20),
        "source": "deployed_manifest",
        "overridden": False,
    }


def test_explicit_threshold_override_is_recorded():
    selected = resolve_validation_model(threshold=0.35)

    assert selected.decision_threshold == {
        "value": pytest.approx(0.35),
        "source": "command_line_override",
        "manifest_value": pytest.approx(0.20),
        "overridden": True,
    }


@pytest.mark.parametrize(
    "threshold",
    [0, 1, -0.1, 1.1, True, "0.25", float("nan"), float("inf"), float("-inf")],
)
def test_invalid_deployed_threshold_override_is_rejected(threshold):
    with pytest.raises(CandidateManifestError, match="invalid_threshold"):
        resolve_validation_model(threshold=threshold)


def test_model_path_is_rejected_without_silent_050_fallback(tmp_path):
    with pytest.raises(
        CandidateManifestError,
        match="--candidate-manifest",
    ):
        resolve_validation_model(model_path=tmp_path / "candidate.pkl")


def test_candidate_manifest_and_model_path_are_rejected_as_ambiguous(tmp_path):
    with pytest.raises(CandidateManifestError, match="--candidate-manifest"):
        resolve_validation_model(
            candidate_manifest=tmp_path / "candidate_manifest.json",
            model_path=tmp_path / "candidate.pkl",
        )


def test_dataset_hash_mismatch_is_fatal(tmp_path):
    data_path = tmp_path / "cohort.csv"
    frame = _canonical_frame(with_target=False)
    frame.to_csv(data_path, index=False)
    provenance = _write_valid_provenance(data_path, frame)
    metadata = json.loads(provenance.read_text(encoding="utf-8"))
    metadata["dataset"]["sha256"] = "0" * 64
    provenance.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(DataProvenanceError, match="hash_mismatch"):
        load_external_data_provenance(data_path)


def test_dataset_without_provenance_is_explicitly_unverified(tmp_path):
    data_path = tmp_path / "cohort.csv"
    _canonical_frame(with_target=False).to_csv(data_path, index=False)

    provenance = load_external_data_provenance(data_path)

    assert provenance.provenance_status == "unverified"
    assert provenance.validation_scope == "external_unverified"
    assert provenance.metadata is None


def test_valid_mcq160e_provenance_is_verified(tmp_path):
    data_path = tmp_path / "cohort.csv"
    frame = _canonical_frame(with_target=True)
    frame.to_csv(data_path, index=False)
    _write_valid_provenance(data_path, frame)

    provenance = load_external_data_provenance(data_path)
    target_verified = validate_external_provenance_frame(
        provenance,
        frame,
        target_present=True,
    )

    assert target_verified is True
    assert provenance.validation_scope == "external_independent"


def test_target_metrics_without_provenance_have_structured_warning(tmp_path):
    data_path = tmp_path / "cohort.csv"
    output_path = tmp_path / "validation.json"
    predictions_path = tmp_path / "predictions.csv"
    frame = _canonical_frame(with_target=True)
    frame["SEQN"] = [101, 102, 103, 104]
    frame.to_csv(data_path, index=False)

    validate_external(
        data_path,
        output_path,
        predictions_path=predictions_path,
        pipeline_loader=_fake_loader,
    )
    result = json.loads(output_path.read_text(encoding="utf-8"))

    assert result["validation_scope"] == "external_unverified"
    assert result["provenance_status"] == "unverified"
    assert result["target_status"] == {
        "present": True,
        "column": TARGET_COLUMN,
        "definition_verified": False,
    }
    assert result["metrics"] is not None
    assert any("target provenance" in warning for warning in result["warnings"])
    assert list(pd.read_csv(predictions_path).columns) == [
        "row_id",
        "positive_class_score",
        "decision_threshold",
        "predicted_class",
    ]
    assert set(MODEL_INPUT_FEATURES).isdisjoint(pd.read_csv(predictions_path).columns)
    assert TARGET_COLUMN not in pd.read_csv(predictions_path).columns
    assert "SEQN" not in pd.read_csv(predictions_path).columns


def test_verified_independent_result_records_target_and_feature_evidence(tmp_path):
    data_path = tmp_path / "cohort.csv"
    output_path = tmp_path / "validation.json"
    predictions_path = tmp_path / "predictions.csv"
    frame = _canonical_frame(with_target=True)
    frame.to_csv(data_path, index=False)
    _write_valid_provenance(data_path, frame)

    validate_external(
        data_path,
        output_path,
        predictions_path=predictions_path,
        pipeline_loader=_fake_loader,
    )
    result = json.loads(output_path.read_text(encoding="utf-8"))

    assert result["validation_scope"] == "external_independent"
    assert result["provenance_status"] == "verified"
    assert result["target_status"]["definition_verified"] is True
    assert result["independence_evidence"] == {
        "declared": True,
        "cryptographically_linked": True,
        "independently_audited": False,
    }
    assert result["feature_contract"]["external_feature_count"] == 27
    assert result["feature_contract"]["transformed_feature_count"] == 27
    assert result["feature_contract"]["target_in_transformed_features"] is False
    assert result["predictions"]["path"] == predictions_path.name
    assert result["predictions"]["sha256"] == sha256_file(predictions_path)
    assert any("not independently audited" in warning for warning in result["warnings"])
