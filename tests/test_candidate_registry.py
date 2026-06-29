import json
from pathlib import Path

import pandas as pd
import pytest

from src.artifact_registry import sha256_file
from src.candidate_registry import (
    CandidateManifestError,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_validation_outputs,
    build_selection_report,
    feature_contract_evidence,
    load_candidate_artifacts,
    publish_candidate_directory,
    select_candidate_evidence,
)
from src.feature_contract import MODEL_INPUT_FEATURES, TARGET_COLUMN
from src.validate_external import resolve_validation_model


def _selection_report(
    run_id: str = "run-001",
    provenance_sha256: str = "0" * 64,
) -> dict:
    return build_selection_report(
        run_id=run_id,
        created_at="2026-01-01T00:00:00+00:00",
        strategy="SMOTE",
        random_seed=42,
        candidate_models_considered=[
            {
                "candidate_id": "candidate-1",
                "algorithm": "FakeClassifier",
                "development_metrics": {"recall": 0.5, "precision": 0.5},
                "status": "selected_for_tuning",
                "selection_reason": "synthetic unit-test selection",
            },
            {
                "candidate_id": "candidate-2",
                "algorithm": "OtherClassifier",
                "development_metrics": {"recall": 0.4, "precision": 0.5},
                "status": "rejected",
                "rejection_reason": "lower development recall",
            },
        ],
        selection_split={"name": "development_validation"},
        primary_metric="recall",
        selection_rule="synthetic unit-test selection",
        threshold_rule="synthetic development threshold",
        selected_model={"candidate_id": "candidate-1"},
        selected_threshold=0.25,
        development_metrics={"recall": 0.5},
        feature_contract=feature_contract_evidence(),
        software_versions={"python": "3.10"},
        data_provenance_reference={
            "path": "data.provenance.json",
            "sha256": provenance_sha256,
        },
    )


def _candidate_manifest(tmp_path: Path) -> tuple[Path, dict, dict[str, Path]]:
    model = tmp_path / "pipeline.pkl"
    model.write_bytes(b"synthetic model bytes; never unpickled")

    config = tmp_path / "model_config.json"
    atomic_write_json(
        config,
        {
            "schema_version": "1.0.0",
            "target": TARGET_COLUMN,
            "input_features": list(MODEL_INPUT_FEATURES),
        },
    )
    provenance = tmp_path / "data.provenance.json"
    atomic_write_json(
        provenance,
        {
            "schema_version": 1,
            "dataset": {
                "identifier": "synthetic-training-cohort",
                "sha256": "0" * 64,
            },
            "source": "synthetic unit-test fixture",
            "target": {
                "column": TARGET_COLUMN,
                "source": "MCQ160E",
                "mapping": {"1": 1, "2": 0},
            },
        },
    )
    selection = tmp_path / "selection_report.json"
    atomic_write_json(
        selection, _selection_report(provenance_sha256=sha256_file(provenance))
    )

    paths = {
        "model": model,
        "feature_config": config,
        "data_provenance": provenance,
        "selection_evidence": selection,
    }
    manifest = {
        "schema_version": 1,
        "run_id": "run-001",
        "model_id": "candidate-run-001",
        "status": "candidate_not_deployed",
        "strategy": "SMOTE",
        "created_at": "2026-01-01T00:00:00+00:00",
        "model": {"path": model.name, "sha256": sha256_file(model)},
        "feature_config": {
            "path": config.name,
            "schema_version": "1.0.0",
            "sha256": sha256_file(config),
        },
        "decision_threshold": {"value": 0.25, "source": "development_selection"},
        "data_provenance": {
            "path": provenance.name,
            "sha256": sha256_file(provenance),
        },
        "selection_evidence": {
            "path": selection.name,
            "sha256": sha256_file(selection),
        },
    }
    manifest_path = tmp_path / "candidate_manifest.json"
    atomic_write_json(manifest_path, manifest)
    return manifest_path, manifest, paths


def _rewrite_manifest(path: Path, manifest: dict) -> None:
    atomic_write_json(path, manifest)


def _rewrite_selection_component(
    manifest_path: Path,
    manifest: dict,
    paths: dict[str, Path],
    **updates,
) -> None:
    report = json.loads(paths["selection_evidence"].read_text(encoding="utf-8"))
    report.update(updates)
    atomic_write_json(paths["selection_evidence"], report)
    manifest["selection_evidence"]["sha256"] = sha256_file(paths["selection_evidence"])
    _rewrite_manifest(manifest_path, manifest)


def _candidate_evidence(precision_recall_pairs):
    return [
        {
            "candidate_id": f"candidate-{index}",
            "algorithm": f"Algorithm{index}",
            "development_metrics": {"precision": precision, "recall": recall},
            "status": "considered",
        }
        for index, (precision, recall) in enumerate(precision_recall_pairs)
    ]


def test_selection_accepts_single_candidate_meeting_precision_minimum():
    selected, evidence, _ = select_candidate_evidence(
        _candidate_evidence([(0.39, 0.90), (0.40, 0.60)])
    )

    assert selected == 1
    assert evidence[1]["status"] == "selected_for_tuning"


def test_selection_chooses_highest_recall_among_eligible_candidates():
    selected, _, _ = select_candidate_evidence(
        _candidate_evidence([(0.50, 0.60), (0.45, 0.80), (0.60, 0.70)])
    )

    assert selected == 1


def test_selection_fallback_uses_global_highest_recall():
    selected, evidence, rule = select_candidate_evidence(
        _candidate_evidence([(0.20, 0.50), (0.30, 0.85), (0.39, 0.70)])
    )

    assert selected == 1
    assert "none met precision" in rule
    assert evidence[1]["selection_reason"] == rule


def test_selection_tie_keeps_first_reproducible_candidate():
    selected, evidence, _ = select_candidate_evidence(
        _candidate_evidence([(0.50, 0.80), (0.60, 0.80)])
    )

    assert selected == 0
    assert "earlier candidate" in evidence[1]["rejection_reason"]


def test_selection_reasons_match_precision_and_recall_decisions():
    selected, evidence, _ = select_candidate_evidence(
        _candidate_evidence([(0.39, 0.95), (0.45, 0.80), (0.55, 0.70)])
    )

    assert selected == 1
    assert "precision below" in evidence[0]["rejection_reason"]
    assert "lower development recall" in evidence[2]["rejection_reason"]


def test_candidate_is_loaded_only_from_complete_valid_manifest(tmp_path):
    manifest_path, _, paths = _candidate_manifest(tmp_path)

    candidate = load_candidate_artifacts(manifest_path)

    assert candidate.model_id == "candidate-run-001"
    assert candidate.model_path == paths["model"]
    assert candidate.decision_threshold == pytest.approx(0.25)
    assert candidate.selection_report["protected_test_consulted"] is False


def test_candidate_identity_distinguishes_sidecar_from_historical_dataset(tmp_path):
    manifest_path, _, _ = _candidate_manifest(tmp_path)

    selected = resolve_validation_model(candidate_manifest=manifest_path)
    provenance = selected.artifact_integrity["data_provenance"]

    assert provenance["candidate_provenance_sidecar_verified"] is True
    assert provenance["training_dataset_digest_recorded"] is True
    assert provenance["training_dataset_file_reverified"] is False


def test_candidate_rejects_incorrect_model_hash(tmp_path):
    manifest_path, manifest, _ = _candidate_manifest(tmp_path)
    manifest["model"]["sha256"] = "0" * 64
    _rewrite_manifest(manifest_path, manifest)

    with pytest.raises(CandidateManifestError, match="hash_mismatch.*model"):
        load_candidate_artifacts(manifest_path)


def test_candidate_rejects_incorrect_configuration_hash(tmp_path):
    manifest_path, manifest, _ = _candidate_manifest(tmp_path)
    manifest["feature_config"]["sha256"] = "0" * 64
    _rewrite_manifest(manifest_path, manifest)

    with pytest.raises(CandidateManifestError, match="hash_mismatch.*feature_config"):
        load_candidate_artifacts(manifest_path)


def test_candidate_rejects_path_escaping_manifest_directory(tmp_path):
    manifest_path, manifest, _ = _candidate_manifest(tmp_path)
    manifest["model"]["path"] = "../outside.pkl"
    _rewrite_manifest(manifest_path, manifest)

    with pytest.raises(CandidateManifestError, match="unsafe_path"):
        load_candidate_artifacts(manifest_path)


def test_candidate_rejects_manifest_without_threshold(tmp_path):
    manifest_path, manifest, _ = _candidate_manifest(tmp_path)
    del manifest["decision_threshold"]
    _rewrite_manifest(manifest_path, manifest)

    with pytest.raises(CandidateManifestError, match="decision_threshold"):
        load_candidate_artifacts(manifest_path)


@pytest.mark.parametrize(
    "threshold",
    [0, 1, -0.1, 1.1, True, "0.25", float("nan"), float("inf"), float("-inf")],
)
def test_candidate_rejects_threshold_outside_open_interval(tmp_path, threshold):
    manifest_path, manifest, _ = _candidate_manifest(tmp_path)
    manifest["decision_threshold"]["value"] = threshold
    _rewrite_manifest(manifest_path, manifest)

    with pytest.raises(CandidateManifestError, match="invalid_threshold"):
        load_candidate_artifacts(manifest_path)


def test_candidate_fails_when_required_component_is_missing(tmp_path):
    manifest_path, _, paths = _candidate_manifest(tmp_path)
    paths["selection_evidence"].unlink()

    with pytest.raises(CandidateManifestError, match="missing_component"):
        load_candidate_artifacts(manifest_path)


def test_candidate_rejects_target_in_input_features(tmp_path):
    manifest_path, manifest, paths = _candidate_manifest(tmp_path)
    config = json.loads(paths["feature_config"].read_text(encoding="utf-8"))
    config["input_features"].append(TARGET_COLUMN)
    atomic_write_json(paths["feature_config"], config)
    manifest["feature_config"]["sha256"] = sha256_file(paths["feature_config"])
    _rewrite_manifest(manifest_path, manifest)

    with pytest.raises(CandidateManifestError, match="feature_contract_mismatch"):
        load_candidate_artifacts(manifest_path)


def test_candidate_rejects_inconsistent_selection_run_id(tmp_path):
    manifest_path, manifest, paths = _candidate_manifest(tmp_path)
    _rewrite_selection_component(
        manifest_path,
        manifest,
        paths,
        run_id="different-run",
    )

    with pytest.raises(CandidateManifestError, match="run_id"):
        load_candidate_artifacts(manifest_path)


def test_candidate_rejects_inconsistent_selection_strategy(tmp_path):
    manifest_path, manifest, paths = _candidate_manifest(tmp_path)
    _rewrite_selection_component(
        manifest_path,
        manifest,
        paths,
        strategy="SCALE_POS_WEIGHT",
    )

    with pytest.raises(CandidateManifestError, match="strategy"):
        load_candidate_artifacts(manifest_path)


def test_candidate_rejects_inconsistent_selection_threshold(tmp_path):
    manifest_path, manifest, paths = _candidate_manifest(tmp_path)
    _rewrite_selection_component(
        manifest_path,
        manifest,
        paths,
        selected_threshold=0.35,
    )

    with pytest.raises(CandidateManifestError, match="threshold"):
        load_candidate_artifacts(manifest_path)


def test_candidate_rejects_inconsistent_provenance_reference(tmp_path):
    manifest_path, manifest, paths = _candidate_manifest(tmp_path)
    report = json.loads(paths["selection_evidence"].read_text(encoding="utf-8"))
    report["data_provenance_reference"]["sha256"] = "f" * 64
    atomic_write_json(paths["selection_evidence"], report)
    manifest["selection_evidence"]["sha256"] = sha256_file(paths["selection_evidence"])
    _rewrite_manifest(manifest_path, manifest)

    with pytest.raises(CandidateManifestError, match="data provenance"):
        load_candidate_artifacts(manifest_path)


def test_candidate_rejects_invalid_json_component(tmp_path):
    manifest_path, manifest, paths = _candidate_manifest(tmp_path)
    paths["feature_config"].write_text("{invalid", encoding="utf-8")
    manifest["feature_config"]["sha256"] = sha256_file(paths["feature_config"])
    _rewrite_manifest(manifest_path, manifest)

    with pytest.raises(CandidateManifestError, match="not valid JSON"):
        load_candidate_artifacts(manifest_path)


def test_candidate_rejects_unsupported_manifest_schema(tmp_path):
    manifest_path, manifest, _ = _candidate_manifest(tmp_path)
    manifest["schema_version"] = 2
    _rewrite_manifest(manifest_path, manifest)

    with pytest.raises(CandidateManifestError, match="unsupported_schema"):
        load_candidate_artifacts(manifest_path)


def test_candidate_rejects_absolute_component_path(tmp_path):
    manifest_path, manifest, paths = _candidate_manifest(tmp_path)
    manifest["model"]["path"] = str(paths["model"].resolve())
    _rewrite_manifest(manifest_path, manifest)

    with pytest.raises(CandidateManifestError, match="unsafe_path"):
        load_candidate_artifacts(manifest_path)


def test_selection_report_records_all_candidates_and_protected_default():
    report = _selection_report()

    assert len(report["candidate_models_considered"]) == 2
    assert report["protected_test_consulted"] is False
    assert report["selected_threshold"] == pytest.approx(0.25)


def test_atomic_json_failure_preserves_previous_final_file(tmp_path):
    destination = tmp_path / "result.json"
    atomic_write_json(destination, {"state": "valid"})

    with pytest.raises(TypeError):
        atomic_write_json(destination, {"not_serializable": object()})

    assert json.loads(destination.read_text(encoding="utf-8")) == {"state": "valid"}
    assert not list(tmp_path.glob(".result.json.*.tmp"))


def test_atomic_csv_failure_preserves_previous_final_file(tmp_path, monkeypatch):
    destination = tmp_path / "predictions.csv"
    destination.write_text("row_id\n99\n", encoding="utf-8")

    def fail_after_partial_write(self, path, index=False):
        del self, index
        Path(path).write_text("partial", encoding="utf-8")
        raise OSError("synthetic write failure")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_after_partial_write)
    with pytest.raises(OSError, match="synthetic write failure"):
        atomic_write_csv(destination, pd.DataFrame({"row_id": [1]}))

    assert destination.read_text(encoding="utf-8") == "row_id\n99\n"
    assert not list(tmp_path.glob(".predictions.csv.*.tmp"))


def test_candidate_publication_failure_removes_staging_and_preserves_previous(tmp_path):
    staging = tmp_path / ".new-candidate.tmp"
    staging.mkdir()
    _candidate_manifest(staging)
    final = tmp_path / "candidate-existing"
    final.mkdir()
    marker = final / "candidate_manifest.json"
    marker.write_text('{"previous": true}', encoding="utf-8")

    with pytest.raises(FileExistsError):
        publish_candidate_directory(staging, final)

    assert not staging.exists()
    assert json.loads(marker.read_text(encoding="utf-8")) == {"previous": True}


def test_related_output_failure_preserves_previous_json_and_csv(tmp_path):
    result = tmp_path / "external_validation.json"
    predictions = tmp_path / "external_predictions.csv"
    result.write_text('{"state": "previous"}', encoding="utf-8")
    predictions.write_text("row_id\n99\n", encoding="utf-8")

    def invalid_result_factory(_name, _digest):
        return {"not_serializable": object()}

    with pytest.raises(TypeError):
        atomic_write_validation_outputs(
            result,
            predictions,
            pd.DataFrame({"row_id": [1]}),
            invalid_result_factory,
            expected_columns=("row_id",),
        )

    assert json.loads(result.read_text(encoding="utf-8")) == {"state": "previous"}
    assert predictions.read_text(encoding="utf-8") == "row_id\n99\n"
    assert not list(tmp_path.glob(".*.tmp"))
