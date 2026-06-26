import json

import pytest

from src.artifact_registry import (
    PROJECT_ROOT,
    ArtifactManifestError,
    load_deployed_artifacts,
    pycaret_model_stem,
    sha256_file,
)


def test_deployed_manifest_resolves_canonical_artifacts():
    artifacts = load_deployed_artifacts(
        verify_hashes=True,
        require_training_data=True,
    )

    assert artifacts.model_id == "nhanes-heart-disease-pycaret-legacy-v1"
    assert artifacts.model_path == PROJECT_ROOT / "models" / "best_pipeline.pkl"
    assert (
        artifacts.feature_config_path == PROJECT_ROOT / "models" / "model_config.json"
    )
    assert artifacts.training_data_path == (
        PROJECT_ROOT / "data" / "02_intermediate" / "process_data.parquet"
    )
    assert artifacts.decision_threshold == pytest.approx(0.2)
    assert sha256_file(artifacts.model_path) == artifacts.model_sha256


def test_pycaret_model_stem_removes_only_pkl_suffix():
    model_path = PROJECT_ROOT / "models" / "best_pipeline.pkl"

    assert pycaret_model_stem(model_path) == str(model_path.with_suffix(""))


def test_manifest_rejects_model_hash_mismatch(tmp_path):
    manifest_path = PROJECT_ROOT / "models" / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model"]["sha256"] = "0" * 64
    invalid_manifest = tmp_path / "invalid_manifest.json"
    invalid_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactManifestError, match="integrity check failed"):
        load_deployed_artifacts(manifest_path=invalid_manifest, verify_hashes=True)


def test_manifest_rejects_paths_outside_repository(tmp_path):
    manifest_path = PROJECT_ROOT / "models" / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model"]["path"] = "../outside.pkl"
    invalid_manifest = tmp_path / "path_traversal_manifest.json"
    invalid_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactManifestError, match="escapes the repository"):
        load_deployed_artifacts(manifest_path=invalid_manifest, verify_hashes=False)


def test_runtime_scripts_do_not_select_model_by_filename_fallback():
    app_source = (PROJECT_ROOT / "src" / "app.py").read_text(encoding="utf-8")
    audit_source = (PROJECT_ROOT / "src" / "audit_fairness.py").read_text(
        encoding="utf-8"
    )
    training_source = (PROJECT_ROOT / "src" / "train_pycaret.py").read_text(
        encoding="utf-8"
    )

    assert "models/best_pipeline" not in app_source
    assert "final_pipeline_v1" not in audit_source
    assert "candidate_not_deployed" in training_source
    assert "final_pipeline_v1" not in training_source


class _FakeEstimator:
    def __init__(self, feature_names):
        self.feature_names_in_ = feature_names


class _FakePyCaretPipeline:
    def __init__(self, metadata_features, estimator_features):
        self.feature_names_in_ = metadata_features
        self.named_steps = {
            "actual_estimator": _FakeEstimator(estimator_features),
        }


def test_pycaret_metadata_may_include_only_the_canonical_target():
    import numpy as np

    from src.artifact_registry import validate_deployed_pipeline_contract
    from src.feature_contract import MODEL_INPUT_FEATURES, TARGET_COLUMN

    transformed_features = [
        *MODEL_INPUT_FEATURES,
        "Race_1.0",
        "Race_2.0",
        "Race_3.0",
        "Race_4.0",
    ]
    pipeline = _FakePyCaretPipeline(
        np.array([*MODEL_INPUT_FEATURES, TARGET_COLUMN]),
        np.array(transformed_features),
    )

    normalized = validate_deployed_pipeline_contract(pipeline)

    assert normalized == MODEL_INPUT_FEATURES
    assert TARGET_COLUMN not in normalized


def test_pycaret_metadata_does_not_ignore_other_unexpected_features():
    import numpy as np

    from src.artifact_registry import validate_deployed_pipeline_contract
    from src.feature_contract import MODEL_INPUT_FEATURES, TARGET_COLUMN

    pipeline = _FakePyCaretPipeline(
        np.array([*MODEL_INPUT_FEATURES, TARGET_COLUMN, "UnexpectedFeature"]),
        np.array(MODEL_INPUT_FEATURES),
    )

    with pytest.raises(ValueError, match="UnexpectedFeature"):
        validate_deployed_pipeline_contract(pipeline)


def test_duplicate_target_metadata_is_not_silently_removed_twice():
    import numpy as np

    from src.artifact_registry import validate_deployed_pipeline_contract
    from src.feature_contract import MODEL_INPUT_FEATURES, TARGET_COLUMN

    pipeline = _FakePyCaretPipeline(
        np.array([*MODEL_INPUT_FEATURES, TARGET_COLUMN, TARGET_COLUMN]),
        np.array(MODEL_INPUT_FEATURES),
    )

    with pytest.raises(ValueError, match="HeartDisease"):
        validate_deployed_pipeline_contract(pipeline)


def test_target_in_final_estimator_is_rejected_as_leakage():
    import numpy as np

    from src.artifact_registry import validate_deployed_pipeline_contract
    from src.feature_contract import MODEL_INPUT_FEATURES, TARGET_COLUMN

    pipeline = _FakePyCaretPipeline(
        np.array([*MODEL_INPUT_FEATURES, TARGET_COLUMN]),
        np.array([*MODEL_INPUT_FEATURES, TARGET_COLUMN]),
    )

    with pytest.raises(
        ValueError,
        match="Target leakage detected in deployed estimator",
    ):
        validate_deployed_pipeline_contract(pipeline)
