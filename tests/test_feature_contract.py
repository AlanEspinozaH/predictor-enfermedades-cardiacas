import json
from pathlib import Path

from src.feature_contract import (
    FEATURE_SCHEMA_VERSION,
    MINIMUM_ELIGIBLE_AGE,
    MODEL_CATEGORICAL_FEATURES,
    MODEL_IGNORED_FEATURES,
    MODEL_INPUT_FEATURES,
    MODEL_NUMERIC_FEATURES,
    TARGET_COLUMN,
    feature_names_from_config,
    validate_feature_names,
)
from src.interfaces import InputData

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_model_config_matches_python_contract():
    config_path = PROJECT_ROOT / "models" / "model_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["schema_version"] == FEATURE_SCHEMA_VERSION
    assert config["target"] == TARGET_COLUMN
    assert tuple(config["numeric_features"]) == MODEL_NUMERIC_FEATURES
    assert tuple(config["categorical_features"]) == MODEL_CATEGORICAL_FEATURES
    assert tuple(config["ignore_features"]) == MODEL_IGNORED_FEATURES
    assert feature_names_from_config(config) == MODEL_INPUT_FEATURES
    assert config["minimum_age"] == MINIMUM_ELIGIBLE_AGE
    validate_feature_names(feature_names_from_config(config))


def test_pydantic_input_fields_match_model_features():
    assert tuple(InputData.model_fields) == MODEL_INPUT_FEATURES


def test_contract_includes_hdl_and_excludes_obsolete_or_identifier_columns():
    assert "HDL" in MODEL_INPUT_FEATURES
    assert "DiastolicBP" not in MODEL_INPUT_FEATURES
    assert "SEQN" not in MODEL_INPUT_FEATURES
