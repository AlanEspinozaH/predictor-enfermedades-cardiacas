from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.data_pipeline as data_pipeline_module
from src.data_ingestion import load_and_process_data
from src.data_pipeline import (
    derive_ldl_with_friedewald,
    encode_alcohol_past_year,
    encode_heart_attack_target,
    load_training_provenance,
    prepare_modeling_cohort,
    read_tabular_data,
    split_development_and_test,
    validate_modeling_cohort,
)
from src.feature_contract import (
    MODEL_NUMERIC_FEATURES,
    TARGET_COLUMN,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _canonical_rows(row_count: int = 4) -> pd.DataFrame:
    data: dict[str, list[float | int | str]] = {}
    for column in MODEL_NUMERIC_FEATURES:
        data[column] = [1.0 + index for index in range(row_count)]

    data["Age"] = [30 + index for index in range(row_count)]
    data["Sex"] = [1, 2, 1, 2][:row_count]
    data["Race"] = [1, 2, 3, 4][:row_count]
    data["Education"] = [1, 2, 3, 4][:row_count]
    for column in ("Smoking", "PhysicalActivity", "HealthInsurance", "Alcohol"):
        data[column] = [0, 1, 0, 1][:row_count]
    data[TARGET_COLUMN] = [0, 1, 0, 1][:row_count]
    data["SEQN"] = list(range(100, 100 + row_count))
    return pd.DataFrame(data)


def test_target_encoding_excludes_refused_unknown_and_missing():
    encoded = encode_heart_attack_target(pd.Series([1, 2, 7, 9, np.nan, 0]))

    assert encoded.iloc[0] == 1
    assert encoded.iloc[1] == 0
    assert encoded.iloc[2:].isna().all()


def test_prepare_cohort_filters_age_and_invalid_target_without_imputing():
    frame = _canonical_rows()
    frame = frame.drop(columns=[TARGET_COLUMN])
    frame["Age"] = [19, 30, 40, 50]
    frame["MCQ160E"] = [1, 2, 7, 1]
    frame["Glucose"] = [90.0, np.nan, 100.0, 110.0]
    # Raw NHANES binary coding should become canonical 1/0.
    frame["Smoking"] = [1, 2, 1, 2]

    cohort = prepare_modeling_cohort(frame)

    assert cohort["SEQN"].tolist() == [101, 103]
    assert cohort[TARGET_COLUMN].tolist() == [0, 1]
    assert pd.isna(cohort.loc[0, "Glucose"])
    assert cohort["Smoking"].tolist() == [0, 0]
    validate_modeling_cohort(cohort)


def test_validation_rejects_fractional_imputed_categories():
    frame = _canonical_rows()
    frame["Education"] = frame["Education"].astype(float)
    frame.loc[0, "Education"] = 2.5

    with pytest.raises(ValueError, match="Education.*invalid values"):
        validate_modeling_cohort(frame)


def test_alcohol_is_harmonized_across_questionnaire_versions():
    older = encode_alcohol_past_year(pd.DataFrame({"ALQ120Q": [0, 1, 365, 777, 999]}))
    newer = encode_alcohol_past_year(pd.DataFrame({"ALQ121": [0, 1, 10, 77, 99]}))

    assert older.tolist()[:3] == [0, 1, 1]
    assert newer.tolist()[:3] == [0, 1, 1]
    assert older.iloc[3:].isna().all()
    assert newer.iloc[3:].isna().all()


def test_prepare_cohort_rejects_fractional_preimputed_binary_values():
    frame = _canonical_rows()
    frame["Smoking"] = frame["Smoking"].astype(float)
    frame.loc[0, "Smoking"] = 0.42

    with pytest.raises(ValueError, match="Binary feature contains invalid values"):
        prepare_modeling_cohort(frame)


def test_training_provenance_must_be_verified_and_unimputed(tmp_path):
    dataset = tmp_path / "cohort.parquet"
    dataset.touch()
    provenance = dataset.with_suffix(".provenance.json")
    provenance.write_text(
        '{"target_source":"MCQ160E","target_provenance_verified":true,'
        '"minimum_age":20,"statistical_imputation_applied":false}',
        encoding="utf-8",
    )

    loaded = load_training_provenance(dataset)
    assert loaded["target_source"] == "MCQ160E"

    provenance.write_text(
        '{"target_source":"MCQ160E","target_provenance_verified":true,'
        '"minimum_age":20,"statistical_imputation_applied":true}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="statistical_imputation_applied"):
        load_training_provenance(dataset)


def test_ingestion_preserves_verified_target_provenance(tmp_path, monkeypatch):
    source = tmp_path / "canonical.csv"
    output = tmp_path / "intermediate.parquet"
    _canonical_rows().to_csv(source, index=False)
    source.with_suffix(".provenance.json").write_text(
        '{"target_source":"MCQ160E","target_provenance_verified":true,'
        '"minimum_age":20,"statistical_imputation_applied":false}',
        encoding="utf-8",
    )

    def fake_to_parquet(self, path, index=False):
        del self, index
        Path(path).touch()

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)
    load_and_process_data(source, output)

    metadata = load_training_provenance(output)
    assert metadata["target_source"] == "MCQ160E"
    assert metadata["target_provenance_verified"] is True


def test_friedewald_only_fills_clinically_eligible_missing_ldl():
    frame = pd.DataFrame(
        {
            "TotalCholesterol": [200.0, 200.0, 100.0, 200.0],
            "HDL": [50.0, 50.0, 90.0, 50.0],
            "Triglycerides": [100.0, 400.0, 100.0, 100.0],
            "LDL": [np.nan, np.nan, np.nan, 125.0],
        }
    )

    result = derive_ldl_with_friedewald(frame)

    assert result.loc[0, "LDL"] == pytest.approx(130.0)
    assert pd.isna(result.loc[1, "LDL"])
    assert pd.isna(result.loc[2, "LDL"])
    assert result.loc[3, "LDL"] == pytest.approx(125.0)


def test_temporal_split_holds_out_latest_nhanes_cycle():
    frame = _canonical_rows(4)
    older = frame.copy()
    older["NHANESCycle"] = "2011-2012"
    newer = frame.copy()
    newer["SEQN"] += 1000
    newer["NHANESCycle"] = "2017-2020"
    combined = pd.concat([older, newer], ignore_index=True)

    split = split_development_and_test(combined)

    assert split.metadata["strategy"] == "temporal_latest_nhanes_cycle"
    assert split.metadata["held_out_cycles"] == ["2017-2020"]
    assert set(split.development["NHANESCycle"]) == {"2011-2012"}
    assert set(split.independent_test["NHANESCycle"]) == {"2017-2020"}
    assert set(split.development["SEQN"]).isdisjoint(
        set(split.independent_test["SEQN"])
    )
    assert split.metadata["imputation_fitted_before_split"] is False


def test_read_xpt_accepts_dataframe_return_without_tuple_unpacking(monkeypatch):
    expected = pd.DataFrame({"SEQN": [1, 2]})

    def fake_read_sas(*args, **kwargs):
        assert kwargs["format"] == "xport"
        return expected

    monkeypatch.setattr(data_pipeline_module.pd, "read_sas", fake_read_sas)

    actual = read_tabular_data("example.XPT")

    pd.testing.assert_frame_equal(actual, expected)


def test_stage4_sources_do_not_preimpute_or_use_stroke_as_target():
    ingestion_source = (PROJECT_ROOT / "src" / "data_ingestion.py").read_text(
        encoding="utf-8"
    )
    collection_source = (PROJECT_ROOT / "data" / "02_processed" / "carga.py").read_text(
        encoding="utf-8"
    )
    training_source = (PROJECT_ROOT / "src" / "train_pycaret.py").read_text(
        encoding="utf-8"
    )

    assert "IterativeImputer" not in ingestion_source
    assert "fit_transform" not in ingestion_source
    assert '"TARGET": "MCQ160F"' not in collection_source
    assert "target_candidates" not in collection_source
    assert '"HDL": "HDL_' in collection_source
    assert '"alcohol_column": "ALQ120Q"' in collection_source
    assert '"alcohol_column": "ALQ121"' in collection_source
    assert "ALQ101" not in collection_source
    assert "split_development_and_test" in training_source
    assert training_source.index("split_development_and_test") < training_source.index(
        "setup(**setup_args)"
    )
    assert "--evaluate-protected-test" in training_source
    assert training_source.index(
        "final_model = finalize_model(tuned_model)"
    ) < training_source.index("data=independent_test")
