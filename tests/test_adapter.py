import unittest

import pandas as pd

from src.adapters import UserInputAdapter
from src.feature_contract import MODEL_INPUT_FEATURES, SexCode


class TestUserInputAdapter(unittest.TestCase):
    def setUp(self):
        self.adapter = UserInputAdapter()
        self.valid_input = {
            "Age": 45,
            "IncomeRatio": 2.5,
            "SystolicBP": 120.0,
            "BMI": 24.5,
            "WaistCircumference": 90.0,
            "Height": 175.0,
            "TotalCholesterol": 200.0,
            "Triglycerides": 150.0,
            "LDL": 100.0,
            "HDL": 50.0,
            "HbA1c": 5.5,
            "Glucose": 90.0,
            "Creatinine": 0.9,
            "UricAcid": 5.0,
            "ALT_Enzyme": 25.0,
            "Albumin": 4.5,
            "Potassium": 4.0,
            "Sodium": 140.0,
            "GGT_Enzyme": 30.0,
            "AST_Enzyme": 22.0,
            "Sex": int(SexCode.MALE),
            "Race": 3,
            "Education": 4,
            "Smoking": 0,
            "PhysicalActivity": 1,
            "HealthInsurance": 1,
            "Alcohol": 0,
        }

    def test_transform_valid_input_uses_exact_contract_order(self):
        df = self.adapter.transform(self.valid_input)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertEqual(tuple(df.columns), MODEL_INPUT_FEATURES)
        self.assertEqual(df.loc[0, "Age"], 45)
        self.assertEqual(df.loc[0, "HDL"], 50.0)
        self.assertEqual(df.loc[0, "Sex"], int(SexCode.MALE))

    def test_female_nhanes_code_is_accepted(self):
        input_data = self.valid_input.copy()
        input_data["Sex"] = int(SexCode.FEMALE)

        df = self.adapter.transform(input_data)

        self.assertEqual(df.loc[0, "Sex"], int(SexCode.FEMALE))

    def test_missing_required_hdl_raises_error(self):
        input_data = self.valid_input.copy()
        del input_data["HDL"]

        with self.assertRaises(ValueError):
            self.adapter.transform(input_data)

    def test_obsolete_diastolic_bp_is_rejected(self):
        input_data = self.valid_input.copy()
        input_data["DiastolicBP"] = 80.0

        with self.assertRaises(ValueError):
            self.adapter.transform(input_data)

    def test_legacy_zero_sex_code_is_rejected(self):
        input_data = self.valid_input.copy()
        input_data["Sex"] = 0

        with self.assertRaises(ValueError):
            self.adapter.transform(input_data)

    def test_age_below_target_population_is_rejected(self):
        input_data = self.valid_input.copy()
        input_data["Age"] = 19

        with self.assertRaises(ValueError):
            self.adapter.transform(input_data)

    def test_invalid_type_raises_error(self):
        input_data = self.valid_input.copy()
        input_data["Age"] = "Not a number"

        with self.assertRaises(ValueError):
            self.adapter.transform(input_data)


if __name__ == "__main__":
    unittest.main()
