from typing import Literal, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.feature_contract import MINIMUM_ELIGIBLE_AGE


@runtime_checkable
class HeartDiseaseModel(Protocol):
    """Interface shared by educational and production model adapters."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model."""
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return positive-class probabilities as a one-dimensional array."""
        ...

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return class labels using the supplied decision threshold."""
        ...


class InputData(BaseModel):
    """Validated input contract for the deployed NHANES model.

    ``Sex`` retains the original NHANES RIAGENDR coding used by the current
    model: 1=male and 2=female.  Extra fields are rejected so that obsolete UI
    variables cannot be silently passed to the model.
    """

    model_config = ConfigDict(extra="forbid")

    Age: int = Field(
        ...,
        ge=MINIMUM_ELIGIBLE_AGE,
        le=100,
        description="Age in years; the target questionnaire applies to adults 20+.",
    )
    IncomeRatio: float = Field(..., ge=0.0, le=5.0, description="Poverty Income Ratio")
    SystolicBP: float = Field(
        ..., ge=80, le=220, description="Systolic blood pressure (mmHg)"
    )
    BMI: float = Field(..., ge=12.0, le=60.0, description="Body Mass Index")
    WaistCircumference: float = Field(
        ..., ge=50, le=180, description="Waist circumference (cm)"
    )
    Height: float = Field(..., ge=130, le=220, description="Height (cm)")
    TotalCholesterol: float = Field(
        ..., ge=100, le=400, description="Total cholesterol (mg/dL)"
    )
    Triglycerides: float = Field(
        ..., ge=30, le=600, description="Triglycerides (mg/dL)"
    )
    LDL: float = Field(..., ge=30, le=300, description="LDL cholesterol (mg/dL)")
    HDL: float = Field(..., ge=10, le=150, description="HDL cholesterol (mg/dL)")
    HbA1c: float = Field(..., ge=4.0, le=15.0, description="Glycated hemoglobin (%)")
    Glucose: float = Field(..., ge=50, le=300, description="Fasting glucose (mg/dL)")
    Creatinine: float = Field(
        ..., ge=0.4, le=5.0, description="Serum creatinine (mg/dL)"
    )
    UricAcid: float = Field(..., ge=2.0, le=12.0, description="Uric acid (mg/dL)")
    ALT_Enzyme: float = Field(..., ge=5, le=200, description="ALT (U/L)")
    Albumin: float = Field(..., ge=2.0, le=6.0, description="Albumin (g/dL)")
    Potassium: float = Field(..., ge=2.0, le=6.0, description="Potassium (mmol/L)")
    Sodium: float = Field(..., ge=120, le=160, description="Sodium (mmol/L)")
    GGT_Enzyme: float = Field(..., ge=5, le=200, description="GGT (U/L)")
    AST_Enzyme: float = Field(..., ge=5, le=200, description="AST (U/L)")

    Sex: Literal[1, 2] = Field(..., description="NHANES code: 1=Male, 2=Female")
    Race: Literal[1, 2, 3, 4, 5] = Field(
        ...,
        description="1=Mexican American, 2=Other Hispanic, 3=White, 4=Black, 5=Other",
    )
    Education: Literal[1, 2, 3, 4, 5] = Field(
        ..., description="NHANES education category (1=lowest, 5=highest)"
    )
    Smoking: Literal[0, 1] = Field(
        ..., description="Smoked at least 100 cigarettes: 0=No, 1=Yes"
    )
    PhysicalActivity: Literal[0, 1] = Field(
        ..., description="Vigorous recreational activity: 0=No, 1=Yes"
    )
    HealthInsurance: Literal[0, 1] = Field(
        ..., description="Has health insurance: 0=No, 1=Yes"
    )
    Alcohol: Literal[0, 1] = Field(
        ..., description="Any alcohol use during the past 12 months: 0=No, 1=Yes"
    )
