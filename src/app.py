"""Streamlit interface for the explicitly deployed legacy academic model."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.adapters import PyCaretAdapter, UserInputAdapter
from src.artifact_registry import (
    ArtifactManifestError,
    DeployedArtifacts,
    load_deployed_artifacts,
    load_validated_pycaret_pipeline,
)
from src.feature_contract import (
    MINIMUM_ELIGIBLE_AGE,
    SexCode,
    feature_names_from_config,
    validate_feature_names,
)

st.set_page_config(
    page_title="Prototipo NHANES de clasificación de antecedente de infarto",
    page_icon="🫀",
    layout="wide",
)


@st.cache_resource
def load_runtime_artifacts() -> tuple[DeployedArtifacts, dict[str, Any]]:
    artifacts = load_deployed_artifacts(verify_hashes=True)
    loaded_config = json.loads(
        artifacts.feature_config_path.read_text(encoding="utf-8")
    )
    validate_feature_names(feature_names_from_config(loaded_config))
    return artifacts, loaded_config


@st.cache_resource
def load_model_pipeline(
    model_path: Path,
    expected_features: tuple[str, ...],
) -> PyCaretAdapter:
    pipeline = load_validated_pycaret_pipeline(
        model_path,
        expected_features=expected_features,
    )
    return PyCaretAdapter(pipeline, expected_features=expected_features)


try:
    artifacts, config = load_runtime_artifacts()
    expected_features = feature_names_from_config(config)
    model = load_model_pipeline(artifacts.model_path, expected_features)
except (
    ArtifactManifestError,
    OSError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
) as exc:
    st.error(f"No se pudo validar o cargar el artefacto desplegado: {exc}")
    st.stop()
except Exception as exc:
    st.error(f"No se pudo cargar el pipeline PyCaret: {exc}")
    st.stop()

threshold = artifacts.decision_threshold

with st.sidebar:
    st.header("🫀 Prototipo académico")
    st.warning(
        "Este sistema no predice un infarto futuro ni constituye un diagnóstico. "
        "El objetivo histórico del modelo es una respuesta autorreportada de "
        "antecedente de infarto en NHANES."
    )
    st.caption(f"Modelo: {artifacts.model_id}")
    st.caption(f"Umbral del manifiesto: {threshold:.2f}")
    st.caption("El umbral heredado no tiene validación clínica independiente.")

st.title("Clasificación exploratoria de antecedente autorreportado de infarto")
st.write(
    "Complete las 27 variables exigidas por el contrato del modelo. "
    "La salida representa la probabilidad asignada por un artefacto académico "
    "heredado a la clase positiva; no debe interpretarse como riesgo clínico futuro."
)

with st.form("patient_data_form"):
    st.subheader("Datos demográficos")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        age = st.number_input(
            "Edad (años)",
            min_value=MINIMUM_ELIGIBLE_AGE,
            max_value=100,
            value=45,
        )
        race_map = {
            "Mexicano-estadounidense": 1,
            "Otro origen hispano": 2,
            "Blanco no hispano": 3,
            "Negro no hispano": 4,
            "Otra categoría": 5,
        }
        race = race_map[st.selectbox("Raza/etnicidad (código NHANES)", race_map)]
    with col2:
        sex_label = st.radio(
            "Sexo (código NHANES)", ["Mujer", "Hombre"], horizontal=True
        )
        sex = int(SexCode.FEMALE if sex_label == "Mujer" else SexCode.MALE)
        education_map = {
            "Menos de 9.º grado": 1,
            "9.º–11.º grado": 2,
            "Secundaria/GED": 3,
            "Estudios superiores incompletos": 4,
            "Graduado universitario": 5,
        }
        education = education_map[st.selectbox("Educación", education_map)]
    with col3:
        height = st.number_input("Estatura (cm)", 130.0, 220.0, 170.0)
        income = st.slider("Índice ingreso/pobreza (PIR)", 0.0, 5.0, 2.5)
    with col4:
        waist = st.number_input("Circunferencia de cintura (cm)", 50.0, 180.0, 90.0)

    st.subheader("Signos y medidas")
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        bmi = st.number_input("IMC", 12.0, 60.0, 25.0, format="%.1f")
    with col_v2:
        systolic_bp = st.slider("Presión sistólica (mmHg)", 80.0, 220.0, 120.0)

    st.subheader("Perfil bioquímico")
    col_b1, col_b2, col_b3, col_b4 = st.columns(4)
    with col_b1:
        total_cholesterol = st.number_input(
            "Colesterol total (mg/dL)", 100.0, 400.0, 200.0
        )
        ldl = st.number_input("LDL (mg/dL)", 30.0, 300.0, 100.0)
    with col_b2:
        hdl = st.number_input("HDL (mg/dL)", 10.0, 150.0, 50.0)
        triglycerides = st.number_input("Triglicéridos (mg/dL)", 30.0, 600.0, 150.0)
    with col_b3:
        hba1c = st.number_input("HbA1c (%)", 4.0, 15.0, 5.7, step=0.1)
        glucose = st.number_input("Glucosa (mg/dL)", 50.0, 300.0, 90.0)
    with col_b4:
        uric_acid = st.number_input("Ácido úrico (mg/dL)", 2.0, 12.0, 5.0, step=0.1)
        creatinine = st.number_input("Creatinina (mg/dL)", 0.4, 5.0, 0.9, step=0.1)

    with st.expander("Enzimas y electrolitos"):
        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            alt = st.number_input("ALT (U/L)", 5.0, 200.0, 25.0)
            albumin = st.number_input("Albúmina (g/dL)", 2.0, 6.0, 4.5)
        with col_e2:
            ast = st.number_input("AST (U/L)", 5.0, 200.0, 25.0)
            potassium = st.number_input("Potasio (mmol/L)", 2.0, 6.0, 4.0)
        with col_e3:
            ggt = st.number_input("GGT (U/L)", 5.0, 200.0, 25.0)
            sodium = st.number_input("Sodio (mmol/L)", 120.0, 160.0, 140.0)

    st.subheader("Estilo de vida y acceso a salud")
    col_l1, col_l2, col_l3, col_l4 = st.columns(4)
    with col_l1:
        smoking = st.checkbox("Fumó al menos 100 cigarrillos en su vida")
    with col_l2:
        alcohol = st.checkbox("Consumió alcohol durante el último año")
    with col_l3:
        physical_activity = st.checkbox("Realiza actividad física recreativa vigorosa")
    with col_l4:
        health_insurance = st.checkbox("Cuenta con seguro de salud", value=True)

    submitted = st.form_submit_button("Ejecutar clasificación del prototipo")

if submitted:
    user_input = {
        "Age": age,
        "IncomeRatio": income,
        "SystolicBP": systolic_bp,
        "BMI": bmi,
        "WaistCircumference": waist,
        "Height": height,
        "TotalCholesterol": total_cholesterol,
        "Triglycerides": triglycerides,
        "LDL": ldl,
        "HDL": hdl,
        "HbA1c": hba1c,
        "Glucose": glucose,
        "Creatinine": creatinine,
        "UricAcid": uric_acid,
        "ALT_Enzyme": alt,
        "Albumin": albumin,
        "Potassium": potassium,
        "Sodium": sodium,
        "GGT_Enzyme": ggt,
        "AST_Enzyme": ast,
        "Sex": sex,
        "Race": race,
        "Education": education,
        "Smoking": int(smoking),
        "PhysicalActivity": int(physical_activity),
        "HealthInsurance": int(health_insurance),
        "Alcohol": int(alcohol),
    }

    try:
        input_frame = UserInputAdapter(expected_features).transform(user_input)
        probability = float(model.predict_proba(input_frame)[0])
        positive = probability >= threshold

        st.divider()
        st.metric("Probabilidad asignada a la clase positiva", f"{probability:.2%}")
        if positive:
            st.warning(
                "Clasificación positiva del prototipo: el patrón de entrada supera "
                f"el umbral heredado de {threshold:.2f}."
            )
        else:
            st.info(
                "Clasificación negativa del prototipo: el patrón de entrada no "
                f"supera el umbral heredado de {threshold:.2f}."
            )
        st.caption(
            "Esta salida no demuestra presencia o ausencia de enfermedad y no "
            "estima la probabilidad de un evento futuro."
        )
    except (TypeError, ValueError) as exc:
        st.error(f"Entrada o salida incompatible con el contrato: {exc}")
    except Exception as exc:
        st.error(f"La inferencia falló: {exc}")
