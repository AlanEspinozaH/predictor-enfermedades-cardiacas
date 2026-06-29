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
    deployed_estimator,
    load_deployed_artifacts,
    load_validated_pycaret_pipeline,
)
from src.feature_contract import (
    MINIMUM_ELIGIBLE_AGE,
    feature_names_from_config,
    validate_feature_names,
)

st.set_page_config(
    page_title="CardioHistory ML",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
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
):
    st.error(
        "No se pudo validar el despliegue. Revise el manifiesto, la integridad "
        "de los artefactos y el contrato de variables."
    )
    st.stop()
except Exception:
    st.error(
        "No se pudo cargar el pipeline PyCaret. Compruebe que utiliza el entorno "
        "Python 3.10 y las dependencias declaradas por el proyecto."
    )
    st.stop()

threshold = artifacts.decision_threshold
pipeline_estimator = deployed_estimator(model.model)
estimator_name = type(pipeline_estimator).__name__
transformed_features = tuple(
    str(name) for name in pipeline_estimator.feature_names_in_
)
manifest_status = str(artifacts.manifest.get("status", "no declarado"))

with st.sidebar:
    st.header("Ficha técnica del despliegue")
    st.markdown(f"**Modelo:** {estimator_name}")
    st.markdown("**Estado:** prototipo heredado")
    st.caption(f"Identificador: {artifacts.model_id}")
    st.caption(f"Estado del manifiesto: {manifest_status}")
    st.markdown(f"**Entrada:** {len(expected_features)} variables")
    st.markdown(
        f"**Transformación:** {len(transformed_features)} características"
    )
    st.markdown("**Clase positiva:** 1")
    st.markdown(f"**Umbral operativo:** {threshold:.2f}")
    st.info("Integridad verificada mediante manifiesto y SHA-256.")
    st.caption(
        "El umbral es heredado y no cuenta con validación independiente."
    )

st.title("CardioHistory ML")
st.subheader(
    "Clasificación explicable de antecedente autorreportado de infarto con "
    "datos NHANES"
)
st.write(
    "Panel académico para observar un flujo reproducible de inferencia tabular "
    "con validación de entradas e integridad de artefactos."
)
st.info("Prototipo académico — no es diagnóstico ni predicción de un evento futuro")
st.caption(
    "El objetivo es histórico y autorreportado: corresponde a la respuesta "
    "asociada con MCQ160E en NHANES."
)

with st.container(border=True):
    st.subheader("Resumen del flujo")
    flow_columns = st.columns(6)
    flow_steps = (
        (f"{len(expected_features)} variables", "Entrada externa"),
        ("PyCaret", "Preprocesamiento"),
        (f"{len(transformed_features)} características", "Entrada transformada"),
        (estimator_name, "Estimador final"),
        ("Clase positiva", "Score de clase 1"),
        (f"Umbral {threshold:.2f}", "Clasificación académica"),
    )
    for column, (primary, secondary) in zip(flow_columns, flow_steps, strict=True):
        with column:
            st.markdown(f"**{primary}**")
            st.caption(secondary)

st.subheader("Entrada del modelo")
st.caption(
    "Complete o confirme las 27 variables. Los códigos se conservan en su "
    "forma NHANES y Pydantic realiza la validación final."
)

with st.form("patient_data_form", border=True):
    profile_tab, measurements_tab, habits_tab = st.tabs(
        (
            "A. Perfil y contexto",
            "B. Mediciones antropométricas y bioquímicas",
            "C. Hábitos",
        )
    )

    with profile_tab:
        profile_left, profile_center, profile_right = st.columns(3)
        with profile_left:
            age = st.number_input(
                "Edad (años)",
                min_value=MINIMUM_ELIGIBLE_AGE,
                max_value=100,
                value=45,
                key="input_Age",
            )
            income = st.slider(
                "Índice ingreso/pobreza (PIR)",
                min_value=0.0,
                max_value=5.0,
                value=2.5,
                key="input_IncomeRatio",
            )
        with profile_center:
            sex_options = {
                "1 — Hombre": 1,
                "2 — Mujer": 2,
            }
            sex_label = st.radio(
                "Sexo (código NHANES)",
                options=tuple(sex_options),
                index=1,
                horizontal=True,
                key="input_Sex",
            )
            sex = sex_options[sex_label]

            race_options = {
                "1 — Mexicano-estadounidense": 1,
                "2 — Otro origen hispano": 2,
                "3 — Blanco no hispano": 3,
                "4 — Negro no hispano": 4,
                "5 — Otra categoría": 5,
            }
            race_label = st.selectbox(
                "Raza/etnicidad (código NHANES)",
                options=tuple(race_options),
                key="input_Race",
            )
            race = race_options[race_label]
        with profile_right:
            education_options = {
                "1 — Menos de 9.º grado": 1,
                "2 — 9.º–11.º grado": 2,
                "3 — Secundaria/GED": 3,
                "4 — Estudios superiores incompletos": 4,
                "5 — Graduado universitario": 5,
            }
            education_label = st.selectbox(
                "Educación (código NHANES)",
                options=tuple(education_options),
                key="input_Education",
            )
            education = education_options[education_label]
            health_insurance = st.checkbox(
                "Cuenta con seguro de salud (0 = No, 1 = Sí)",
                value=True,
                key="input_HealthInsurance",
            )

    with measurements_tab:
        st.markdown("**Mediciones antropométricas y signos**")
        measure_1, measure_2, measure_3, measure_4 = st.columns(4)
        with measure_1:
            systolic_bp = st.slider(
                "Presión arterial sistólica (mmHg)",
                min_value=80.0,
                max_value=220.0,
                value=120.0,
                key="input_SystolicBP",
            )
        with measure_2:
            bmi = st.number_input(
                "Índice de masa corporal (IMC)",
                min_value=12.0,
                max_value=60.0,
                value=25.0,
                format="%.1f",
                key="input_BMI",
            )
        with measure_3:
            waist = st.number_input(
                "Circunferencia de cintura (cm)",
                min_value=50.0,
                max_value=180.0,
                value=90.0,
                key="input_WaistCircumference",
            )
        with measure_4:
            height = st.number_input(
                "Estatura (cm)",
                min_value=130.0,
                max_value=220.0,
                value=170.0,
                key="input_Height",
            )

        st.markdown("**Perfil bioquímico**")
        bio_1, bio_2, bio_3, bio_4 = st.columns(4)
        with bio_1:
            total_cholesterol = st.number_input(
                "Colesterol total (mg/dL)",
                min_value=100.0,
                max_value=400.0,
                value=200.0,
                key="input_TotalCholesterol",
            )
            ldl = st.number_input(
                "Colesterol LDL (mg/dL)",
                min_value=30.0,
                max_value=300.0,
                value=100.0,
                key="input_LDL",
            )
            hdl = st.number_input(
                "Colesterol HDL (mg/dL)",
                min_value=10.0,
                max_value=150.0,
                value=50.0,
                key="input_HDL",
            )
            triglycerides = st.number_input(
                "Triglicéridos (mg/dL)",
                min_value=30.0,
                max_value=600.0,
                value=150.0,
                key="input_Triglycerides",
            )
        with bio_2:
            hba1c = st.number_input(
                "Hemoglobina glucosilada, HbA1c (%)",
                min_value=4.0,
                max_value=15.0,
                value=5.7,
                step=0.1,
                key="input_HbA1c",
            )
            glucose = st.number_input(
                "Glucosa sérica del perfil bioquímico NHANES (LBXSGL) (mg/dL)",
                min_value=50.0,
                max_value=300.0,
                value=90.0,
                key="input_Glucose",
            )
            creatinine = st.number_input(
                "Creatinina sérica (mg/dL)",
                min_value=0.4,
                max_value=5.0,
                value=0.9,
                step=0.1,
                key="input_Creatinine",
            )
            uric_acid = st.number_input(
                "Ácido úrico (mg/dL)",
                min_value=2.0,
                max_value=12.0,
                value=5.0,
                step=0.1,
                key="input_UricAcid",
            )
        with bio_3:
            alt = st.number_input(
                "ALT (U/L)",
                min_value=5.0,
                max_value=200.0,
                value=25.0,
                key="input_ALT_Enzyme",
            )
            ast = st.number_input(
                "AST (U/L)",
                min_value=5.0,
                max_value=200.0,
                value=25.0,
                key="input_AST_Enzyme",
            )
            albumin = st.number_input(
                "Albúmina (g/dL)",
                min_value=2.0,
                max_value=6.0,
                value=4.5,
                key="input_Albumin",
            )
        with bio_4:
            potassium = st.number_input(
                "Potasio (mmol/L)",
                min_value=2.0,
                max_value=6.0,
                value=4.0,
                key="input_Potassium",
            )
            sodium = st.number_input(
                "Sodio (mmol/L)",
                min_value=120.0,
                max_value=160.0,
                value=140.0,
                key="input_Sodium",
            )
            ggt = st.number_input(
                "GGT (U/L)",
                min_value=5.0,
                max_value=200.0,
                value=25.0,
                key="input_GGT_Enzyme",
            )

    with habits_tab:
        habits_1, habits_2, habits_3 = st.columns(3)
        with habits_1:
            smoking = st.checkbox(
                "Fumó al menos 100 cigarrillos en su vida (0 = No, 1 = Sí)",
                key="input_Smoking",
            )
        with habits_2:
            physical_activity = st.checkbox(
                "Realiza actividad física recreativa vigorosa (0 = No, 1 = Sí)",
                key="input_PhysicalActivity",
            )
        with habits_3:
            alcohol = st.checkbox(
                "Consumió alcohol durante los últimos 12 meses (0 = No, 1 = Sí)",
                key="input_Alcohol",
            )

    submitted = st.form_submit_button(
        "Ejecutar clasificación académica",
        key="submit_academic_classification",
        use_container_width=True,
    )

if submitted:
    values_by_feature = {
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
        user_input = {
            feature: values_by_feature[feature] for feature in expected_features
        }
        input_frame = UserInputAdapter(expected_features).transform(user_input)
        score = float(model.predict_proba(input_frame)[0])
        classification = int(score >= threshold)

        st.divider()
        with st.container(border=True):
            st.subheader(f"Salida del prototipo: clase {classification}")
            score_column, threshold_column = st.columns(2)
            with score_column:
                st.metric("Score de la clase positiva", f"{score:.6f}")
                st.caption(
                    "El score es una salida del modelo y no una probabilidad "
                    "clínica calibrada."
                )
            with threshold_column:
                st.metric("Umbral operativo", f"{threshold:.2f}")
                st.caption(
                    "Umbral heredado sin validación independiente demostrada."
                )

            st.markdown(
                f"**Regla de decisión:** Clase 1 si score ≥ {threshold:.2f}; "
                "clase 0 en otro caso."
            )
            if classification == 1:
                st.write(
                    "La entrada fue asignada a la clase asociada por el modelo "
                    "con el antecedente autorreportado."
                )
            else:
                st.write(
                    "La entrada no fue asignada a la clase asociada por el "
                    "modelo con el antecedente autorreportado."
                )
            st.info("Esta salida no confirma ni descarta una condición médica.")
    except (KeyError, TypeError, ValueError) as exc:
        st.error(
            "No se pudo validar la entrada. Revise los campos indicados por el "
            f"contrato: {exc}"
        )
    except Exception:
        st.error(
            "La inferencia no pudo completarse. La entrada no fue clasificada; "
            "revise el entorno y el artefacto desplegado."
        )

with st.expander("Cómo funciona el modelo"):
    st.write(
        "Pydantic valida las 27 variables y el adaptador conserva su orden "
        "canónico. El pipeline PyCaret las transforma en 31 características y "
        "un ensemble de 60 árboles XGBoost produce el score de la clase 1. La "
        f"clasificación aplica después el umbral operativo {threshold:.2f}."
    )

with st.expander("Alcance y limitaciones"):
    st.write(
        "El objetivo es un antecedente autorreportado asociado con MCQ160E. El "
        "modelo no predice eventos futuros; el umbral es heredado, no se ha "
        "demostrado calibración ni validación clínica, y el uso previsto es "
        "exclusivamente académico."
    )
