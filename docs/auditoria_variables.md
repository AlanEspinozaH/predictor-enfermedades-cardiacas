# Contrato de variables

## Objetivo

`HeartDisease` representa la codificación de `MCQ160E`:

- `1`: respuesta explícita “Sí, alguna vez le dijeron que tuvo un ataque cardíaco”.
- `0`: respuesta explícita “No”.
- `7`, `9`, ausencia o no elegibilidad: se excluyen del objetivo.

El nombre histórico `HeartDisease` se conserva por compatibilidad, aunque
`HeartAttackHistory` sería semánticamente más preciso.

## Variables numéricas

| Variable | Unidad/interpretación | Rango aceptado por la UI |
|---|---|---:|
| `Age` | años | 20–100 |
| `IncomeRatio` | razón ingreso/umbral de pobreza | 0–5 |
| `SystolicBP` | mmHg | 80–220 |
| `BMI` | kg/m² | 12–60 |
| `WaistCircumference` | cm | 50–180 |
| `Height` | cm | 130–220 |
| `TotalCholesterol` | mg/dL | 100–400 |
| `Triglycerides` | mg/dL | 30–600 |
| `LDL` | mg/dL | 30–300 |
| `HDL` | mg/dL | 10–150 |
| `HbA1c` | % | 4–15 |
| `Glucose` | mg/dL | 50–300 |
| `Creatinine` | mg/dL | 0.4–5 |
| `UricAcid` | mg/dL | 2–12 |
| `ALT_Enzyme` | U/L | 5–200 |
| `Albumin` | g/dL | 2–6 |
| `Potassium` | mmol/L | 2–6 |
| `Sodium` | mmol/L | 120–160 |
| `GGT_Enzyme` | U/L | 5–200 |
| `AST_Enzyme` | U/L | 5–200 |

Los rangos de interfaz son controles de entrada del prototipo; no constituyen
intervalos diagnósticos ni reglas clínicas.

## Variables categóricas

| Variable | Codificación |
|---|---|
| `Sex` | NHANES: `1=hombre`, `2=mujer` |
| `Race` | `1` mexicano-estadounidense; `2` otro hispano; `3` blanco no hispano; `4` negro no hispano; `5` otra categoría |
| `Education` | categorías NHANES `1` a `5` |
| `Smoking` | `0=no`, `1=sí` |
| `PhysicalActivity` | `0=no`, `1=sí`; actividad recreativa vigorosa (`PAQ650`) |
| `HealthInsurance` | `0=no`, `1=sí` |
| `Alcohol` | `0=no`, `1=sí`; consumo durante los últimos 12 meses, armonizado por ciclo |

## Variables excluidas

- `SEQN`: identificador, nunca predictor.
- `DiastolicBP`: no pertenece al artefacto desplegado.
- Variables BRFSS heredadas: fuera del contrato NHANES vigente.

## Tratamiento de faltantes

El conjunto previo al entrenamiento conserva faltantes. Los imputadores se
ajustan exclusivamente dentro del pipeline con datos de desarrollo. El objetivo
no se imputa.
