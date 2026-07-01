# Model Card de CardioHistory ML

**Clasificación explicable de antecedente autorreportado de infarto con datos
NHANES**

## 1. Identificación

| Campo | Valor verificado |
|---|---|
| Nombre visible | CardioHistory ML |
| `model_id` | `nhanes-heart-disease-pycaret-legacy-v1` |
| `status` interno del manifiesto | `legacy_prototype_deployed` |
| Ruta | `models/best_pipeline.pkl` |
| Formato | Pipeline PyCaret serializado (`pycaret_pickle`) |
| Estimador final | `xgboost.sklearn.XGBClassifier` |
| Tarea | Clasificación binaria |
| Registro | `models/model_manifest.json` |
| Entorno comprobado | Python 3.10.20 |

El artefacto se considera un prototipo académico desplegado. Su estado de
despliegue significa que es el artefacto seleccionado por el manifiesto, no que
esté aprobado para uso clínico.

## 2. Objetivo

La clase positiva `1` representa un antecedente autorreportado de infarto
asociado con `MCQ160E`: la persona informó que un profesional de salud le había
dicho alguna vez que sufrió un ataque cardíaco. La respuesta explícita `2` se
codifica como clase `0`; respuestas desconocidas, rechazadas o ausentes no deben
formar parte del objetivo corregido.

El modelo no determina una enfermedad actual, no estima un horizonte temporal,
no predice un infarto futuro y no constituye un dispositivo médico.

## 3. Datos y población

El proyecto se basa en NHANES, que combina demografía, cuestionarios, examen
físico y laboratorio. La cohorte corregida prevista incluye personas de 20 años
o más con respuesta explícita a `MCQ160E`. `SEQN` sirve para vincular componentes
y controlar duplicados, pero no es un predictor.

El Parquet histórico declarado en el manifiesto se conserva por trazabilidad del
modelo desplegado. La construcción histórica no está documentada de forma
suficiente para reproducir exactamente el entrenamiento, y ese Parquet no debe
usarse para reentrenar un candidato corregido.

NHANES es observacional y transversal. Parte de la información es
autorreportada; la disponibilidad de laboratorios cambia entre ciclos; la unión
de componentes puede modificar la composición de la cohorte; y este prototipo no
aplica pesos complejos de encuesta para producir estimaciones nacionales.

## 4. Entradas

La entrada externa tiene exactamente 27 variables. El orden canónico está en
`src/feature_contract.py` y se refleja en `models/model_config.json`. Pydantic
controla tipos, rangos y dominios; los rangos son controles del prototipo, no
intervalos clínicos.

| Grupo | Variables |
|---|---|
| Demografía y contexto | `Age`, `IncomeRatio`, `Sex`, `Race`, `Education` |
| Antropometría y signos | `SystolicBP`, `BMI`, `WaistCircumference`, `Height` |
| Perfil bioquímico | `TotalCholesterol`, `Triglycerides`, `LDL`, `HDL`, `HbA1c`, `Glucose`, `Creatinine`, `UricAcid`, `ALT_Enzyme`, `Albumin`, `Potassium`, `Sodium`, `GGT_Enzyme`, `AST_Enzyme` |
| Estilo de vida y acceso | `Smoking`, `PhysicalActivity`, `HealthInsurance`, `Alcohol` |

Las 20 variables numéricas aceptan valores enteros o reales según el campo.
Las 7 categóricas restringen sus códigos con literales; por ejemplo, `Sex` usa
la codificación NHANES `1=hombre`, `2=mujer`. `Glucose` significa **Glucosa
sérica del perfil bioquímico NHANES (`LBXSGL`)**.

`HeartDisease`, `SEQN` y `DiastolicBP` no pertenecen a la entrada del modelo.

## 5. Preprocesamiento desplegado

El pipeline serializado realiza:

1. imputación numérica por media;
2. imputación categórica por moda;
3. codificación ordinal de `Sex`;
4. one-hot encoding de `Race`;
5. un paso residual sin columnas efectivas;
6. `RobustScaler`;
7. inferencia con `XGBClassifier`.

La codificación expande las 27 variables externas a 31 características. El
objetivo no aparece entre esas 31 características. Aunque el escalamiento no
suele ser necesario para árboles, `RobustScaler` pertenece al pipeline
serializado y debe conservarse para reproducir la inferencia del artefacto.

## 6. Configuración principal

Los siguientes valores describen el estimador desplegado; no se presentan como
óptimos:

| Parámetro | Valor |
|---|---:|
| `booster` | `gbtree` |
| `objective` | `binary:logistic` |
| `eval_metric` efectivo | `logloss` |
| `n_estimators` | 60 |
| `max_depth` | 6 |
| `learning_rate` | 0.4 |
| `min_child_weight` | 1 |
| `subsample` | 0.7 |
| `colsample_bytree` | 0.7 |
| `reg_alpha` | 0.001 |
| `reg_lambda` | 0.7 |
| `scale_pos_weight` | 38 |
| `random_state` | 42 |
| `n_jobs` | 1 |

## 7. Complejidad observada

La inspección técnica realizada en la etapa de caracterización registró:

| Medida | Valor |
|---|---:|
| Rondas | 60 |
| Árboles | 60 |
| Nodos | 5 168 |
| Hojas | 2 614 |
| Profundidad máxima observada | 6 |

Estas cantidades describen el artefacto; no son métricas de calidad predictiva.

## 8. Salida y umbral

El estimador declara `classes_ = [0, 1]`. `PyCaretAdapter` identifica la columna
de `predict_proba()` correspondiente a la clase `1` y devuelve ese score. El
score debe ser numérico, finito y pertenecer a `[0, 1]`.

La clasificación usa:

```text
clase = 1 si score >= 0.20; en otro caso, clase = 0
```

`0.20` es el umbral operativo definido en el manifiesto. Es menor que el umbral
convencional `0.50`, pero no existe evidencia independiente suficiente para
justificarlo como óptimo. El score no debe interpretarse como calibrado ni como
riesgo clínico.

## 9. Evidencia disponible

En el entorno Windows/Conda con Python 3.10.20 se ha comprobado:

- integridad SHA-256 del pipeline y la configuración;
- carga validada del artefacto mediante PyCaret;
- entrada externa de 27 variables en orden canónico;
- 31 características esperadas por el estimador sin el objetivo;
- inferencia funcional y score válido para una fila sintética;
- 123 pruebas no integradas aprobadas;
- 18 pruebas de integración aprobadas;
- smoke test aprobado.

Esta evidencia demuestra integridad y funcionamiento técnico actual. No es una
validación predictiva, independiente o clínica.

## 10. Métricas y comparación

**No hay métricas históricas reproducibles suficientes para justificar que el
artefacto desplegado sea el mejor modelo.**

El repositorio contiene funciones para calcular matriz de confusión, precision,
recall, especificidad, F1, ROC-AUC, PR-AUC y Brier score cuando existen etiquetas
y scores válidos. Su existencia en el código no demuestra ningún valor para el
artefacto desplegado. No se reportan aquí valores de desempeño porque no hay una
evaluación histórica reproducible e independiente que los sustente.

## 11. Limitaciones

- El entrenamiento histórico no es completamente reproducible con la evidencia
  actual.
- El umbral `0.20` no se ha validado independientemente.
- El objetivo es autorreportado y puede contener errores de recuerdo o
  clasificación.
- La selección de participantes y disponibilidad de laboratorios pueden inducir
  sesgo poblacional.
- Existe desbalance de clases y el modelo usa `scale_pos_weight=38`.
- No se ha demostrado calibración de los scores.
- No hay evaluación independiente de transporte entre ciclos o poblaciones.
- El informe histórico de equidad fue invalidado por codificación incorrecta de
  `Sex`, datos no independientes y una estadística centinela no válida; por tanto,
  no existe una conclusión de equidad vigente.
- La deserialización puede presentar incompatibilidades entre versiones de
  Python, PyCaret, scikit-learn y otras dependencias.
- El artefacto no puede utilizarse clínicamente.

## 12. Uso previsto y uso no permitido

### Uso previsto

- demostración académica de un pipeline de clasificación tabular;
- estudio de validación de entradas, preprocesamiento e integridad de artefactos;
- explicación de gradient boosting y decisiones de umbral;
- base controlada para diseñar futuros candidatos reproducibles.

### Uso no permitido

- diagnóstico, tamizaje o tratamiento;
- estimación individual de riesgo prospectivo;
- decisiones de seguros, empleo o acceso a servicios;
- sustitución del criterio de profesionales de salud;
- afirmaciones poblacionales sin diseño de encuesta adecuado;
- conclusiones de equidad o superioridad frente a otros modelos.

Antes de considerar otro uso se necesitarían datos reproducibles, evaluación
protegida, comparación de candidatos, calibración, análisis de subgrupos e
investigación independiente apropiada al contexto.
