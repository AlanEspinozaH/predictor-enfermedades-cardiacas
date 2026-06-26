# Implementación educativa desde cero

`src/model.py` implementa un clasificador de boosting de árboles inspirado en
XGBoost para estudiar:

- margen logístico;
- gradiente y hessiano de log-loss;
- peso óptimo de hojas;
- ganancia de una partición;
- regularización L2 (`lambda_`);
- penalización por partición (`gamma`);
- tasa de aprendizaje;
- ensamble aditivo de árboles.

## Mejoras aplicadas

- `fit()` reinicia los árboles anteriores.
- Se valida forma, finitud y clases de `X` e `y`.
- El margen base se calcula desde la prevalencia de entrenamiento.
- Se impide predecir antes de ajustar.
- La sigmoide se estabiliza numéricamente.
- `src/tree/xgboost_scratch.py` es un alias compatible, no una segunda
  implementación divergente.

## Alcance

No implementa histogramas, cuantiles aproximados, aprendizaje distribuido,
tratamiento nativo de faltantes, poda avanzada, muestreo de filas/columnas ni
todas las regularizaciones de XGBoost oficial. Su propósito es didáctico.
