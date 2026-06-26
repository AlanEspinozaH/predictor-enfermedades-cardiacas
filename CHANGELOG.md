# Changelog

## 2026-06-26 — Validación compatible con metadatos PyCaret

- El validador permite retirar exclusivamente `HeartDisease` de los metadatos externos `pipeline.feature_names_in_`.
- Cualquier otra característica inesperada, duplicada, ausente o desordenada sigue provocando un error de contrato.
- Se añadió una comprobación independiente que rechaza `HeartDisease` dentro de `actual_estimator.feature_names_in_`.
- Streamlit utiliza el mismo cargador validado que las pruebas de integración.
- Se añadió una prueba de integración que carga `models/best_pipeline.pkl`, construye las 27 entradas canónicas y ejecuta `predict()` y `predict_proba()`.
- Las 39 pruebas existentes se conservaron; la suite unitaria ampliada contiene 44 pruebas aprobadas en el entorno de revisión.

## 2026-06-26 — Limpieza y corrección estructural

- Retirados nombres personales, rutas locales, logs, cachés y material histórico.
- Corregido el contrato de entrada y la semántica de la interfaz.
- Añadida inferencia directa y validada mediante `predict_proba`.
- Añadidas métricas compartidas e intervalos Wilson para recall.
- Corregidos scripts de auditoría y validación externa.
- Unificada la implementación educativa `XGBoostScratch`.
- Actualizados Docker, pre-commit, documentación y pruebas.
- La evaluación del test protegido pasó a ser optativa y explícita; los candidatos se comparan sin consultarlo.
- El pipeline finalizado se evalúa antes de guardarlo, de modo que métricas y artefacto correspondan al mismo objeto entrenado.
- Se precisó la semántica NHANES de actividad recreativa vigorosa y consumo de alcohol en los últimos 12 meses.
