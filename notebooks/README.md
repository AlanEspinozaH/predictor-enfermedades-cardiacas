# Notebooks retirados de la rama activa

Los notebooks heredados fueron eliminados de la entrega corregida porque mezclaban
flujos incompatibles con el código actual: imputación antes de separar el conjunto
de prueba, selección variable entre `MCQ160E` y `MCQ160F`, evaluación sobre datos
conocidos por el modelo y rutas locales de un equipo anterior.

El flujo reproducible vigente está implementado en:

1. `data/02_processed/carga.py`: descarga y armonización de ciclos NHANES.
2. `src/data_ingestion.py`: construcción del cohort elegible sin imputación.
3. `src/train_pycaret.py`: separación previa del test, entrenamiento y creación de candidatos.
4. `src/validate_external.py`: validación sobre un archivo externo explícito.
5. `src/audit_fairness.py`: auditoría exploratoria claramente marcada como no independiente.

Los notebooks antiguos deben consultarse únicamente desde el historial de Git, no
reincorporarse como documentación vigente.
