# Plan de trabajo restante, priorizado

## Fase 1 — Validación local reproducible

1. Ejecutar `pip check` y toda la suite con Python 3.10 de `heart-ml`.
2. Ejecutar `pre-commit run --all-files`.
3. Construir la imagen Docker y comprobar el endpoint de salud.
4. Guardar las salidas como evidencia de entrega, no como archivos versionados.

**Cierre:** dependencias consistentes, pruebas verdes y aplicación iniciando.

## Fase 2 — Reconstrucción de datos

1. Ejecutar `data/02_processed/carga.py`.
2. Revisar procedencia, ciclos, filas, positivos y faltantes.
3. Verificar que solo `MCQ160E` define el objetivo.
4. Verificar edad mínima, dominios categóricos y ausencia de duplicados.

**Cierre:** Parquet corregido y `.provenance.json` aceptados por el pipeline.

## Fase 3 — Entrenamiento de candidato

1. Ejecutar `src/train_pycaret.py --strategy SMOTE`.
2. Ejecutar `src/train_pycaret.py --strategy SCALE_POS_WEIGHT`.
3. Revisar manifiestos y comparar candidatos solo con evidencia de desarrollo.
4. Elegir una única estrategia y congelar previamente la regla de aceptación.

**Cierre:** candidato fechado, reproducible y no desplegado.

## Fase 4 — Evaluación independiente

1. Ejecutar una sola vez la estrategia elegida con `--evaluate-protected-test`.
2. Reportar matriz, recall, precisión, especificidad, F1, ROC-AUC, PR-AUC y Brier.
3. Añadir curva de calibración e intervalos de confianza.
4. Auditar subgrupos con tamaños y positivos suficientes.
5. Documentar fallos y evitar lenguaje clínico no sustentado.

**Cierre:** decisión explícita de aceptar o rechazar el candidato.

## Fase 5 — Despliegue controlado

1. Copiar el candidato aprobado a una ruta estable.
2. Actualizar `model_manifest.json` con hashes, umbral y métricas.
3. Actualizar `MODEL_CARD.md` y `DATA_CARD.md`.
4. Ejecutar nuevamente pruebas, Docker e inferencia de humo.

## Estrategia de uso eficiente de tokens

Trabajar por unidades cerradas y verificables:

1. Un archivo o subsistema por mensaje.
2. Enviar únicamente el diff, error o fragmento relevante, no todo el repositorio.
3. Ejecutar localmente comandos cortos y devolver solo la salida completa del fallo.
4. No solicitar reescrituras generales mientras exista un error bloqueante.
5. Priorizar en este orden: pruebas, contrato, datos, entrenamiento, evaluación,
   documentación y mejoras opcionales.
6. Mantener una lista de pendientes en este documento para evitar reanalizar lo ya cerrado.
