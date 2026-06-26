# Model Card

## Identificación

- ID: `nhanes-heart-disease-pycaret-legacy-v1`.
- Ruta: `models/best_pipeline.pkl`.
- Estado: prototipo académico heredado desplegado.
- Registro: `models/model_manifest.json`.

## Uso previsto

Demostración de un flujo de inferencia reproducible con validación de esquema,
manifiesto y aplicación Streamlit.

## Usos no previstos

- Diagnóstico médico.
- Predicción de infarto futuro.
- Tamizaje clínico real.
- Decisiones de tratamiento, seguros o acceso a servicios.
- Conclusiones de equidad clínica.

## Objetivo histórico

Clase binaria derivada de una respuesta autorreportada sobre antecedente de
ataque cardíaco. No es un desenlace prospectivo.

## Contrato técnico del artefacto heredado

- La entrada de inferencia contiene exactamente 27 variables canónicas.
- `HeartDisease` no pertenece a `MODEL_INPUT_FEATURES`, a la interfaz ni al `DataFrame` de predicción.
- PyCaret conserva `HeartDisease` en ciertos metadatos externos del pipeline; el cargador retira únicamente ese nombre antes de validar el orden de entrada.
- El estimador final utiliza 31 características transformadas por la codificación y debe excluir explícitamente `HeartDisease`.
- Cualquier característica externa inesperada distinta del objetivo invalida el artefacto.

## Limitaciones conocidas

- El artefacto se entrenó antes de corregir completamente la construcción de la
  cohorte y la estrategia de evaluación.
- Las métricas históricas usaron datos no independientes.
- El umbral heredado no cuenta con validación independiente.
- El dataset asociado conserva decisiones históricas de preparación.
- No existe evidencia suficiente de calibración, transporte o equidad.

## Despliegue de candidatos

Un candidato puede reemplazar el artefacto solo después de:

1. verificar procedencia de datos;
2. revisar el manifiesto del candidato;
3. inspeccionar métricas en test protegido;
4. comprobar calibración y subgrupos;
5. actualizar explícitamente el manifiesto desplegado y sus hashes;
6. ejecutar toda la suite de pruebas.
