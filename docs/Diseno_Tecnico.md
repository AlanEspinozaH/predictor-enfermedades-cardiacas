# Diseño técnico vigente

## 1. Objetivo del sistema

El repositorio implementa un prototipo académico para clasificar la respuesta
binaria asociada a `MCQ160E` en NHANES. No modela tiempo hasta evento ni riesgo
prospectivo.

## 2. Fuentes canónicas

- Esquema ejecutable: `src/feature_contract.py`.
- Metadatos del esquema: `models/model_config.json`.
- Artefacto desplegado y hashes: `models/model_manifest.json`.
- Preparación científica: `src/data_pipeline.py`.

La aplicación, las pruebas, el entrenamiento y la validación deben respetar ese
contrato. Cualquier discrepancia produce un error explícito.

## 3. Flujo de inferencia

```text
Formulario Streamlit
        │
        ▼
InputData (Pydantic, extra="forbid")
        │
        ▼
UserInputAdapter: orden exacto de 27 columnas
        │
        ▼
PyCaretAdapter: predict_proba() del pipeline
        │
        ▼
Umbral declarado en model_manifest.json
        │
        ▼
Clasificación académica con advertencias de alcance
```

No se crean variables faltantes, no se rellenan con cero y no se acepta
`DiastolicBP`.

## 4. Flujo de datos y entrenamiento

```text
Descarga de componentes NHANES
        │
        ▼
Unión por SEQN y ciclo
        │
        ▼
MCQ160E: 1→1, 2→0, otros→excluir
        │
        ▼
Filtro edad >= 20; predictores con faltantes conservados
        │
        ▼
Test temporal del último ciclo, si es viable
(o split estratificado reproducible)
        │
        ▼
PyCaret sobre desarrollo: imputación dentro del pipeline
        │
        ▼
Selección de modelo y umbral sin utilizar el test
        │
        ▼
Evaluación única en test protegido
        │
        ▼
models/candidates/<timestamp>/
```

## 5. Gestión de artefactos

El despliegue no busca automáticamente el `.pkl` más reciente. El manifiesto
identifica:

- ruta y hash SHA-256 del pipeline;
- ruta, versión y hash del esquema;
- umbral;
- datos asociados al artefacto heredado;
- limitaciones.

Los candidatos no se despliegan hasta revisar procedencia, métricas, calibración
y desempeño por subgrupos.

## 6. Modelo educativo

`src/model.py` contiene una implementación simplificada de boosting de árboles.
Se reinicia en cada `fit`, valida entradas, usa margen base según prevalencia y
aplica gradiente/hessiano de log-loss. No implementa todas las optimizaciones,
regularizaciones ni estrategias para valores faltantes de XGBoost oficial.

## 7. Seguridad y reproducibilidad

- Los pickles solo deben cargarse si su origen es confiable.
- Los hashes detectan cambios accidentales, pero no vuelven seguro un pickle malicioso.
- Las rutas se resuelven desde la raíz del repositorio.
- Cachés, logs, salidas temporales y candidatos no revisados no se versionan.
