# CardioHistory ML

**Clasificación explicable de antecedente autorreportado de infarto con datos NHANES**

CardioHistory ML es un prototipo académico de aprendizaje automático. Su
objetivo corresponde a `MCQ160E`: si una persona declaró que un profesional de
salud le había informado alguna vez que sufrió un infarto.

> La aplicación no diagnostica una condición actual, no predice un infarto
> futuro, no estima riesgo cardiovascular prospectivo y no es un dispositivo
> médico.

Proyecto académico desarrollado por el equipo del curso de Inteligencia
Artificial.

## Qué hace

- Recibe 27 variables demográficas, antropométricas, bioquímicas y de estilo de
  vida mediante Streamlit.
- Valida tipos, dominios y rangos con Pydantic.
- Ordena la entrada según un contrato canónico.
- Verifica la integridad del modelo y su configuración antes de deserializar.
- Obtiene el score de la clase positiva del pipeline PyCaret/XGBoost.
- Aplica el umbral operativo `0.20` declarado en el manifiesto.
- Presenta una clasificación académica con advertencias de alcance.

No ofrece diagnóstico, recomendación terapéutica, tamizaje clínico ni una
medición formal de incertidumbre.

## Estado actual del modelo

| Elemento | Estado verificado |
|---|---|
| Identificador interno | `nhanes-heart-disease-pycaret-legacy-v1` |
| Estado del manifiesto | `legacy_prototype_deployed` |
| Formato | Pipeline PyCaret serializado |
| Estimador final | `xgboost.sklearn.XGBClassifier` |
| Entrada externa | 27 variables canónicas |
| Entrada transformada | 31 características |
| Clase positiva | `1` |
| Umbral operativo | `0.20`, definido en el manifiesto y sin validación independiente |

El artefacto desplegado corresponde a un prototipo académico desplegado. La inferencia y la integridad
técnica se han verificado, pero no existen métricas históricas reproducibles
suficientes para presentarlo como superior a otros modelos ni como validado para
uso clínico.

## Arquitectura resumida

```text
Streamlit
  → InputData (Pydantic)
  → UserInputAdapter (27 variables en orden canónico)
  → pipeline PyCaret (preprocesamiento a 31 características)
  → XGBClassifier
  → score de la clase 1
  → umbral 0.20
  → clasificación académica
```

`models/model_manifest.json` identifica el pipeline, su configuración, sus
hashes SHA-256 y el umbral. El flujo no busca automáticamente el pickle más
reciente.

## Inicio rápido en Windows 11

Requiere Conda y Python 3.10. Desde la raíz del repositorio:

```powershell
conda create -n heart-ml python=3.10 -y
conda activate heart-ml
python --version
python -m pip install -r requirements-dev.txt
python -m pip check
```

La versión debe comenzar con `Python 3.10`. `requirements-dev.txt` incluye las
dependencias de ejecución declaradas en `requirements.txt`.

### Ejecutar la aplicación

```powershell
python -m streamlit run src/app.py
```

Streamlit sirve la aplicación localmente, normalmente en
`http://localhost:8501`.

### Ejecutar el smoke test

```powershell
python scripts/smoke_test.py
```

El smoke test verifica hashes, carga validada, contrato de 27 variables,
transformación esperada a 31 características, score de la clase positiva y
aplicación del umbral del manifiesto.

### Ejecutar las pruebas

```powershell
python -m pytest -q -m "not integration"
python -m pytest -q -m integration
```

La línea base comprobada con Python 3.10.20 es de 96 pruebas no integradas y 4
pruebas de integración aprobadas. Estas comprobaciones demuestran funcionamiento
técnico, no desempeño predictivo ni validez clínica.

La guía completa está en [Instalación](docs/INSTALLATION.md).

### Validación técnica trazable

El modo desplegado resuelve y verifica `models/model_manifest.json`:

```powershell
python -m src.validate_external `
  --data-path ruta\cohorte.csv `
  --output-path results\validation.json `
  --predictions-path results\predictions.csv
```

Un candidato requiere su manifiesto v1 completo; no se aceptan pickles por una
ruta arbitraria:

```powershell
python -m src.validate_external `
  --candidate-manifest models\candidates\<run_id>\candidate_manifest.json `
  --data-path ruta\cohorte.csv `
  --output-path results\validation.json `
  --predictions-path results\predictions.csv
```

No existe fallback silencioso a `0.50`: el umbral procede del manifiesto o de
un `--threshold` explícito que queda registrado. Sin un sidecar de procedencia
verificable, el resultado se clasifica como `external_unverified`. Incluso con
procedencia declarada como independiente, el programa registra que no realizó
una auditoría independiente. Estas salidas son evidencia técnica, no validación
clínica.

## Artefactos y datos

Para inferencia se requieren el manifiesto y los artefactos de modelo y
configuración declarados en él:

- `models/model_manifest.json`;
- `models/best_pipeline.pkl`;
- `models/model_config.json`.

Solo deben deserializarse pickles de origen confiable. La verificación SHA-256
detecta cambios, pero no neutraliza código malicioso dentro de un pickle externo.

Los archivos Parquet se utilizan en análisis, validación o entrenamiento. La
aplicación Streamlit no necesita el Parquet para abrirse ni realizar inferencia.
El Parquet histórico asociado al modelo se conserva por trazabilidad y no debe
tratarse como una cohorte corregida para reentrenamiento.

## Estructura

```text
.
├── data/                       # Datos históricos y flujos de preparación
├── docs/
│   ├── ARCHITECTURE.md         # Arquitectura efectiva
│   ├── INSTALLATION.md         # Instalación y operación
│   ├── MODEL_CARD.md          # Evidencia y límites del artefacto
│   └── MODEL_EXPLANATION.md   # Fundamento de XGBoost
├── models/                     # Manifiesto, configuración y pipeline
├── scripts/
│   └── smoke_test.py          # Verificación mínima de extremo a extremo
├── src/                        # Aplicación, adaptadores y entrenamiento
├── tests/                      # Pruebas unitarias y de integración
├── requirements.txt
└── requirements-dev.txt
```

## Documentación

- [Instalación y ejecución](docs/INSTALLATION.md)
- [Arquitectura técnica](docs/ARCHITECTURE.md)
- [Ficha del modelo desplegado](docs/MODEL_CARD.md)
- [Explicación técnica y matemática](docs/MODEL_EXPLANATION.md)

## Limitaciones científicas

- El objetivo es un antecedente autorreportado y puede contener error de
  recuerdo o clasificación.
- NHANES es observacional y transversal; este prototipo no modela tiempo hasta
  un evento.
- El entrenamiento histórico no es completamente reproducible con la evidencia
  actualmente conservada.
- El umbral `0.20` no tiene validación independiente demostrada.
- No se ha demostrado calibración, transporte a otras poblaciones ni equidad por
  subgrupos.
- El desbalance de clases y `scale_pos_weight=38` exigen evaluar métricas por
  clase y calibración antes de interpretar scores.
- Docker dispone de un procedimiento documentado, pero aún no se presenta como
  validado en Windows 11.

## Licencia

El código se distribuye bajo la licencia MIT incluida en `LICENSE`. Esto no
otorga automáticamente derechos de redistribución sobre datos o artefactos de
terceros.
