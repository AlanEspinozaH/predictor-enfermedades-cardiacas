# Prototipo académico NHANES para clasificación de antecedente de infarto

Proyecto de aprendizaje automático desarrollado con datos de **NHANES** y dos
componentes diferenciados:

1. Un pipeline serializado de PyCaret para demostrar inferencia en Streamlit.
2. Una implementación educativa, simplificada, de boosting con árboles desde cero.

> **Alcance correcto:** el objetivo histórico es `MCQ160E`, es decir, si una
> persona declaró que alguna vez un profesional le informó que había sufrido un
> ataque cardíaco. El proyecto **no predice un infarto futuro**, no es un sistema
> de diagnóstico y no es un dispositivo médico.

## Estado del repositorio

El código fue saneado para eliminar información personal, rutas locales, logs,
cachés, documentación histórica no vigente y notebooks que contradecían el
pipeline corregido.

El artefacto `models/best_pipeline.pkl` se conserva únicamente como **modelo
académico heredado**. Sus métricas anteriores no se consideran una validación
independiente. El flujo de entrenamiento corregido genera candidatos nuevos sin
reemplazar automáticamente el modelo desplegado.

## Correcciones principales

- Contrato único de 27 variables entre interfaz, validación y configuración.
- `HDL` es obligatorio y `DiastolicBP` ya no forma parte de la entrada.
- Codificación NHANES de sexo: `1 = hombre`, `2 = mujer`.
- Edad mínima de 20 años para la cohorte asociada a `MCQ160E`.
- Rechazo explícito de variables faltantes o inesperadas; no se rellenan con cero.
- Un único artefacto desplegado resuelto por `models/model_manifest.json`.
- Verificación SHA-256 del modelo y de su configuración.
- Compatibilidad controlada con metadatos PyCaret: `HeartDisease` puede retirarse únicamente de `pipeline.feature_names_in_`, nunca del contrato de entrada ni de las características del estimador final.
- Construcción de objetivo: `1 -> positivo`, `2 -> negativo`; `7`, `9` y ausentes
  se excluyen.
- División del test antes de imputar, ajustar o seleccionar umbral.
- Candidatos guardados en `models/candidates/<timestamp>/` sin despliegue automático.
- Auditoría por grupos marcada como exploratoria y no independiente.
- Interfaz corregida para no presentar la salida como riesgo clínico futuro.
- Docker actualizado a una imagen Python 3.10 basada en Debian Bookworm.

## Estructura

```text
.
├── data/
│   ├── 02_intermediate/        # Artefacto heredado y cohortes generadas
│   └── 02_processed/           # Construcción reproducible de datos NHANES
├── docs/                       # Documentación canónica
├── models/
│   ├── best_pipeline.pkl       # Artefacto heredado actualmente desplegado
│   ├── model_config.json       # Contrato serializado de características
│   └── model_manifest.json     # Registro, hashes y umbral del despliegue
├── notebooks/README.md         # Motivo de retiro de notebooks antiguos
├── src/
│   ├── app.py                  # Interfaz Streamlit
│   ├── feature_contract.py     # Fuente canónica del esquema
│   ├── artifact_registry.py    # Resolución e integridad de artefactos
│   ├── data_pipeline.py        # Cohorte, objetivo y particiones
│   ├── train_pycaret.py        # Entrenamiento de candidatos
│   ├── validate_external.py    # Evaluación externa explícita
│   ├── audit_fairness.py       # Diagnóstico exploratorio por sexo
│   └── tree/                   # Árbol y pérdida del modelo educativo
└── tests/                      # Pruebas unitarias y de contrato
```

## Entorno recomendado

- Python 3.10
- Windows 11, Linux o contenedor Docker
- Entorno aislado con Conda o `venv`

### Conda

```powershell
conda create -n heart-ml python=3.10 pip -y
conda activate heart-ml
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

### `venv` en Windows

```powershell
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

## Verificación

Ejecutar desde la raíz:

```powershell
python -m pip check
python -m pytest -q
python -m pytest -q -m integration
```

La marca `integration` carga el archivo real `models/best_pipeline.pkl`, ejecuta `predict()` y `predict_proba()` con exactamente las 27 variables canónicas y abre la aplicación mediante `streamlit.testing.v1.AppTest`.

Con el entorno del usuario:

```powershell
C:\Users\USUARIO\anaconda3\envs\heart-ml\python.exe -m pip check
C:\Users\USUARIO\anaconda3\envs\heart-ml\python.exe -m pytest -q
```

## Ejecutar la aplicación

```powershell
python -m streamlit run src/app.py
```

La aplicación verifica el manifiesto y los hashes antes de cargar el pickle. No
cargue archivos `.pkl` provenientes de fuentes no confiables.

## Reconstruir datos y entrenar un candidato

```powershell
python data/02_processed/carga.py
python src/train_pycaret.py --strategy SMOTE
python src/train_pycaret.py --strategy SCALE_POS_WEIGHT
```

El primer comando crea una cohorte sin imputación estadística y un archivo de
procedencia. Los dos comandos siguientes generan candidatos usando únicamente
información de desarrollo: por defecto **no consultan el test protegido**. Tras
elegir una sola estrategia con criterios fijados previamente, se permite una
única evaluación explícita:

```powershell
python src/train_pycaret.py --strategy ESTRATEGIA_ELEGIDA --evaluate-protected-test
```

Cada ejecución guarda un candidato fechado y nunca modifica automáticamente el
despliegue.

## Validación externa

```powershell
python -m src.validate_external ^
  --data-path ruta\cohorte_externa.parquet ^
  --output-path results\predicciones.csv
```

Una cohorte externa debe incluir exactamente las 27 características. Para
calcular métricas debe incluir además `HeartDisease` con valores binarios `0/1`.

## Docker

```powershell
docker build -t nhanes-heart-prototype .
docker run --rm -p 8501:8501 nhanes-heart-prototype
```

## Documentación

- [Índice documental](docs/README.md)
- [Diseño técnico](docs/Diseno_Tecnico.md)
- [Contrato de variables](docs/auditoria_variables.md)
- [Ficha del modelo](docs/MODEL_CARD.md)
- [Ficha de datos](docs/DATA_CARD.md)
- [Métricas y evaluación](docs/definicion_metricas.md)
- [Plan de trabajo restante](docs/PLAN_TRABAJO.md)
- [Correcciones de auditoría](docs/CORRECCIONES_AUDITORIA.md)
- [Evidencia de verificación](docs/VERIFICACION.md)

## Limitaciones esenciales

- El artefacto desplegado es heredado y no debe usarse para afirmaciones clínicas.
- El umbral `0.20` se conserva por trazabilidad, no por validación independiente.
- El Parquet heredado no debe utilizarse para reentrenar candidatos nuevos.
- La equidad no puede concluirse hasta evaluar un candidato en un test protegido.
- La implementación `XGBoostScratch` es pedagógica y no sustituye XGBoost oficial.

## Licencia

El código se distribuye bajo la licencia MIT incluida en `LICENSE`. La licencia
del código no otorga automáticamente derechos de redistribución sobre datos de
terceros ni sobre artefactos entrenados.
