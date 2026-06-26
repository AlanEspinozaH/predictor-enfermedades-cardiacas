# Evidencia de verificación

Fecha: 2026-06-26.

## Comprobaciones ejecutadas en el entorno de revisión

- `python -m pytest -q`: **44 pruebas aprobadas y 1 prueba de integración omitida** porque PyCaret no está instalado en el contenedor de revisión.
- `ruff check .`: **sin incidencias**.
- `ruff format --check .`: **25 archivos Python correctamente formateados**.
- Análisis sintáctico con `ast.parse`: **25 archivos Python válidos**.
- Verificación SHA-256 mediante `load_deployed_artifacts(verify_hashes=True)`: **modelo y configuración coinciden con el manifiesto**.
- Escaneo de nombres y rutas históricas: sin coincidencias de los integrantes anteriores ni del usuario local antiguo; solo permanece la ruta de ejemplo `C:\Users\USUARIO\...` solicitada para ejecutar pruebas en el equipo del responsable actual.

## Entorno usado

- Python 3.13.5.
- NumPy 2.3.5.
- pandas 2.2.3.
- scikit-learn 1.8.0.
- Pydantic 2.13.4.
- Ruff 0.15.20.

Este entorno sirve para pruebas unitarias y análisis estático, pero **no sustituye** el entorno objetivo Python 3.10 fijado por el proyecto.

## Prueba de integración del artefacto real

El archivo `tests/test_real_artifact_integration.py` no utiliza mocks. En el entorno objetivo con las dependencias declaradas:

1. carga `models/best_pipeline.pkl` mediante PyCaret;
2. confirma que solo `HeartDisease` se normaliza desde los metadatos externos;
3. verifica que el estimador final contiene 31 características transformadas y no contiene el objetivo;
4. construye un `DataFrame` con exactamente las 27 variables canónicas;
5. ejecuta `predict()` y `predict_proba()`;
6. inicia `src/app.py` mediante `streamlit.testing.v1.AppTest` y exige que no aparezca el error de contrato anterior.

El responsable del proyecto aportó además la siguiente evidencia de una ejecución real previa en su entorno `heart-ml`: 27 columnas, objetivo ausente y ejecución correcta de `predict()` y `predict_proba()`. Esa evidencia motivó y delimita esta corrección, pero no se presenta como una ejecución independiente del contenedor de revisión.

## Comprobaciones no ejecutadas aquí

- La prueba de integración del pickle y Streamlit fue omitida automáticamente porque PyCaret y Streamlit no están instalados en este contenedor Python 3.13.
- Reconstrucción del Parquet, porque PyArrow no estaba instalado y la descarga completa de NHANES no formaba parte de la prueba local.
- Entrenamiento de un candidato nuevo.
- Construcción Docker, porque Docker no estaba disponible.
- `pip check` del entorno `heart-ml` del usuario. El `pip check` del entorno compartido de revisión detectó un conflicto ajeno al proyecto entre MoviePy y Pillow, por lo que no se presenta como evidencia del repositorio.

## Verificación obligatoria en Windows

Ejecutar desde la raíz del proyecto:

```powershell
C:\Users\USUARIO\anaconda3\envs\heart-ml\python.exe -m pip check
C:\Users\USUARIO\anaconda3\envs\heart-ml\python.exe -m pytest -q
C:\Users\USUARIO\anaconda3\envs\heart-ml\python.exe -m pytest -q -m integration
C:\Users\USUARIO\anaconda3\envs\heart-ml\python.exe -m streamlit run src/app.py
pre-commit run --all-files
```

Después, y únicamente cuando Docker Desktop esté disponible:

```powershell
docker build -t nhanes-heart-prototype .
docker run --rm -p 8501:8501 nhanes-heart-prototype
```
