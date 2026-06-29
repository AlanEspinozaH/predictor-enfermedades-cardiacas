# Instalación y ejecución de CardioHistory ML

Esta guía cubre el entorno principal del proyecto: Windows 11, PowerShell,
Conda y Python 3.10. CardioHistory ML es un prototipo académico; su salida no es
un diagnóstico ni una recomendación clínica.

## 1. Requisitos

- Windows 11 y PowerShell.
- Git para obtener el repositorio.
- Miniconda o Anaconda.
- Python 3.10 dentro del entorno `heart-ml`.
- Un navegador moderno para Streamlit.

Comprueba las herramientas:

```powershell
git --version
conda --version
```

Si PowerShell no reconoce Conda, abre Anaconda Prompt o ejecuta `conda init
powershell`, cierra la terminal y vuelve a abrirla.

## 2. Obtener el proyecto

```powershell
git clone <URL_DEL_REPOSITORIO>
cd <NOMBRE_DE_LA_CARPETA_CLONADA>
```

Todos los comandos siguientes se ejecutan desde la raíz. Confírmala con:

```powershell
Test-Path .\pyproject.toml
Test-Path .\src\app.py
Test-Path .\models\model_manifest.json
```

Los tres resultados deben ser `True`.

## 3. Crear o activar el entorno Conda

Si el entorno no existe, créalo una sola vez:

```powershell
conda create -n heart-ml python=3.10 -y
```

Al iniciar una sesión de trabajo, activa el entorno y confirma el intérprete
antes de ejecutar cualquier comando Python:

```powershell
conda activate heart-ml
python --version
where.exe python
```

La versión debe comenzar con `Python 3.10` y la ruta debe pertenecer a
`envs\heart-ml`.

## 4. Instalar dependencias

Para ejecutar Streamlit y realizar inferencia:

```powershell
python -m pip install -r requirements.txt
python -m pip check
```

Para desarrollo, análisis y pruebas instala en su lugar el conjunto ampliado:

```powershell
python -m pip install -r requirements-dev.txt
python -m pip check
```

`requirements-dev.txt` incluye `requirements.txt`. No se necesitan archivos de
requisitos específicos de Windows.

## 5. Artefactos de inferencia y datos

`models/model_manifest.json` es el registro canónico. En la versión actual
declara el pipeline y la configuración siguientes:

```powershell
Test-Path .\models\model_manifest.json
Test-Path .\models\best_pipeline.pkl
Test-Path .\models\model_config.json
```

Los tres resultados deben ser `True`. La aplicación resuelve esas rutas desde
el manifiesto y verifica los hashes SHA-256 del modelo y de la configuración
antes de deserializar. No cargues pickles obtenidos de fuentes no confiables.

Los Parquet pertenecen a flujos de análisis, validación o entrenamiento. No son
necesarios para abrir Streamlit ni ejecutar inferencia. En particular,
`data/02_intermediate/process_data.parquet` es un artefacto histórico de
trazabilidad, no una cohorte corregida para reentrenar.

La variable interna `Glucose` corresponde a **Glucosa sérica del perfil
bioquímico NHANES (`LBXSGL`)**.

## 6. Verificación rápida

Con el entorno activo:

```powershell
python scripts/smoke_test.py
```

El comando verifica la integridad de los artefactos, carga el pipeline mediante
el registro canónico, valida una entrada sintética de 27 variables, comprueba las
31 características esperadas por el estimador y ejecuta la clasificación con el
umbral del manifiesto. No usa el Parquet ni datos de una persona real.

Una ejecución correcta termina con:

```text
SMOKE TEST PASSED
```

## 7. Ejecutar Streamlit

```powershell
python -m streamlit run src/app.py
```

La URL local predeterminada es `http://localhost:8501`. Para usar otro puerto:

```powershell
python -m streamlit run src/app.py --server.port 8502
```

La interfaz debe mostrar las 27 entradas, incluir `HDL`, excluir
`DiastolicBP` y describir el resultado como clasificación académica de
antecedente autorreportado de infarto.

## 8. Pruebas

Las pruebas no integradas no deserializan el artefacto real:

```powershell
python -m pytest -q -m "not integration"
```

Las pruebas de integración cargan el pipeline real y ejercitan la aplicación:

```powershell
python -m pytest -q -m integration
```

La línea base verificada con Python 3.10.20 es:

```text
44 passed, 3 deselected
3 passed, 44 deselected
```

Estos resultados verifican compatibilidad y comportamiento técnico. No son
métricas de desempeño predictivo ni evidencia de validación clínica.

## 9. Docker como procedimiento no validado

El repositorio incluye `Dockerfile` con Python 3.10 y un `HEALTHCHECK`. El
procedimiento documentado es:

```powershell
docker build -t cardiohistory-ml .
docker run --rm -p 8501:8501 cardiohistory-ml
```

Después se consultaría `http://localhost:8501/_stcore/health`. Este procedimiento
aún no se presenta como validado mediante una construcción y ejecución
satisfactorias en Windows 11.

## 10. Problemas frecuentes

### `conda` no se reconoce

Usa Anaconda Prompt o ejecuta `conda init powershell` y reinicia PowerShell.

### Se está usando otra versión de Python

```powershell
conda activate heart-ml
python --version
where.exe python
```

No cargues el pickle con Python 3.13. El entorno soportado para el artefacto
actual es Python 3.10.

### `ModuleNotFoundError`

Confirma el entorno activo y repite la instalación de
`requirements-dev.txt`. No instales paquetes individualmente para ocultar una
instalación incompleta.

### Falta un artefacto o falla un hash

Repite las tres comprobaciones de la sección 5. Si el repositorio usa Git LFS,
ejecuta `git lfs pull`. No reemplaces el modelo ni edites el manifiesto para
evitar el control de integridad.

### Advertencia de compatibilidad al deserializar

PyCaret puede informar diferencias entre dependencias actuales y las usadas al
serializar. Registra la advertencia y ejecuta el smoke test y las pruebas de
integración. Una ejecución correcta no convierte el artefacto en portable a
cualquier versión.

### El puerto 8501 está ocupado

Usa el comando con `--server.port 8502` de la sección 7.

### Error de contrato de características

No añadas `HeartDisease`, `DiastolicBP` ni columnas de relleno a la entrada.
Verifica que el repositorio esté completo y ejecuta las pruebas de integración.

## 11. Ejecución con el intérprete absoluto

Cuando no sea posible activar Conda, usa explícitamente el Python del entorno:

```powershell
$python = "$env:USERPROFILE\anaconda3\envs\heart-ml\python.exe"
& $python --version
& $python scripts/smoke_test.py
& $python -m streamlit run src/app.py
```
