# Instalación en Windows 11

Esta guía permite instalar, verificar y ejecutar el proyecto en un equipo nuevo con Windows 11.

> **Alcance:** este repositorio es un prototipo académico. No constituye un sistema de diagnóstico ni una herramienta clínica validada.

---

## 1. Requisitos previos

Instala estas herramientas antes de continuar:

1. **Git para Windows**  
   Se utiliza para clonar el repositorio.

2. **Miniconda o Anaconda**  
   Se recomienda Miniconda por ser más ligera.

3. **PowerShell**  
   Windows 11 ya lo incluye.

4. **Navegador web moderno**  
   Chrome, Edge o Firefox.

### Comprobar Git

Abre PowerShell y ejecuta:

```powershell
git --version
```

Debe mostrar una versión de Git. Si el comando no existe, instala Git para Windows y vuelve a abrir PowerShell.

### Comprobar Conda

```powershell
conda --version
```

Si `conda` no se reconoce, abre **Anaconda Prompt** o ejecuta:

```powershell
conda init powershell
```

Cierra y abre PowerShell después de inicializarlo.

---

## 2. Obtener el proyecto

### Opción A — Clonar desde GitHub

En GitHub, abre el botón **Code**, copia la URL HTTPS y ejecuta:

```powershell
cd C:\
New-Item -ItemType Directory -Force C:\dev | Out-Null
cd C:\dev

git clone <URL_DEL_REPOSITORIO>
cd <NOMBRE_DE_LA_CARPETA_CLONADA>
```

Ejemplo:

```powershell
git clone https://github.com/usuario/Heart-Disease-Prediction-ML.git
cd Heart-Disease-Prediction-ML
```

### Opción B — Descargar el ZIP

1. En GitHub, selecciona **Code > Download ZIP**.
2. Descomprime el archivo en una ruta corta, por ejemplo:

```text
C:\dev\Heart-Disease-Prediction-ML
```

3. Entra en la carpeta desde PowerShell:

```powershell
cd "C:\dev\Heart-Disease-Prediction-ML"
```

> No ejecutes los comandos desde una carpeta exterior ni desde una copia anidada del repositorio.

---

## 3. Confirmar que estás en la raíz correcta

Ejecuta:

```powershell
Get-Location

Test-Path ".\pyproject.toml"
Test-Path ".\src\app.py"
Test-Path ".\tests"
Test-Path ".\models\best_pipeline.pkl"
```

Los cuatro comandos `Test-Path` deben devolver:

```text
True
```

La raíz del proyecto debe contener, como mínimo:

```text
.streamlit\
data\
docs\
models\
src\
tests\
pyproject.toml
requirements.txt
requirements-dev.txt
README.md
```

---

## 4. Crear el entorno Conda

El proyecto debe ejecutarse con **Python 3.10.x** por compatibilidad con PyCaret y con el modelo serializado.

```powershell
conda create -n heart-ml python=3.10 -y
conda activate heart-ml
```

Comprueba el entorno:

```powershell
python --version
where.exe python
```

La versión debe comenzar con:

```text
Python 3.10
```

La ruta de Python debe apuntar al entorno `heart-ml`.

---

## 5. Instalar las dependencias

Actualiza las herramientas básicas de instalación:

```powershell
python -m pip install --upgrade pip setuptools wheel
```

Instala las dependencias de ejecución:

```powershell
python -m pip install -r requirements.txt
```

Instala las dependencias de desarrollo y pruebas:

```powershell
python -m pip install -r requirements-dev.txt
```

Comprueba que no existan dependencias rotas:

```powershell
python -m pip check
```

Resultado esperado:

```text
No broken requirements found.
```

---

## 6. Verificar los artefactos necesarios

La aplicación requiere estos archivos:

```powershell
Test-Path ".\models\best_pipeline.pkl"
Test-Path ".\models\model_config.json"
Test-Path ".\models\model_manifest.json"
Test-Path ".\data\02_intermediate\process_data.parquet"
```

Todos deben devolver:

```text
True
```

Si alguno falta, la descarga o el clon del repositorio está incompleto. Si el proyecto utiliza Git LFS, ejecuta:

```powershell
git lfs install
git lfs pull
```

---

## 7. Ejecutar las pruebas

### Pruebas de integración del artefacto real

```powershell
python -m pytest -q -m integration
```

Estas pruebas cargan el modelo real y verifican su compatibilidad con la aplicación.

### Suite completa

```powershell
python -m pytest -q
```

Todas las pruebas deben terminar aprobadas.

En la línea base validada al redactar esta guía se obtuvo:

```text
3 passed en integración
47 passed en la suite completa
```

El número exacto puede aumentar en versiones posteriores; el criterio correcto es que no existan errores ni fallos.

Las advertencias de dependencias antiguas no equivalen necesariamente a una prueba fallida. Revisa el resumen final de pytest.

---

## 8. Ejecutar la aplicación

Desde la raíz del repositorio y con `heart-ml` activado:

```powershell
python -m streamlit run ".\src\app.py"
```

Streamlit mostrará una dirección similar a:

```text
Local URL: http://localhost:8501
```

Abre en el navegador:

```text
http://localhost:8501
```

La aplicación debe:

- cargar sin errores de contrato;
- mostrar el formulario completo;
- incluir el campo `HDL`;
- no incluir `DiastolicBP`;
- permitir ejecutar una clasificación de prueba.

Para detener Streamlit, vuelve a PowerShell y presiona:

```text
Ctrl + C
```

### Si el puerto 8501 está ocupado

```powershell
python -m streamlit run ".\src\app.py" --server.port 8502
```

Después abre:

```text
http://localhost:8502
```

> El aviso del navegador “No es seguro” es normal cuando se abre una aplicación local por HTTP. No significa que la instalación haya fallado.

---

## 9. Verificaciones de calidad opcionales

### Ruff

```powershell
python -m ruff check .
python -m ruff format --check .
```

### Pre-commit

Este comando requiere que el proyecto sea un repositorio Git:

```powershell
python -m pre_commit run --all-files
```

Si aparece un error indicando que Git no está disponible o que no estás en un repositorio, comprueba:

```powershell
git --version
git status
```

---

## 10. Ejecución sin activar Conda

También puedes usar directamente el intérprete del entorno:

```powershell
$python = "$env:USERPROFILE\anaconda3\envs\heart-ml\python.exe"

& $python -m pip check
& $python -m pytest -q
& $python -m streamlit run ".\src\app.py"
```

En una instalación de Miniconda, la ruta puede ser:

```powershell
$python = "$env:USERPROFILE\miniconda3\envs\heart-ml\python.exe"
```

Comprueba primero que exista:

```powershell
Test-Path $python
```

---

## 11. Problemas frecuentes

### `conda` no se reconoce

Usa **Anaconda Prompt** o ejecuta:

```powershell
conda init powershell
```

Luego reinicia PowerShell.

### `ModuleNotFoundError`

Confirma que el entorno está activo:

```powershell
conda activate heart-ml
python --version
```

Después reinstala las dependencias:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

### `streamlit` no se reconoce

No ejecutes `streamlit` directamente. Usa:

```powershell
python -m streamlit run ".\src\app.py"
```

### `pytest` encuentra pruebas duplicadas

Probablemente hay un repositorio dentro de otro. Desde la raíz:

```powershell
Test-Path ".\Heart-Disease-Prediction-ML"
Test-Path ".\Heart-Disease-Prediction-ML-main"
```

Ambos deberían devolver `False`, salvo que ese nombre corresponda intencionalmente a otra carpeta no relacionada.

### Faltan el modelo o el dataset

Comprueba:

```powershell
Test-Path ".\models\best_pipeline.pkl"
Test-Path ".\data\02_intermediate\process_data.parquet"
```

Si el repositorio usa Git LFS:

```powershell
git lfs pull
```

### Advertencia de versiones al cargar el modelo

PyCaret puede mostrar una advertencia si algunas versiones instaladas difieren de las usadas al serializar el modelo. Si las pruebas de integración, la suite completa y la aplicación funcionan, la advertencia no impide necesariamente la ejecución. Debe registrarse y revisarse antes de una publicación productiva.

### Error de contrato de características

Actualiza la copia local y repite la instalación:

```powershell
git pull
python -m pip install -r requirements.txt
python -m pytest -q -m integration
```

No añadas manualmente `HeartDisease` a las entradas del formulario ni modifiques el modelo para ocultar el error.

---

## 12. Instalación opcional con Docker Desktop

Instala Docker Desktop y comprueba:

```powershell
docker --version
```

Desde la raíz del proyecto:

```powershell
docker build -t nhanes-heart-prototype .
docker run --rm -p 8501:8501 nhanes-heart-prototype
```

Abre:

```text
http://localhost:8501
```

Detén el contenedor con:

```text
Ctrl + C
```

Docker es opcional; la instalación principal recomendada en Windows utiliza Conda.

---

## 13. Actualizar una instalación existente

Desde la raíz del repositorio:

```powershell
git status
git pull
conda activate heart-ml
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip check
python -m pytest -q
```

Si tienes modificaciones locales, no ejecutes `git pull` sin revisar primero:

```powershell
git status
git diff
```

---

## 14. Resumen rápido

```powershell
git clone <URL_DEL_REPOSITORIO>
cd <NOMBRE_DE_LA_CARPETA_CLONADA>

conda create -n heart-ml python=3.10 -y
conda activate heart-ml

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt

python -m pip check
python -m pytest -q -m integration
python -m pytest -q

python -m streamlit run ".\src\app.py"
```

Abrir:

```text
http://localhost:8501
```
