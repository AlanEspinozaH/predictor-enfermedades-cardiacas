# Arquitectura de CardioHistory ML

**Clasificación explicable de antecedente autorreportado de infarto con datos
NHANES**

## 1. Propósito y alcance

La arquitectura implementa dos responsabilidades separadas:

1. inferencia con el artefacto heredado identificado por el manifiesto;
2. experimentación académica para preparar datos y generar candidatos no
   desplegados.

El objetivo es la clase binaria asociada con `MCQ160E`: antecedente
autorreportado de infarto. El sistema no modela una condición actual ni un
evento futuro.

En inferencia, las responsabilidades se distribuyen de esta forma:

- **interfaz:** captura valores y presenta el resultado;
- **validación:** rechaza valores fuera del esquema y campos adicionales;
- **adaptación:** construye un `DataFrame` en el orden canónico;
- **registro de artefactos:** resuelve rutas, hashes y umbral;
- **preprocesamiento:** transforma las 27 entradas en 31 características;
- **inferencia:** obtiene el score correspondiente a la clase positiva `1`;
- **umbral:** aplica el valor operativo `0.20` del manifiesto;
- **presentación:** muestra una clasificación académica con sus límites.

## 2. Diagrama de componentes

```mermaid
flowchart LR
    UI[Streamlit UI] --> VALIDATION[Pydantic InputData]
    VALIDATION --> ADAPTER[UserInputAdapter]
    ADAPTER --> PIPELINE[PyCaret Pipeline]
    REGISTRY[Artifact registry + manifest] -->|resolve paths and verify hashes| PIPELINE
    PIPELINE --> PREPROCESS[Mean/mode imputation + encoding + RobustScaler]
    PREPROCESS --> MODEL[XGBClassifier]
    MODEL --> SCORE[Positive-class score]
    SCORE --> THRESHOLD[Operational threshold 0.20]
    THRESHOLD --> RESULT[Academic classification]
```

El bloque `PyCaret Pipeline` agrupa el preprocesamiento serializado y el
estimador final. El diagrama los separa para hacer visible el flujo efectivo;
no representa servicios independientes.

## 3. Flujo efectivo de inferencia

```text
27 variables externas
→ validación Pydantic
→ orden canónico
→ pipeline PyCaret
→ 31 características transformadas
→ XGBClassifier
→ score de la clase 1
→ umbral 0.20
→ clasificación académica
```

El flujo comienza en `src/app.py`. `InputData` valida la entrada y configura
`extra="forbid"`; no se crean variables ausentes ni se aceptan campos
obsoletos. `UserInputAdapter` vuelca la entrada validada con el orden de
`MODEL_INPUT_FEATURES`. `PyCaretAdapter` conserva ese contrato, ejecuta
`predict_proba()` e identifica la columna asociada con la clase `1`.

El resultado binario se calcula comparando ese score con el umbral del
manifiesto. Cambiar el umbral modificaría la decisión, pero no reentrenaría el
pipeline.

## 4. Responsabilidad de los módulos

| Componente | Responsabilidad efectiva |
|---|---|
| `src/app.py` | Construye la interfaz Streamlit, carga el despliegue mediante el registro, recoge las 27 entradas y presenta score y clasificación con advertencias de alcance. |
| `src/interfaces.py` | Define el protocolo `HeartDiseaseModel` y el esquema Pydantic `InputData`, incluidos tipos, rangos y dominios categóricos. |
| `src/adapters.py` | `UserInputAdapter` valida y ordena la entrada; `PyCaretAdapter` valida el contrato del pipeline y extrae el score de la clase positiva. |
| `src/feature_contract.py` | Es la fuente canónica del orden de 27 variables, grupos numérico/categórico, objetivo y edad mínima. |
| `src/artifact_registry.py` | Lee el manifiesto, resuelve rutas internas seguras, verifica SHA-256, carga PyCaret y comprueba metadatos externos y ausencia del objetivo en el estimador. |
| `src/train_pycaret.py` | Prepara experimentos PyCaret sobre particiones de desarrollo, selecciona estrategias y umbrales de candidatos y guarda candidatos fechados sin desplegarlos automáticamente. |
| `src/evaluation.py` | Calcula métricas binarias a partir de etiquetas y scores ya obtenidos, valida el umbral y ofrece un intervalo de Wilson para recall. No aporta por sí solo evidencia del artefacto desplegado. |
| `src/model.py` | Implementa `XGBoostScratch`, un booster educativo simplificado para estudiar gradientes, Hessianos, hojas y ensamble aditivo. No es el estimador desplegado. |
| `src/tree/` | Contiene el árbol educativo, la pérdida logística de segundo orden y un alias compatible hacia `XGBoostScratch`. |
| `scripts/smoke_test.py` | Ejecuta una comprobación sintética de integridad, carga, contrato, score y umbral mediante las interfaces canónicas. |
| `models/model_manifest.json` | Identifica el modelo desplegado, configuración, hashes, datos históricos asociados, estado y umbral operativo. |

Otros módulos relevantes son `src/data_pipeline.py`, que valida y divide una
cohorte de modelado, y `src/validate_external.py`, que evalúa explícitamente una
cohorte aportada por el usuario. Ninguno forma parte del arranque de Streamlit.

## 5. Artefactos e integridad

El despliegue se define en `models/model_manifest.json`, no por búsqueda del
archivo `.pkl` más reciente. El manifiesto declara:

- `models/best_pipeline.pkl`, pipeline PyCaret serializado;
- `models/model_config.json`, espejo serializado del contrato de entrada;
- hashes SHA-256 del modelo y la configuración;
- identificador y estado del modelo;
- umbral operativo `0.20`;
- referencia al Parquet histórico asociado.

`load_deployed_artifacts(verify_hashes=True)` verifica modelo y configuración
antes de la carga. `load_validated_pycaret_pipeline()` usa el cargador de
PyCaret y comprueba el contrato del artefacto. El metadato externo heredado de
PyCaret puede incluir `HeartDisease`; el registro normaliza exclusivamente ese
nombre y confirma por separado que no aparezca entre las 31 características del
estimador final.

Un hash detecta diferencias respecto del archivo registrado, pero no vuelve
seguro un pickle malicioso. Solo debe deserializarse el artefacto confiable del
repositorio. Un pickle externo puede ejecutar código durante su carga.

## 6. Preprocesamiento desplegado

La secuencia observada del artefacto heredado incluye:

1. imputación numérica por media;
2. imputación categórica por moda;
3. codificación ordinal de `Sex`;
4. one-hot encoding de `Race`;
5. un paso residual sin columnas efectivas;
6. `RobustScaler`;
7. `XGBClassifier`.

El one-hot encoding explica la expansión neta de 27 a 31 características. El
escalamiento no suele ser necesario para árboles, pero forma parte del pipeline
serializado y debe conservarse para reproducir su inferencia.

## 7. Separación entre inferencia y entrenamiento

La aplicación puede ejecutar inferencia porque conserva el manifiesto, la
configuración y el pipeline desplegado. No requiere el Parquet para iniciar.

El entrenamiento es otro flujo. La preparación corregida combina componentes
NHANES por `SEQN`, restringe el objetivo a respuestas explícitas de `MCQ160E`,
excluye identificadores de los predictores y separa desarrollo y test antes de
ajustar imputación o modelo. Los candidatos se guardan bajo
`models/candidates/<timestamp>/` y no sustituyen el despliegue por defecto.

El Parquet histórico `data/02_intermediate/process_data.parquet` se conserva
para trazabilidad del artefacto heredado. No representa la cohorte corregida y
no debe emplearse como base de un nuevo entrenamiento. Aunque existe código para
reconstruir datos y entrenar candidatos, la procedencia y las decisiones del
entrenamiento histórico no están documentadas con detalle suficiente para
reproducir exactamente el pickle desplegado.

## 8. Niveles de verificación

### Pruebas no integradas

```powershell
python -m pytest -q -m "not integration"
```

Cubren contratos, adaptadores, registro, procesamiento, métricas y la
implementación educativa sin exigir la carga completa del artefacto real.

### Pruebas de integración

```powershell
python -m pytest -q -m integration
```

Cargan el pipeline real, verifican sus metadatos y las 31 características del
estimador, ejecutan `predict()`/`predict_proba()` con 27 entradas y comprueban el
arranque de Streamlit.

### Smoke test

```powershell
python scripts/smoke_test.py
```

Es la verificación operativa mínima de extremo a extremo. Usa una fila sintética,
comprueba hashes antes de deserializar, valida orden y dimensionalidad, obtiene
un score finito en `[0, 1]` y aplica el umbral del manifiesto.

Estos tres niveles verifican funcionamiento técnico. No sustituyen una
evaluación predictiva en una partición independiente.

## 9. Decisiones y limitaciones

- PyCaret se conserva como orquestador del pipeline serializado.
- XGBoost es el estimador final desplegado; `XGBoostScratch` es solo didáctico.
- El contrato externo tiene exactamente 27 variables y orden estricto.
- El estimador recibe 31 características después del preprocesamiento.
- El umbral `0.20` es heredado y no está validado independientemente.
- No hay evidencia reproducible suficiente para comparar este artefacto con
  candidatos y declararlo superior.
- La compatibilidad de un pickle depende de versiones de Python y bibliotecas;
  el entorno actualmente comprobado es Python 3.10.20.
- No se han demostrado calibración, transporte poblacional ni equidad.
- La arquitectura no contiene API HTTP propia, base de datos ni microservicios;
  Streamlit ejecuta el flujo local en un solo proceso.
