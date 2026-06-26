# Trazabilidad de correcciones

| Hallazgo | Estado | Corrección |
|---|---|---|
| UI sin `HDL` y con `DiastolicBP` | Corregido | Contrato estricto y UI con `HDL` |
| Sexo 0/1 frente a NHANES 1/2 | Corregido | `SexCode` y validación 1/2 |
| Pytest bloqueado por prueba BRFSS | Corregido | Prueba heredada reescrita |
| Metadato PyCaret incluía `HeartDisease` como entrada inesperada | Corregido | Normalización exclusiva del target externo y control independiente de fuga en `actual_estimator` |
| Artefactos distintos en app/entrenamiento/auditoría | Corregido | Manifiesto único y candidatos no desplegados |
| Entrenamiento sobrescribía configuración | Corregido | Configuración canónica separada |
| Riesgo de usar `SEQN` | Corregido | Fuera de `MODEL_INPUT_FEATURES` |
| Objetivo mezclaba infarto, ACV y ausentes | Corregido en flujo nuevo | Solo `MCQ160E`; códigos no explícitos excluidos |
| Evaluación contaminada | Corregido en flujo nuevo | Test separado; no se consulta al comparar estrategias y el objeto evaluado es exactamente el guardado |
| Docker Buster y `.streamlit` ausente | Corregido | Bookworm y configuración incluida |
| `read_sas()` desempaquetado | Corregido | Retorno directo del DataFrame |
| `dict()` de Pydantic 2 | Corregido | `model_dump()` |
| README contradictorio | Corregido | Documento único y sin sobreafirmaciones |
| Umbrales inconsistentes | Parcial | Manifiesto único; legado sigue no validado |
| SHAP sin preprocesamiento | Corregido preventivamente | Retirado de UI |
| `fit()` acumulaba árboles | Corregido | Reinicio explícito y prueba de regresión |
| Dos clases `XGBoostScratch` | Corregido | Implementación única y alias compatible |
| Informe de equidad inválido | Invalidado | JSON reemplazado y script corregido |
| Nombres, rutas y logs antiguos | Corregido | Retirados de la rama activa |
| Documentación y notebooks obsoletos | Corregido | Eliminados o reemplazados por fuentes canónicas |

## Pendiente científico

La corrección de código no transforma el pickle heredado en un modelo validado.
El cierre científico exige generar datos corregidos, entrenar un candidato y
revisar la evaluación independiente.
