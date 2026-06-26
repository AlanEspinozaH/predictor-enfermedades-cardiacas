# Data Card

## Fuente

National Health and Nutrition Examination Survey (NHANES), combinando tablas de
demografía, cuestionarios, examen físico y laboratorio por `SEQN` y ciclo.

## Población objetivo del dataset corregido

- Participantes de 20 años o más.
- Respuesta explícita `1` o `2` en `MCQ160E`.
- Predictores disponibles o faltantes; los faltantes no se imputan antes del split.

## Transformaciones principales

- Unión uno a uno por participante dentro de cada componente.
- Promedio de medidas sistólicas disponibles.
- Armonización de nombres a 27 características canónicas.
- Actividad física definida por `PAQ650` (actividad recreativa vigorosa).
- Alcohol definido como consumo durante los últimos 12 meses mediante `ALQ120Q`/`ALQ121`; respuestas no aplicables o ausentes permanecen faltantes.
- Codificación de `MCQ160E`: `1 -> 1`, `2 -> 0`.
- Exclusión de respuestas desconocidas, rechazadas o ausentes del objetivo.
- Derivación de LDL mediante Friedewald solo cuando es elegible y sin reemplazar
  una medición existente.
- Conservación de `SEQN` únicamente para detectar duplicados y fugas.

## Artefacto heredado

`data/02_intermediate/process_data.parquet` se conserva por trazabilidad del
modelo desplegado, pero **no debe utilizarse para reentrenar**. No representa la
cohorte corregida.

## Artefactos corregidos esperados

- `data/02_processed/nhanes_heart_attack_modeling_raw.parquet`.
- archivo lateral `.provenance.json`.
- `data/02_intermediate/nhanes_heart_attack_modeling.parquet` cuando se ejecuta
  la etapa de ingestión validada.

## Limitaciones

- NHANES es observacional y transversal.
- Parte de las variables son autorreportadas.
- La disponibilidad de laboratorios cambia entre ciclos.
- La unión de componentes puede reducir el tamaño de la cohorte y alterar su
  composición.
- El prototipo no incorpora los pesos complejos de encuesta para inferencia
  poblacional; por ello no debe presentar estimaciones nacionales.
