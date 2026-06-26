# Informe de corrección del repositorio

## Resultado

El proyecto pasó de ser una colección de scripts, notebooks y documentos
contradictorios a un prototipo con fuentes canónicas explícitas para esquema,
artefacto, datos y evaluación.

## Trabajo completado

- Eliminación de nombres personales, rutas de equipos anteriores, logs y cachés.
- Retiro de documentos y notebooks históricos que no representaban el flujo actual.
- Contrato estricto entre UI y modelo.
- Corrección de `HDL`, `DiastolicBP`, sexo, edad y exclusión de `SEQN`.
- Corrección de lectura XPT y rutas reproducibles.
- Objetivo restringido a `MCQ160E` con exclusión de códigos no válidos.
- Separación del test antes de imputación y ajuste.
- Registro de candidatos sin sobreescribir el despliegue.
- Invalidación explícita del informe de equidad anterior.
- Corrección conceptual de la interfaz: clasificación de antecedente, no riesgo futuro.
- Eliminación de SHAP en la UI hasta disponer de una explicación alineada con el
  preprocesamiento y validada.
- Unificación de la implementación educativa de boosting.
- Docker, pre-commit, documentación y pruebas actualizados.

## Lo que no puede afirmarse todavía

No existe aún evidencia suficiente para declarar precisión clínica, utilidad de
tamizaje, calibración externa o equidad. Esas conclusiones requieren reconstruir
la cohorte, reentrenar y evaluar un candidato nuevo en un test protegido.
