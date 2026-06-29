# Changelog

## 2.1.0 — Trazabilidad de candidatos y validación verificable (2026-06-29)

- Incorporado un manifiesto de candidato v1 con rutas relativas seguras,
  hashes SHA-256 y relaciones verificadas entre modelo, configuración,
  procedencia y `selection_report.json`.
- Registrados todos los candidatos considerados y una selección reproducible:
  mayor recall con `precision >= 0.40`, o mayor recall global como fallback.
- Eliminado el umbral implícito `0.50`; manifiestos, overrides y reportes exigen
  un valor finito que cumpla `0 < threshold < 1`.
- Añadida procedencia externa enlazada al hash del dataset, con separación entre
  declaración de independencia y auditoría independiente.
- Publicados JSON y CSV relacionados mediante staging, hash y reemplazo atómico;
  los directorios de candidatos aparecen solo tras verificar sus componentes.
- Añadidas pruebas sintéticas de selección, integridad, atomicidad, privacidad,
  procedencia y compatibilidad de los modos desplegado/candidato.

## 2.0.0 — Renovación académica para Inteligencia Artificial

- Consolidado el contrato canónico de 27 variables y la verificación del orden
  de entrada frente al pipeline desplegado.
- Incorporados el registro de artefactos, la resolución de rutas y la
  verificación SHA-256 antes de deserializar el pipeline confiable.
- Caracterizado el estimador final `xgboost.sklearn.XGBClassifier`, su
  preprocesamiento de 27 a 31 características y el umbral operativo definido en
  el manifiesto.
- Corregida la semántica de `Glucose` a glucosa sérica del perfil bioquímico
  NHANES (`LBXSGL`).
- Añadido un smoke test de inferencia que reutiliza las interfaces canónicas.
- Consolidada la documentación de instalación, arquitectura, ficha del modelo y
  explicación técnica.
- Delimitado el objetivo a la clasificación de antecedente autorreportado de
  infarto asociado con `MCQ160E`, sin afirmaciones diagnósticas o prospectivas.
- Preparada la base documental para una renovación visual posterior; la interfaz
  no se declara renovada en esta versión.

## 1.1.0 — Compatibilidad del artefacto desplegado (2026-06-26)

- Normalizado exclusivamente `HeartDisease` cuando aparece como metadato externo
  de PyCaret, manteniéndolo fuera de las características del estimador.
- Unificada la carga validada del artefacto entre Streamlit y las pruebas de
  integración.
- Añadidas comprobaciones del contrato externo de 27 variables y de las 31
  características transformadas.

## 1.0.0 — Saneamiento estructural (2026-06-26)

- Retirados datos personales, rutas locales, logs, cachés y material histórico
  incompatible con el flujo vigente.
- Corregidos el contrato de entrada, la codificación NHANES y el objetivo basado
  exclusivamente en `MCQ160E`.
- Separados los candidatos de entrenamiento del artefacto desplegado y de la
  evaluación protegida.
- Unificada la implementación educativa `XGBoostScratch` y actualizadas pruebas y
  controles de calidad.
