# Política de evaluación

## Separaciones obligatorias

1. **Desarrollo:** entrenamiento, validación cruzada, selección de modelo y umbral.
2. **Test protegido:** una sola evaluación después de congelar todas las decisiones.
3. **Cohorte externa:** deseable para evaluar transporte a otra población o periodo.

Cuando existen varios ciclos NHANES, el último ciclo se reserva temporalmente si
ambas clases quedan representadas. De lo contrario se utiliza un split
estratificado reproducible.

## Métricas mínimas

- Matriz de confusión: TN, FP, FN y TP.
- Recall o sensibilidad.
- Precisión o valor predictivo positivo.
- Especificidad.
- F1.
- ROC-AUC.
- PR-AUC, especialmente importante con baja prevalencia.
- Brier score para calidad probabilística.
- Prevalencia y cantidad de positivos.
- Umbral exacto aplicado.
- Intervalos de confianza para métricas por subgrupo.

## Umbral

El umbral debe elegirse solo con datos de desarrollo. La regla actual de
entrenamiento busca máximo recall sujeto a precisión mínima de 0.40; si no existe
un punto que cumpla, utiliza máximo F1. Esa regla es académica y no una política
clínica validada. El script no evalúa el test protegido por defecto: la opción
`--evaluate-protected-test` se reserva para la estrategia final ya seleccionada.

El umbral `0.20` del modelo heredado se conserva por trazabilidad. No debe
presentarse como óptimo ni transferirse automáticamente a un candidato nuevo.

## Criterios de aceptación

No se fija un valor como “recall >= 85 %” sin evidencia. Antes de aceptar un
candidato deben declararse:

- métrica primaria;
- mínimo aceptable y justificación;
- límite inferior del intervalo de confianza;
- coste relativo de falsos positivos y falsos negativos;
- calibración;
- desempeño por sexo, edad y otros grupos con tamaño suficiente.
