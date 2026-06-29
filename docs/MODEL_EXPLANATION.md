# Explicación del modelo de CardioHistory ML

Este documento relaciona la teoría de XGBoost con el artefacto realmente
desplegado en CardioHistory ML. No describe una red neuronal ni atribuye al
modelo resultados que no hayan sido medidos.

## 1. Naturaleza del modelo

El estimador final es `xgboost.sklearn.XGBClassifier` con booster `gbtree`. Es
un ensemble secuencial de árboles de decisión construido mediante gradient
boosting: cada árbol nuevo se ajusta para corregir, en términos de la función de
pérdida, el margen acumulado por los árboles anteriores.

No es una red neuronal ni un modelo de deep learning. La implementación
educativa de `src/model.py` reproduce algunas ideas matemáticas con árboles
simplificados, pero el despliegue usa la biblioteca oficial XGBoost dentro de un
pipeline PyCaret.

El objetivo técnico es clasificación binaria mediante `binary:logistic`. La
clase `1` representa antecedente autorreportado de infarto asociado con
`MCQ160E`; no representa un evento futuro.

## 2. Predicción aditiva

Para una entrada transformada (x), el margen del ensemble después de (M)
rondas puede expresarse como:

$$
F_M(x)=F_0(x)+\sum_{m=1}^{M}\eta f_m(x)
$$

donde:

- (F_0(x)) es el margen inicial;
- (f_m(x)) es el valor de la hoja alcanzada en el árbol (m);
- (\eta) es la tasa de aprendizaje;
- (M) es el número de rondas/árboles.

En el artefacto desplegado, (M=60) y (\eta=0.4). El objetivo
`binary:logistic` transforma el margen en un score acotado:

$$
p(y=1\mid x)=\sigma(F_M(x))
=\frac{1}{1+e^{-F_M(x)}}
$$

XGBoost entrega este valor como salida de la clase positiva. Puede interpretarse
como una probabilidad estimada por el modelo, pero que esté entre 0 y 1 no
demuestra calibración. En la documentación operativa se usa el término *score*
para evitar esa sobreinterpretación.

## 3. Pérdida, gradiente y Hessiano

Para clasificación binaria, la pérdida logística por observación es:

$$
\ell(y_i,p_i)=-\left[y_i\log(p_i)+(1-y_i)\log(1-p_i)\right]
$$

y la pérdida total agrega las observaciones más la regularización de los
árboles. El `eval_metric` efectivo del artefacto es `logloss`.

Al comenzar una nueva ronda, XGBoost deriva la pérdida respecto del margen
actual. Para log-loss binaria sin escribir todavía el peso de clase:

$$
g_i=\frac{\partial \ell_i}{\partial F}=p_i-y_i
$$

$$
h_i=\frac{\partial^2 \ell_i}{\partial F^2}=p_i(1-p_i)
$$

El gradiente (g_i) indica dirección y magnitud de la corrección; el Hessiano
(h_i) representa la curvatura local. XGBoost usa una aproximación de segundo
orden para evaluar el siguiente árbol:

$$
\mathcal{L}^{(m)}\approx
\sum_i\left[g_i f_m(x_i)+\frac{1}{2}h_i f_m(x_i)^2\right]
+\Omega(f_m)
$$

Cada árbol agrupa observaciones con gradientes y Hessianos similares en hojas.
Así intenta reducir la pérdida del ensemble acumulado, no aprender una regla
aislada desde cero.

En este proyecto, `scale_pos_weight=38` multiplica la contribución de las
observaciones positivas en el objetivo. En consecuencia, sus gradientes y
Hessianos tienen mayor influencia durante el ajuste.

## 4. Peso óptimo de una hoja

Para una hoja (j), sean:

$$
G_j=\sum_{i\in I_j}g_i,
\qquad
H_j=\sum_{i\in I_j}h_i
$$

Con regularización L2, el valor óptimo de esa hoja es:

$$
w_j^*=-\frac{G_j}{H_j+\lambda}
$$

El artefacto usa `reg_lambda=0.7`. Un (\lambda) mayor reduce la magnitud de
los valores de hoja cuando la evidencia acumulada no es suficiente.

La configuración también utiliza regularización L1 con
`reg_alpha=0.001`. Con (\alpha>0), el gradiente agregado se somete a un
umbral suave:

$$
w_j^*=-\frac{\operatorname{sign}(G_j)
\max(|G_j|-\alpha,0)}{H_j+\lambda}
$$

Por ello, la expresión L2 simple explica la idea central, pero la segunda
ecuación representa mejor la presencia de `reg_alpha` en el modelo real.

Los valores (w_j) son contribuciones de hojas al margen. No son pesos
sinápticos ni parámetros de neuronas.

## 5. Ganancia de una división

Para una partición candidata con grupos izquierdo (L) y derecho (R), la
ganancia regularizada L2 puede escribirse como:

$$
\operatorname{Gain}=\frac{1}{2}
\left[
\frac{G_L^2}{H_L+\lambda}
+\frac{G_R^2}{H_R+\lambda}
-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}
\right]-\gamma
$$

La división es atractiva si modelar dos hojas reduce más la pérdida que mantener
una sola, después de considerar la penalización (\gamma). Cuando hay L1, los
términos de gradiente se ajustan mediante el umbral suave descrito antes.

XGBoost evalúa candidatos de corte para las características disponibles en cada
árbol, acumula (G) y (H) a cada lado y selecciona particiones con mayor
ganancia permitida por sus restricciones. Este proceso se repite recursivamente
hasta alcanzar condiciones como profundidad, peso mínimo o falta de ganancia.

## 6. Regularización en el artefacto

La configuración real combina varios controles:

| Control | Valor | Efecto técnico esperado |
|---|---:|---|
| `max_depth` | 6 | Limita la profundidad y, por tanto, la complejidad de interacciones por árbol. |
| `reg_alpha` | 0.001 | Aplica una penalización L1 pequeña a los valores de hoja. |
| `reg_lambda` | 0.7 | Regulariza en L2 la magnitud de los valores de hoja. |
| `subsample` | 0.7 | Cada árbol usa una fracción de filas, introduciendo aleatoriedad. |
| `colsample_bytree` | 0.7 | Cada árbol considera una fracción de características. |
| `min_child_weight` | 1 | Exige un mínimo de Hessiano acumulado para formar un hijo. |

`subsample` no es dropout: muestrea filas antes de construir árboles. Además,
el booster es `gbtree`, no `dart`, por lo que no hay abandono de árboles al
estilo del booster DART.

## 7. Sesgo, varianza y capacidad

La configuración debe analizarse como un conjunto:

- Árboles más profundos pueden reducir sesgo al representar interacciones más
  complejas, pero pueden aumentar varianza.
- Un mayor número de árboles aumenta la capacidad del ensemble.
- Una tasa de aprendizaje pequeña suele requerir más árboles para alcanzar una
  capacidad comparable.
- `learning_rate=0.4` es relativamente elevado; cada nuevo árbol tiene una
  contribución importante al margen.
- `subsample=0.7`, `colsample_bytree=0.7` y las penalizaciones de hojas pueden
  reducir sobreajuste.
- La profundidad máxima configurada y observada es 6; el artefacto contiene 60
  árboles, 5 168 nodos y 2 614 hojas.

No puede concluirse que el equilibrio entre sesgo y varianza sea adecuado u
óptimo sin curvas de aprendizaje, validación reproducible y comparación con
candidatos.

## 8. Desbalance de clases

El valor:

```text
scale_pos_weight = 38
```

da mayor peso a los errores de las observaciones positivas durante el
entrenamiento. En términos generales, esto puede aumentar la detección de la
clase positiva y también puede aumentar falsos positivos, reduciendo precision.
Son consecuencias posibles, no resultados demostrados para este artefacto.

El peso altera la función optimizada y puede modificar la distribución de los
scores. Por eso deben evaluarse al menos métricas por clase, matriz de confusión,
PR-AUC y calibración en datos protegidos. El valor `38` por sí solo no demuestra
un nivel concreto de recall, precision o utilidad.

## 9. Score, calibración y umbral

Estos conceptos no son equivalentes:

- **score:** salida numérica de `predict_proba()` para la clase `1`;
- **probabilidad estimada:** interpretación del score producido por
  `binary:logistic`;
- **calibración:** correspondencia empírica entre scores y frecuencias observadas;
- **umbral:** regla externa que convierte el score en clase;
- **clase final:** `0` o `1` después de aplicar la regla.

El despliegue usa:

$$
\widehat y=
\begin{cases}
1, & p\ge 0.20 \\
0, & p<0.20
\end{cases}
$$

El umbral `0.20` es menor que el convencional `0.50`, de modo que, para los
mismos scores, puede clasificar como positivas más entradas. Esto puede cambiar el
balance entre falsos positivos y falsos negativos. No prueba que el umbral sea
adecuado: el manifiesto lo identifica como heredado y no validado
independientemente.

Cambiar el umbral no modifica árboles, hojas ni scores y no reentrena el modelo;
solo cambia la decisión posterior.

## 10. Incertidumbre

La distancia entre un score y `0.20` no es una medida formal de incertidumbre.
Un score de `0.21` y uno de `0.80` caen en la misma clase, pero esa distancia no
proporciona por sí sola un intervalo de confianza, una probabilidad calibrada ni
una garantía de corrección.

Cuantificar incertidumbre requeriría métodos y evaluación adicionales, por
ejemplo intervalos por remuestreo para métricas, análisis de calibración,
variación entre modelos y validación externa. Ninguna de esas evidencias está
demostrada para el artefacto heredado.

## 11. Comparación con redes neuronales

| Concepto | XGBoost del proyecto | Red neuronal |
|---|---|---|
| Unidad básica | Árbol de decisión con hojas | Neurona y conexiones ponderadas |
| Composición | Suma secuencial de 60 árboles | Capas de transformaciones |
| Estructura interna | Particiones por característica y umbral | Capas de unidades conectadas |
| Parámetros aprendidos | Cortes, estructura y valores de hoja | Pesos sinápticos y sesgos |
| No linealidad | Rutas discretas de los árboles y sigmoide final | Funciones de activación por unidad |
| Optimización | Gradient boosting con gradiente/Hessiano de la pérdida | Descenso por gradiente y backpropagation neuronal |
| Control estocástico usado aquí | `subsample` y `colsample_bytree` | Puede incluir mini-batches y dropout |
| Capas ocultas | No existen | Habituales |
| Dropout | No se usa; `subsample` no es dropout | Puede desactivar unidades durante entrenamiento |

No hay activaciones por neurona, capas ocultas, pesos sinápticos ni
backpropagation neuronal. Los valores de las hojas no deben llamarse pesos
neuronales. El uso de gradientes no convierte XGBoost en una red neuronal:
ambas familias optimizan funciones, pero lo hacen con estructuras diferentes.

## 12. Preprocesamiento real: de 27 a 31

El modelo no recibe directamente las 27 columnas externas. El pipeline PyCaret
aplica imputación numérica por media, imputación categórica por moda,
codificación ordinal de `Sex`, one-hot encoding de `Race`, un paso residual sin
columnas efectivas y `RobustScaler`.

`Race` tiene cinco categorías. Al sustituir una columna por sus indicadores, la
expansión neta es de cuatro columnas, coherente con:

```text
27 variables externas → 31 características transformadas
```

Observaciones críticas:

- `RobustScaler` no suele ser necesario para árboles porque sus cortes dependen
  del orden de los valores, no de una distancia euclidiana.
- Debe conservarse porque forma parte del pipeline serializado y retirarlo
  cambiaría el flujo reproducido.
- La codificación, los tipos y el orden de columnas deben permanecer idénticos a
  `MODEL_INPUT_FEATURES`.
- `HeartDisease` no aparece entre las 31 características del estimador.
- `Glucose` corresponde a **Glucosa sérica del perfil bioquímico NHANES
  (`LBXSGL`)**.

## 13. Por qué no puede declararse superior

La complejidad del artefacto y el funcionamiento de la inferencia no demuestran
calidad predictiva. Para comparar este XGBoost con otros candidatos se
necesitaría, como mínimo:

1. dataset y procedencia reproducibles;
2. construcción verificable de `MCQ160E` y de la cohorte;
3. partición protegida no utilizada para seleccionar modelo o umbral;
4. validación cruzada sobre desarrollo con el mismo protocolo para candidatos;
5. métricas por clase, ROC-AUC, PR-AUC y matriz de confusión;
6. intervalos de confianza o análisis de variabilidad;
7. evaluación de calibración;
8. análisis de sesgo y desempeño por subgrupos con tamaños suficientes;
9. evaluación independiente en otro periodo o población.

El repositorio contiene componentes para parte de ese proceso, pero no conserva
evidencia histórica suficiente para atribuir esos resultados al pickle
desplegado. Por ello, la formulación científica correcta es: artefacto funcional
y caracterizado técnicamente, con desempeño predictivo histórico no demostrado de
forma reproducible.

## 14. Referencias académicas esenciales

1. Chen, T. y Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*.
   Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge
   Discovery and Data Mining. DOI: `10.1145/2939672.2939785`.
2. Documentación oficial de XGBoost: parámetros de árboles, objetivo
   `binary:logistic` y predicción.
3. CDC/NCHS. Documentación oficial de NHANES para cuestionarios, demografía,
   examen y laboratorio, incluida la variable `MCQ160E`.
4. Documentación oficial de PyCaret 3 para clasificación y pipelines.
5. Documentación de scikit-learn sobre métricas, calibración y validación de
   modelos.

Estas fuentes sustentan los conceptos y la semántica técnica. No sustituyen la
evidencia experimental que falta para evaluar el artefacto desplegado.
