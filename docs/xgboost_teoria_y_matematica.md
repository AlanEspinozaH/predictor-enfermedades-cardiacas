# XGBoost: teoría y desarrollo matemático

Para clasificación binaria, el ensamble produce un margen:

\[
F_t(x)=F_{t-1}(x)+\eta f_t(x)
\]

y una probabilidad:

\[
p(x)=\sigma(F_t(x))=\frac{1}{1+e^{-F_t(x)}}.
\]

Con entropía cruzada binaria, para cada observación:

\[
g_i = p_i-y_i,
\qquad
h_i=p_i(1-p_i).
\]

Para una hoja con índices \(I\), gradiente acumulado \(G=\sum_{i\in I}g_i\) y
hessiano \(H=\sum_{i\in I}h_i\), el peso regularizado es:

\[
w^*=-\frac{G}{H+\lambda}.
\]

La ganancia de dividir una región en izquierda y derecha es:

\[
\text{Gain}=\frac{1}{2}
\left[
\frac{G_L^2}{H_L+\lambda}+
\frac{G_R^2}{H_R+\lambda}-
\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}
\right]-\gamma.
\]

Se acepta una partición solo cuando su ganancia neta es positiva. La tasa
\(\eta\) reduce la contribución de cada árbol y puede mejorar generalización a
costa de requerir más estimadores.

La implementación del repositorio recorre umbrales de forma exhaustiva y, por
tanto, es adecuada solo para conjuntos pequeños y fines pedagógicos.
