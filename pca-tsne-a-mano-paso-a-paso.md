# PCA y t-SNE a mano, paso a paso

> Guía de apoyo para el **Bloque 3** de la exposición (PCA · t-SNE).
> Objetivo: entender *qué hace por dentro* cada algoritmo resolviendo un ejemplo
> minúsculo **con lápiz y papel**, para poder explicarlo "a cerebro limpio".

Todos los números de esta guía están **verificados** contra `numpy` y `scikit-learn`:
salen idénticos a lo que produce el notebook.

---

## El ejemplo que usaremos (el mismo para los dos)

Para poder hacer las cuentas a mano, usamos **4 flores** y **solo 2 medidas**
(largo y ancho de pétalo, en cm). Es Iris en miniatura: en el notebook reducimos
**4 medidas → 2**; aquí reducimos **2 medidas → 1** para que quepa en la cabeza.

| Flor | Largo (x) | Ancho (y) |
|:----:|:---------:|:---------:|
| F1   | 5         | 4         |
| F2   | 4         | 5         |
| F3   | 2         | 1         |
| F4   | 1         | 2         |

Si los dibujas, F1 y F2 son pétalos **grandes**; F3 y F4 son pétalos **pequeños**.
Esa es la historia que los dos algoritmos van a "descubrir" solos.

---
---

# PARTE A — PCA paso a paso

> **Pregunta que responde PCA:** "Tengo muchas columnas. ¿Cuál es el **eje nuevo**
> a lo largo del cual los datos más se estiran (más varían)?" Ese eje resume la
> mayor cantidad de información en una sola dirección.

PCA es **100% resoluble a mano**. Son 6 pasos.

## Paso 1 — Centrar los datos (restar la media)

PCA mide *variación*, así que primero movemos la nube al origen.

Media de cada columna:

```
x̄ = (5 + 4 + 2 + 1) / 4 = 12/4 = 3
ȳ = (4 + 5 + 1 + 2) / 4 = 12/4 = 3
```

Restamos la media `(3, 3)` a cada flor → datos **centrados**:

| Flor | x − x̄ | y − ȳ |
|:----:|:-----:|:-----:|
| F1   |  +2   |  +1   |
| F2   |  +1   |  +2   |
| F3   |  −1   |  −2   |
| F4   |  −2   |  −1   |

> Comprobación rápida: cada columna debe sumar 0. (2+1−1−2 = 0 ✓)

## Paso 2 — Matriz de covarianza

La covarianza dice **cómo varían las columnas y cómo se mueven juntas**. Con 2
columnas es una matriz 2×2 simétrica:

$$
C=\begin{bmatrix} \mathrm{Var}(x) & \mathrm{Cov}(x,y)\\ \mathrm{Cov}(x,y) & \mathrm{Var}(y)\end{bmatrix}
$$

Fórmulas (sobre los datos **centrados**, dividiendo entre n = 4):

$$\mathrm{Var}(x)=\frac{1}{n}\sum (x_i-\bar x)^2,\qquad
\mathrm{Cov}(x,y)=\frac{1}{n}\sum (x_i-\bar x)(y_i-\bar y)$$

Cuentas, columna por columna:

```
Var(x)   = ( 2²  + 1²  + (−1)² + (−2)² ) / 4 = (4+1+1+4)/4 = 10/4 = 2.5
Var(y)   = ( 1²  + 2²  + (−2)² + (−1)² ) / 4 = (1+4+4+1)/4 = 10/4 = 2.5
Cov(x,y) = (2·1 + 1·2 + (−1)(−2) + (−2)(−1))/4 = (2+2+2+2)/4 = 8/4 = 2
```

$$
C=\begin{bmatrix} 2.5 & 2\\ 2 & 2.5 \end{bmatrix}
$$

> **Lectura:** `Cov(x,y) = 2 > 0` y grande → largo y ancho **crecen juntos**
> (pétalos grandes tienen las dos medidas grandes). Esa correlación es justo lo
> que PCA va a explotar.

## Paso 3 — Autovalores (cuánta varianza hay en cada eje)

Buscamos los números λ que cumplen $\det(C-\lambda I)=0$. Para una 2×2 esto es
una ecuación cuadrática que sí se resuelve a mano:

$$
\det\begin{bmatrix} 2.5-\lambda & 2\\ 2 & 2.5-\lambda\end{bmatrix}
=(2.5-\lambda)^2 - (2)(2) = 0
$$

```
(2.5 − λ)² = 4
 2.5 − λ   = ± 2
        λ  = 2.5 ∓ 2
```

$$\boxed{\lambda_1 = 4.5 \qquad \lambda_2 = 0.5}$$

Cada λ es la **varianza capturada** por su eje. El mayor (4.5) es la **primera
componente principal (PC1)**.

> Comprobación: la suma de autovalores debe igualar la suma de la diagonal de C.
> `4.5 + 0.5 = 5 = 2.5 + 2.5` ✓

## Paso 4 — Autovectores (la dirección de cada eje)

Para cada λ resolvemos $(C-\lambda I)\,v = 0$.

**PC1 (λ₁ = 4.5):**

```
[ 2.5−4.5      2    ] [v1]   [ −2   2 ] [v1]
[    2     2.5−4.5  ] [v2] = [  2  −2 ] [v2] = 0
→  −2·v1 + 2·v2 = 0  →  v1 = v2
```

Dirección `(1, 1)`. La normalizamos (longitud 1) dividiendo entre √2:

$$u_1=\left(\tfrac{1}{\sqrt2},\ \tfrac{1}{\sqrt2}\right)\approx(0.707,\ 0.707)$$

**PC2 (λ₂ = 0.5):**

```
[ 2  2 ] [v1]
[ 2  2 ] [v2] = 0   →   2·v1 + 2·v2 = 0   →   v1 = −v2
```

Dirección `(1, −1)`, normalizada:

$$u_2=\left(\tfrac{1}{\sqrt2},\ -\tfrac{1}{\sqrt2}\right)\approx(0.707,\ -0.707)$$

> **Interpretación bonita (úsala en la expo):**
> - **PC1 = (1, 1)/√2** pondera largo y ancho **por igual** → es el **"tamaño
>   general"** del pétalo. Es el eje donde más se diferencian las flores.
> - **PC2 = (1, −1)/√2** es **largo − ancho** → la **"forma"** (alargado vs.
>   redondo). Aporta poco.

## Paso 5 — Varianza explicada (¿cuánto perdemos al reducir?)

$$\text{ratio} = \frac{\lambda_i}{\lambda_1+\lambda_2}$$

```
PC1:  4.5 / (4.5 + 0.5) = 4.5 / 5 = 0.90  → 90 %
PC2:  0.5 / (4.5 + 0.5) = 0.5 / 5 = 0.10  → 10 %
```

> **Esto es el `explained_variance_ratio_` del notebook.** Si nos quedamos solo
> con PC1 (reducir 2D → 1D), conservamos el **90 %** de la información y tiramos
> solo el 10 % (la "forma"). En Iris pasa lo mismo pero con ~98 % usando 2 ejes.

## Paso 6 — Proyectar (calcular las coordenadas nuevas)

La coordenada de cada flor sobre PC1 es el **producto punto** del punto centrado
con `u₁`:

$$z_i = u_1\cdot(\text{punto centrado}) = \frac{(x_i-\bar x)+(y_i-\bar y)}{\sqrt2}$$

| Flor | centrado | cuenta            | PC1 (z) | PC2     |
|:----:|:--------:|:-----------------:|:-------:|:-------:|
| F1   | (+2,+1)  | (2+1)/√2          | **+2.12** | +0.71 |
| F2   | (+1,+2)  | (1+2)/√2          | **+2.12** | −0.71 |
| F3   | (−1,−2)  | (−1−2)/√2         | **−2.12** | +0.71 |
| F4   | (−2,−1)  | (−2−1)/√2         | **−2.12** | −0.71 |

> **Léelo:** sobre **PC1** las flores grandes (F1, F2) quedan en **+2.12** y las
> pequeñas (F3, F4) en **−2.12**. ¡Una sola coordenada separó grandes de pequeños!
> Eso es exactamente lo que hace el scatter de PCA en el notebook, pero en 1D.
>
> Detalle fino: la varianza de la columna PC1 es `(2.12² ·4)/4 = 4.5 = λ₁`. El
> autovalor **es** la varianza de la proyección. Cierra el círculo.

## Comprobación contra scikit-learn

Si corres `PCA(n_components=2).fit(X)` con estas 4 flores, obtienes:

- `components_` → las mismas direcciones `(0.707, 0.707)` y `(0.707, −0.707)`.
- `explained_variance_ratio_` → `[0.9, 0.1]` (idéntico).
- `transform(X)` → las mismas coordenadas de la tabla.

Dos avisos para que nada te sorprenda en vivo:
1. **El signo de un eje es arbitrario.** sklearn puede devolver PC2 volteado
   `(−0.707, 0.707)`; es el mismo eje, igual de válido.
2. sklearn divide entre **n−1** (no n), así que su `explained_variance_` sale
   `[6, 0.667]` en vez de `[4.5, 0.5]`. **El porcentaje (90/10) es idéntico**;
   solo cambia la escala. Usamos ÷n porque da números redondos.

## PCA en 6 pasos

```
1. Centrar:        restar la media de cada columna.
2. Covarianza:     matriz C (cómo varían y se correlacionan las columnas).
3. Autovalores λ:  resolver det(C − λI) = 0  → cuánta varianza por eje.
4. Autovectores u: resolver (C − λI)v = 0    → dirección de cada eje.
5. % varianza:     λ_i / Σλ  → cuánto conserva cada componente.
6. Proyectar:      z = u · (punto centrado)  → coordenadas nuevas.
```

---
---

# PARTE B — t-SNE paso a paso

> **Pregunta que responde t-SNE:** "No me importan los ejes ni las distancias
> globales. Solo quiero un mapa 2D donde **los vecinos sigan siendo vecinos**."

**Honestidad:** PCA se termina a mano. t-SNE **no**:
su último paso es una optimización iterativa que hace la computadora. Pero sí
podemos calcular a mano **todo lo que el algoritmo intenta lograr**, y eso es lo
que de verdad hay que entender.

### El ejemplo para t-SNE

4 puntos en "alta dimensión", agrupados en **2 parejas**. Lo único que t-SNE
necesita son las **distancias** entre ellos:

```
   A — B   están cerca   (distancia = 1)
   C — D   están cerca   (distancia = 1)
   las parejas entre sí, lejos (distancia = 8)
```

## Paso 1 — Distancias par a par

| dist | A | B | C | D |
|:----:|:-:|:-:|:-:|:-:|
| A    | 0 | 1 | 8 | 8 |
| B    | 1 | 0 | 8 | 8 |
| C    | 8 | 8 | 0 | 1 |
| D    | 8 | 8 | 1 | 0 |

## Paso 2 — Afinidades en alta dimensión (la gaussiana) → P

Convertimos cada distancia en una **"probabilidad de ser vecinos"** con una
campana de Gauss: cerca → afinidad alta; lejos → casi cero.

$$p_{j|i}=\frac{\exp\!\big(-d_{ij}^2 / 2\sigma^2\big)}{\sum_{k\neq i}\exp\!\big(-d_{ik}^2/2\sigma^2\big)}$$

Tomamos $2\sigma^2 = 4$. Calculamos el numerador $\exp(-d^2/4)$:

```
d = 1  →  exp(−1/4)  = exp(−0.25) ≈ 0.779
d = 8  →  exp(−64/4) = exp(−16)   ≈ 0.0000001 ≈ 0
```

Para el punto **A**: su afinidad con B es 0.779, con C y D es ≈ 0.
Al normalizar (dividir entre la suma de su fila):

```
p(B|A) = 0.779 / 0.779 ≈ 1        p(C|A) ≈ 0        p(D|A) ≈ 0
```

**Cada punto, en alta dimensión, prácticamente solo "ve" a su vecino de pareja.**

> **¿Qué es la `perplexity`?** Es cómo se elige σ. σ se ajusta para que cada
> punto tenga un número *efectivo* de vecinos igual a la perplexity (default 30).
> σ chico → pocos vecinos (lo que pasa aquí); σ grande → mira más lejos.

## Paso 3 — Simetrizar → matriz P final

t-SNE usa una afinidad simétrica (que A vea a B = que B vea a A), repartida
entre los n = 4 puntos:

$$p_{ij}=\frac{p_{j|i}+p_{i|j}}{2n}$$

```
p(A,B) = (1 + 1) / (2·4) = 2/8 = 0.25
p(C,D) = (1 + 1) / (2·4) = 2/8 = 0.25
pares cruzados (A-C, A-D, B-C, B-D) ≈ 0
```

| P    | A | B | C | D |
|:----:|:-:|:-:|:-:|:-:|
| A    | – | **0.25** | 0 | 0 |
| B    | **0.25** | – | 0 | 0 |
| C    | 0 | 0 | – | **0.25** |
| D    | 0 | 0 | **0.25** | – |

> **P es el "objetivo".** Dice literalmente: *"A va con B, C va con D, y entre
> parejas no exijo nada."* Fíjate que **no dice a qué distancia** poner las
> parejas entre sí — esa info no existe en P. (Guarda esta idea para el final.)

## Paso 4 — Afinidades en el mapa 2D (la t-Student) → Q

Ahora proponemos posiciones en el mapa, $y_i$, y medimos sus afinidades con una
**distribución t de Student** (campana de **colas pesadas**), no una gaussiana:

$$q_{ij}=\frac{\big(1+\|y_i-y_j\|^2\big)^{-1}}{\sum_{k\neq l}\big(1+\|y_k-y_l\|^2\big)^{-1}}$$

> **¿Por qué cola pesada y no otra gaussiana?** Es el truco que resuelve el
> *problema del hacinamiento* (*crowding*): en 2D no cabe tanta gente "cerca"
> como en alta dimensión. La cola pesada de la t deja que los puntos **no
> vecinos** se vayan **lejos** sin ser penalizados, dando aire a los grupos.

Probemos un mapa **bueno** (1D para simplificar), con las parejas juntas y
separadas entre sí: `y = [A:0, B:1, C:20, D:21]`.

Numerador $(1+d^2)^{-1}$:

```
dentro de pareja (d=1):   1/(1+1)   = 0.5
entre parejas   (d≈20):   1/(1+400) ≈ 0.0025
```

Al normalizar queda:

```
q(A,B) ≈ 0.247     q(C,D) ≈ 0.247     cruzados ≈ 0.001
```

**Q se parece muchísimo a P.** (0.247 ≈ 0.25 dentro de parejas; ≈0 entre ellas.)

## Paso 5 — Comparar P con Q y mover los puntos (KL + gradiente)

Medimos qué tan distintas son P y Q con la **divergencia de Kullback–Leibler**
(0 = idénticas):

$$\text{KL}(P\,\|\,Q)=\sum_{i\neq j} p_{ij}\,\log\frac{p_{ij}}{q_{ij}}$$

Con nuestro mapa **bueno**:  `KL ≈ 0.01`  → casi perfecto.

¿Y si proponemos un mapa **malo** que separa a A de B? `y = [A:0, B:20, C:1, D:21]`:

`KL ≈ 5.31`  → enorme. t-SNE "siente" que ese mapa está mal.

**El paso de optimización** mueve los $y_i$ para **minimizar KL**. La intuición es
de **resortes**, par por par:

```
si  p_ij > q_ij  →  el resorte JALA   (i y j deberían estar más cerca)
si  p_ij < q_ij  →  el resorte EMPUJA  (i y j deberían estar más lejos)
```

Se repite muchas iteraciones hasta que el mapa se "relaja". (La fórmula del
gradiente existe, pero *eso* es lo que hace la computadora; tú solo necesitas la
idea de jalar/empujar.)

## t-SNE en 5 pasos

```
1. Distancias:   matriz de distancias par a par en alta dimensión.
2. Afinidad P:   gaussiana → "probabilidad de ser vecinos" (σ ← perplexity).
3. Simetrizar:   p_ij = (p_{j|i} + p_{i|j}) / 2n.
4. Afinidad Q:   colocar puntos en 2D, medir con t-Student (cola pesada).
5. Minimizar KL: mover los puntos (jalar/empujar) hasta que Q ≈ P.
```

---

## Por qué esto explica los avisos del notebook

Las cuentas de arriba **demuestran** las 3 advertencias que ya tienes en las
celdas de t-SNE:

- **"Las distancias entre clusters no significan nada."** Lo viste en el Paso 3:
  la matriz **P nunca contiene** la distancia entre parejas (es ≈0 en todos los
  pares cruzados). Si esa información no entra, el mapa la inventa: puedes poner
  las parejas a distancia 20 o 200 y el KL casi no cambia.

- **"Es estocástico (corre dos veces, sale distinto)."** El Paso 5 arranca de
  posiciones **aleatorias** y solo pide que *vecinos queden juntos*. Hay muchos
  mapas con KL igual de bajo → cada corrida cae en uno distinto. Por eso fijas
  `random_state`.

- **"No tiene `transform` para datos nuevos."** P se calcula con **todos los
  puntos a la vez** (la normalización del Paso 2 mira a todos los vecinos). Un
  punto nuevo cambiaría todas las afinidades → hay que recalcular desde cero.

---

## PCA vs t-SNE, resumido en una cuenta

| | **PCA** | **t-SNE** |
|---|---|---|
| Qué calcula | Autovalores/autovectores de la **covarianza** | Afinidades **P** (gauss) y **Q** (t-Student) |
| Qué optimiza | Maximizar varianza proyectada | Minimizar **KL(P‖Q)** |
| ¿A mano de principio a fin? | **Sí** (álgebra lineal exacta) | No: el último paso es iterativo |
| Tipo | **Lineal** (ejes rectos) | **No lineal** (preserva vecindarios) |
| Conserva | Varianza **global** | Vecindarios **locales** |
| ¿`transform` a datos nuevos? | **Sí** | **No** |
| ¿Determinista? | Sí | **No** (fija `random_state`) |
| Distancias entre grupos | Significan algo | **No** confiar en ellas |

---

## Resumen

**PCA:** *"Centro los datos, calculo la matriz de covarianza, y le saco
autovalores y autovectores. Cada autovector es un eje nuevo; su autovalor me dice
cuánta varianza captura. Ordeno de mayor a menor, me quedo con los primeros y
proyecto. En mi ejemplo, un solo eje —el 'tamaño del pétalo'— conserva el 90 % de
la información."*

**t-SNE:** *"Convierto distancias en 'probabilidades de ser vecinos' con una
gaussiana: eso es P, mi objetivo. Luego coloco los puntos en 2D y mido sus
afinidades con una t de Student de colas pesadas: eso es Q. Muevo los puntos hasta
que Q se parezca a P (minimizo KL). Como P solo guarda quién es vecino de quién,
t-SNE preserva los grupos pero las distancias entre grupos no significan nada."*


---
