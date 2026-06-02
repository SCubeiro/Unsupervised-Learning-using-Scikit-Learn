# 🎤 Guion de la presentación (Canva)

> **Filosofía:** las diapositivas son el **calentamiento** — dan el **contexto técnico, curioso e histórico**
> y los **fundamentos** para entender el tema. La **parte fuerte es el notebook**, donde mostramos el código en vivo.
> Por eso las slides llevan **poco texto y muchas imágenes**: el que habla desarrolla, la diapositiva acompaña.

## Cómo usar este guion

- **En pantalla:** lo poco que va escrito/visual en la slide (máx. 3 ideas).
- **🎤 Guion:** lo que dice el presentador (no se escribe en la slide).
- **💡 Dato:** el gancho curioso/histórico para enganchar a la clase.
- **🖼️ Imagen:** qué visual buscar/poner.

## Diseño sugerido

- Un **color por bloque** (igual que el notebook): 🔵 azul = clustering · 🟢 verde = más clustering · 🟠 naranja = reducción de dimensión.
- Plantilla limpia, una idea por slide, tipografía grande.
- **Tiempo total de slides: ~10–12 min.** Luego el notebook (la demo) se lleva el grueso.

## Reparto (igual que el notebook)

| Bloque | Presentador | Slides |
| --- | --- | --- |
| Apertura + 🔵 Clustering/K-Means | **Presentador 1** | 1–6 |
| 🟢 DBSCAN + Jerárquico | **Presentador 2** | 7–9 |
| 🟠 PCA + t-SNE + cierre | **Presentador 3** | 10–16 |

---

## 🟦 APERTURA — *Presentador 1*

### Diapositiva 1 — Portada

- **En pantalla:** Título *"Aprendizaje No Supervisado con Scikit-Learn"* · los 3 nombres · materia.
- **🖼️ Imagen:** las flores de Iris, o un fondo de puntos formando grupos.
- **🎤 Guion:** *"Buenas, hoy les vamos a mostrar cómo una máquina puede encontrar patrones en datos sin que nadie le diga las respuestas. Lo basamos en la Lección 6 del curso de Machine Learning de freeCodeCamp."*

### Diapositiva 2 — Hoja de ruta

- **En pantalla:** La pregunta central: **"¿Puede una máquina agrupar datos sin etiquetas?"** + dos cajas: *Diapositivas = contexto* → *Notebook = demostración*.
- **🎤 Guion:** *"Primero les damos el contexto y la intuición con unas diapositivas cortas; la parte fuerte viene después, cuando abramos el notebook y lo corramos en vivo. Hoy respondemos dos preguntas: ¿qué grupos hay en los datos? y ¿cómo vemos datos de muchas dimensiones?"*

### Diapositiva 3 — ¿Qué es el aprendizaje no supervisado?

- **En pantalla:** Supervisado (con respuestas ✅) **vs** No supervisado (sin respuestas ❓). Dos objetivos: **agrupar** y **simplificar**.
- **🖼️ Imagen:** a la izquierda datos etiquetados por color; a la derecha los mismos puntos en gris (sin etiqueta).
- **💡 Dato:** *El término "análisis de clústeres" viene de los años 30, ¡de la psicología y la antropología, mucho antes de las computadoras modernas!*
- **🎤 Guion:** *"En el aprendizaje supervisado le damos ejemplos con la respuesta correcta. En el no supervisado le quitamos las respuestas: solo le damos los datos y le pedimos que descubra la estructura solo. Sirve para dos cosas: agrupar lo parecido, y resumir datos complejos."*

### Diapositiva 4 — El dataset: Iris 🌸

- **En pantalla:** 150 flores · 4 medidas · 3 especies. Foto de las 3 especies (setosa, versicolor, virginica).
- **💡 Dato:** *El dataset de Iris lo popularizó el estadístico **Ronald Fisher en 1936** (las flores las midió el botánico Edgar Anderson). ¡Tiene casi 90 años y sigue siendo el "hola mundo" del machine learning!*
- **🎤 Guion:** *"Vamos a usar Iris: medidas de pétalos y sépalos de 150 flores de 3 especies. El truco pedagógico: le ocultamos la especie al modelo, y al final destapamos la respuesta para ver si la adivinó solo."*

---

## 🔵 BLOQUE 1 — Clustering y K-Means — *Presentador 1*

### Diapositiva 5 — ¿Qué es clustering?

- **En pantalla:** "Agrupar lo parecido." 3 íconos de aplicaciones: 🎵 recomendaciones · 🛒 segmentar clientes · 🚨 detectar fraude.
- **🖼️ Imagen:** una nube de puntos que se separa en 3 grupos de colores.
- **🎤 Guion:** *"Clustering es agrupar observaciones parecidas sin saber de antemano los grupos. Lo usan Spotify para recomendarte música, las tiendas para segmentar clientes, y los bancos para detectar transacciones raras."*

### Diapositiva 6 — K-Means

- **En pantalla:** "Elige K centros → asigna cada punto al más cercano → repite." Una imagen de centroides.
- **🖼️ Imagen:** GIF/diagrama de centroides moviéndose hasta estabilizarse.
- **💡 Dato:** *El algoritmo lo creó **Stuart Lloyd en los Bell Labs en 1957**… ¡pero lo dejaron como reporte interno y no se publicó hasta 1982! El nombre "k-means" se lo puso James MacQueen en 1967.*
- **🎤 Guion:** *"K-Means es el más clásico: tú eliges cuántos grupos quieres, él pone unos centros y va moviendo cada punto al centro más cercano hasta estabilizarse. Un uso curioso: comprimir imágenes, reduciéndolas a K colores. Y esto… mejor véanlo funcionando — vamos al notebook en un momento."*

---

## 🟢 BLOQUE 2 — DBSCAN y Jerárquico — *Presentador 2*

### Diapositiva 7 — K-Means tiene un límite

- **En pantalla:** Dos "lunas" entrelazadas; K-Means las parte mal. **"¿Y si los grupos no son redondos?"**
- **🖼️ Imagen:** el clásico *two-moons*: a la izquierda mal cortado por K-Means.
- **🎤 Guion:** *"K-Means asume que los grupos son como pelotas redondas. Pero si los datos tienen forma de luna, los parte mal. Para eso existen otros algoritmos."*

### Diapositiva 8 — DBSCAN

- **En pantalla:** "Agrupa por **densidad**." Núcleo / borde / **ruido (outlier)**. No hay que decirle cuántos grupos.
- **🖼️ Imagen:** puntos densos formando clusters + puntos sueltos marcados como ruido.
- **💡 Dato:** *DBSCAN es de **1996** (Ester, Kriegel, Sander y Xu). En 2014 ganó el premio "Test of Time" — un reconocimiento a los trabajos que siguen siendo influyentes décadas después.*
- **🎤 Guion:** *"DBSCAN agrupa por densidad: donde hay muchos puntos juntos, hay un grupo; lo que queda aislado es ruido. Dos ventajas enormes: no le dices cuántos grupos hay, y detecta outliers solo. Por eso se usa muchísimo en detección de fraude y datos de GPS."*

### Diapositiva 9 — Clustering jerárquico

- **En pantalla:** Un **dendrograma** (árbol). "Fusiona lo más parecido hasta tener un solo grupo." Eliges cuántos grupos **cortando el árbol**.
- **🖼️ Imagen:** un dendrograma claro.
- **💡 Dato:** *El dendrograma viene de la **biología**: es el mismo tipo de árbol que usan los biólogos para dibujar la evolución de las especies (de hecho "dendro" = árbol en griego).*
- **🎤 Guion:** *"El jerárquico arma un árbol: empieza con cada punto solo y va fusionando los más parecidos hasta que queda uno. Lo bonito es que no eliges el número de grupos al inicio: cortas el árbol a la altura que tenga sentido. Lo verán dibujado en el notebook."*

---

## 🟠 BLOQUE 3 — PCA y t-SNE — *Presentador 3*

### Diapositiva 10 — El reto de las muchas dimensiones

- **En pantalla:** "Iris tiene 4 columnas. Las imágenes, cientos. **No podemos dibujar en 4D.**" → reducir a 2D.
- **🖼️ Imagen:** un cubo 3D proyectando su sombra 2D en el piso.
- **🎤 Guion:** *"Hasta ahora graficamos solo 2 de las 4 medidas. Pero ¿cómo vemos las 4 a la vez? ¿O las 64 de una imagen? No podemos dibujar en 4 dimensiones, así que las resumimos a 2 perdiendo lo menos posible."*

### Diapositiva 11 — PCA

- **En pantalla:** "Encuentra los 2 ejes con **más variación**." Imagen de proyección 3D→2D.
- **🖼️ Imagen:** nube de puntos alargada con una flecha en la dirección de máxima dispersión.
- **💡 Dato:** *PCA lo inventó **Karl Pearson en 1901** — ¡tiene más de 120 años! Curiosidad moderna: si le aplicas PCA al ADN de europeos, el resultado dibuja casi el mapa de Europa. "Los genes reflejan la geografía."*
- **🎤 Guion:** *"PCA busca el mejor ángulo para 'fotografiar' los datos: los 2 ejes nuevos que capturan la mayor variación. Es como elegir la sombra que más información conserva. Es la técnica más veterana de todas, y se usa hasta en reconocimiento facial."*

### Diapositiva 12 — t-SNE

- **En pantalla:** "Mantiene cerca a los **vecinos**." Imagen de dígitos separados en 10 grupos.
- **🖼️ Imagen:** el famoso mapa t-SNE de MNIST (los 10 dígitos en islas de colores).
- **💡 Dato:** *t-SNE es reciente: **2008**, de Laurens van der Maaten y **Geoffrey Hinton** (uno de los "padres" del deep learning, premio Turing 2018). Hay PCA de 1901 y t-SNE de 2008: ¡un siglo de diferencia! Su sucesor moderno se llama UMAP.*
- **🎤 Guion:** *"t-SNE es no lineal y sirve sobre todo para ver: coloca los puntos en 2D cuidando que los vecinos sigan juntos. Es lo que se usa para visualizar lo que 'aprenden' las redes neuronales. En el notebook verán cómo separa los 10 dígitos donde PCA no puede."*

---

## 🟪 CIERRE — *Presentador 3 (con apoyo de los 3)*

### Diapositiva 13 — ¿Qué algoritmo cuándo?

- **En pantalla:** tabla simple:

  | Quiero… | Uso… |
  | --- | --- |
  | Grupos y sé cuántos | K-Means |
  | Formas raras / hay ruido | DBSCAN |
  | Ver la jerarquía | Jerárquico |
  | Comprimir / acelerar | PCA |
  | Solo visualizar alta dimensión | t-SNE |

- **🎤 Guion:** *"En resumen, cada herramienta para su problema: esta tablita es la chuleta para elegir."*

### Diapositiva 14 — 🔴 Ahora la parte fuerte: el notebook

- **En pantalla:** "👉 Pasamos al notebook" + captura del notebook.
- **🎤 Guion:** *"Hasta aquí el contexto. Ahora lo importante: vamos a abrir el notebook y correr todo esto en vivo, paso a paso, viendo el código y los resultados."* **(Cambiar a Jupyter y exponer el notebook por bloques, cada quien el suyo.)**

> 🔁 **Aquí se expone el notebook** (la demostración técnica). Cada presentador corre su bloque. Al volver, las 2 últimas slides cierran.

### Diapositiva 15 — 3 ideas para llevarse

1. No supervisado = **sin etiquetas**: el modelo descubre solo.
2. **Mismo patrón para todo:** `.fit()` → `.predict()`/`.labels_` o `.transform()`.
3. **Evaluar es difícil:** no hay "accuracy"; pudimos validar solo porque Iris traía las especies.

- **🎤 Guion:** *"Tres ideas: el modelo trabaja sin respuestas; todos los algoritmos de scikit-learn se usan casi igual; y medir qué tan bien lo hizo es lo difícil del no supervisado."*

### Diapositiva 16 — Gracias + referencias

- **En pantalla:** "¿Preguntas?" + link del curso de freeCodeCamp + scikit-learn.org.
- **🎤 Guion:** *"Gracias, ¿alguna pregunta?"*

---

## ⏱️ Tips de exposición

- **Menos texto en pantalla, más en tu boca:** la slide es un apoyo, no un teleprompter.
- **Ensayen la transición al notebook** (slide 14): que el cambio a Jupyter sea fluido y el kernel ya esté corriendo.
- **Cada quien abre su bloque** leyendo el recuadro 🎯 *Problema* / 🛠️ *Idea* del notebook, y lo cierra con el ✅ *Qué demostramos*.
- **Tiempo:** slides ~10–12 min · notebook ~12–15 min · preguntas ~3 min.
