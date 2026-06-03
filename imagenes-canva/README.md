# 🖼️ Imágenes para Canva

Gráficas exportadas automáticamente desde `presentacion-aprendizaje-no-supervisado.ipynb`.
Arrastra el archivo indicado a la diapositiva correspondiente del guion (`guion-presentacion-canva.md`).

## Principales (una por diapositiva)

| Slide | Archivo | Qué muestra |
| --- | --- | --- |
| 4 | `slide04-iris-especies-reales.png` | Las 150 flores de Iris coloreadas por especie real (setosa / versicolor / virginica). |
| 6 | `slide06-kmeans-centroides.png` | K-Means: 3 clusters hallados + centroides (las **X** negras). |
| 7 | `slide07-dos-lunas-kmeans-vs-dbscan.png` | *Two-moons*: K-Means corta recto (mal) vs DBSCAN sigue la forma (bien). |
| 8 | `slide08-dbscan-iris-ruido.png` | DBSCAN aplicado a Iris. |
| 9 | `slide09-dendrograma.png` | Dendrograma de Iris (corte rojo → 3 clusters). |
| 11 | `slide11-pca-2d.png` | PCA: 4D → 2D, coloreado por especie real. |
| 12 | `slide12-pca-vs-tsne-digitos.png` | Dígitos 64-D: PCA los encima vs t-SNE separa 10 grupos. |

## Extras (opcionales)

| Sugerida en | Archivo | Qué muestra |
| --- | --- | --- |
| 6 | `slide06-extra-codo-silueta.png` | Método del codo (inercia) + silueta vs K. |
| 6 | `slide06-extra-kmeans-k6-sobreajuste.png` | K=6: parte de más grupos que en realidad son uno. |
| 11 | `slide11-extra-pca-scree-varianza.png` | Scree plot: varianza explicada por componente. |
| 12 | `slide12-extra-digitos-muestras.png` | Tira de dígitos 8×8 (qué es un punto de 64-D). |
| 12 | `slide12-extra-tsne-iris.png` | t-SNE sobre Iris (alternativa al de dígitos). |

> **Resolución:** ~100 dpi — buena para colocarlas a tamaño medio en la slide.
> Si las quieres nítidas a pantalla completa, se pueden re-renderizar a mayor DPI re-ejecutando el notebook.
