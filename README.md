# Unsupervised Learning using Scikit-Learn

> Proyecto de la materia **Desarrollo de Aplicaciones para Análisis de Datos** — Licenciatura en
> Ciencia de Datos. Recreación y explicación de las principales técnicas de **aprendizaje no
> supervisado** con [scikit-learn](https://scikit-learn.org/).

## Acerca del proyecto

A partir del notebook `sklearn-unsupervised-learning.ipynb`, el proyecto recorre —con código
ejecutable y visualizaciones— las técnicas más comunes de aprendizaje no supervisado:

- **Introducción** al aprendizaje no supervisado.
- **Clustering** (agrupamiento): K-Means, DBSCAN y clustering jerárquico.
- **Reducción de dimensionalidad**: PCA y t-SNE.
- **Resumen** y referencias.

> **El entregable son dos piezas:** las **diapositivas (Canva)** dan el contexto técnico/histórico, y el **notebook propio**
> (`presentacion-aprendizaje-no-supervisado.ipynb`) es la demostración técnica en vivo. El guion de las diapositivas está en
> `guion-presentacion-canva.md`.

## Contenido del repositorio

| Archivo / Carpeta | Descripción |
| --- | --- |
| `presentacion-aprendizaje-no-supervisado.ipynb` | **Notebook de la presentación** (versión propia, enriquecida y pedagógica). Entregable principal. |
| `guion-presentacion-canva.md` | Guion diapositiva por diapositiva para la presentación en Canva. |
| `sklearn-unsupervised-learning.ipynb` | Notebook **original del video** (Lección 6), conservado como referencia. |
| `requirements.txt` | Dependencias con sus versiones exactas (reproducibilidad). |
| `Ambiente/Ambiente.md` | Explicación de cada comando del ambiente + capturas del proceso. |
| `Ambiente/Imagenes/` | Capturas de pantalla usadas en la explicación. |

## Requisitos previos

| Herramienta | macOS | Windows |
| --- | --- | --- |
| **Git** | Incluido con las *Command Line Tools* de Xcode, o `brew install git` | Descargar de [git-scm.com](https://git-scm.com/download/win) |
| **Python 3.12** | `brew install python@3.12` | Descargar de [python.org](https://www.python.org/downloads/) y marcar **"Add Python to PATH"** al instalar |

Para comprobar que están instalados:

```bash
git --version
python3 --version   # en Windows:  py --version
```

## Reproducir el proyecto (Mac y Windows)

> Los comandos son los mismos en ambos sistemas; **lo único que cambia es crear y activar el
> entorno virtual** (Paso 2). Sigue la columna que corresponda a tu sistema.

### 1. Clonar el repositorio

```bash
git clone https://github.com/SCubeiro/Unsupervised-Learning-using-Scikit-Learn.git
cd Unsupervised-Learning-using-Scikit-Learn
```

### 2. Crear y activar el entorno virtual

**macOS / Linux** (terminal `bash` o `zsh`):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

**Windows** (PowerShell):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

> **Tip Windows:** si PowerShell muestra el error *"running scripts is disabled on this system"*,
> ejecuta una sola vez en esa ventana:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` y vuelve a activar.
> Alternativas: **CMD** → `.venv\Scripts\activate.bat` · **Git Bash** → `source .venv/Scripts/activate`

Con el entorno activo verás el prefijo **`(.venv)`** al inicio de la línea.

### 3. Instalar las dependencias

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> `requirements.txt` fue congelado en macOS. Si en Windows algún paquete diera problema, instala el
> stack principal directamente (funciona igual en ambos sistemas):
>
> ```bash
> pip install ipykernel jupyter numpy pandas matplotlib seaborn scikit-learn plotly
> ```

### 4. Registrar el kernel en Jupyter

```bash
python -m ipykernel install --user --name=unsupervised-learning-sklearn --display-name "Unsupervised Learning using Scikit-Learn"
```

### 5. Verificar que todo funciona

```bash
python -c "import numpy, pandas, sklearn, seaborn, plotly; print('OK')"
```

Si imprime `OK`, el ambiente quedó listo.

### 6. Abrir el notebook

```bash
jupyter lab        # o:  jupyter notebook
```

Abre **`presentacion-aprendizaje-no-supervisado.ipynb`** (el notebook de la presentación) y, en el selector de kernel,
elige el de este proyecto — **"Unsupervised Learning using Scikit-Learn"** o **"Python 3"** del entorno `.venv`.
*(También puedes abrir la carpeta en **VS Code** y seleccionar ese mismo kernel.)*

## Explicación del ambiente (con capturas)

La explicación detallada de **qué hace cada comando**, junto con las capturas del proceso, está en
**[`Ambiente/Ambiente.md`](Ambiente/Ambiente.md)**.

---

*Materia: Desarrollo de Aplicaciones para Análisis de Datos · Licenciatura en Ciencia de Datos.*
