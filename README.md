# Análisis de Engagement con Cadenas de Markov

Este proyecto presenta un dashboard interactivo para analizar y predecir los niveles de engagement de empleados utilizando Cadenas de Markov. La aplicación permite a los usuarios filtrar datos, visualizar las probabilidades de transición entre diferentes estados de engagement y simular escenarios futuros.

## 📜 Descripción

El objetivo principal es modelar la dinámica del engagement de los empleados a lo largo del tiempo. Utilizando un enfoque basado en Cadenas de Markov, el proyecto permite cuantificar cómo los empleados transitan entre diferentes niveles de engagement (por ejemplo, de "Bajo" a "Medio", o de "Alto" a "Medio"). El dashboard proporciona una interfaz intuitiva para explorar estos análisis.

## 📂 Estructura del Proyecto

El repositorio está organizado de la siguiente manera:

-   **/data**: Contiene los conjuntos de datos utilizados en el análisis.
    -   `data_globant.csv`: Datos brutos iniciales.
    -   `Final_Data.csv`: Datos procesados y listos para el análisis.
-   **/scripts**: Incluye los scripts y notebooks utilizados para el procesamiento de datos, análisis exploratorio (EDA) y la definición de funciones del modelo.
    -   `EDA.py`: Script para el análisis exploratorio de los datos.
    -   `Procesamiento.ipynb`: Notebook con el paso a paso del preprocesamiento de datos.
    -   `Model_Functions.py`: Módulo con las funciones principales para el modelo de Markov.
    -   `Final_Model.ipynb`: Notebook que integra las funciones y realiza el modelado final.
-   **/dashboard**: Contiene la aplicación web interactiva.
    -   `app.py`: Script principal de la aplicación Streamlit.
    -   `Final_Data.csv`: Copia de los datos procesados para que la aplicación sea autocontenida.

## ✨ Características Principales

El dashboard interactivo ofrece las siguientes funcionalidades:

-   **Filtros Dinámicos**: Permite segmentar los datos por diferentes categorías (como área, nivel de seniority, etc.) para un análisis más granular.
-   **Matriz de Transición**: Visualiza la matriz de probabilidades que muestra la probabilidad de pasar de un estado de engagement a otro en un solo paso de tiempo.
-   **Transiciones de N-Pasos**: Calcula y muestra la matriz de transición después de un número `n` de pasos (días), permitiendo hacer proyecciones a futuro.
-   **Simulación de Montecarlo**: Evalúa la precisión del modelo predictivo mediante la simulación de múltiples trayectorias de engagement.
-   **Simulación de Paseo Aleatorio (Random Walk)**: Simula y grafica una posible trayectoria futura del estado de engagement de un empleado a partir de un estado inicial.

## 🚀 Cómo Empezar

Sigue estos pasos para configurar y ejecutar el proyecto en tu entorno local.

### Prerrequisitos

-   Python 3.8 o superior
-   `pip` (manejador de paquetes de Python)

### Instalación

1.  **Clona el repositorio:**
    ```bash
    git clone https://github.com/tu-usuario/Globant_Reto.git
    cd Globant_Reto
    ```

2.  **Crea y activa un entorno virtual:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # En Windows usa: .venv\Scripts\activate
    ```

3.  **Instala las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

## 📈 Uso

Para iniciar el dashboard interactivo, ejecuta el siguiente comando en la raíz del proyecto:

```bash
streamlit run dashboard/app.py
```

Esto abrirá una nueva pestaña en tu navegador web con la aplicación en funcionamiento. Desde allí, podrás interactuar con los filtros y visualizaciones.

## 🛠️ Tecnologías Utilizadas

-   **Lenguaje**: Python
-   **Dashboard**: Streamlit
-   **Análisis de Datos**: Pandas, NumPy
-   **Modelado**: Scikit-learn
-   **Visualización**: Matplotlib, Seaborn
