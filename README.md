# Análisis Estadístico de la Señal

**Asignatura:** Procesamiento Digital de Señales  
**Título de la práctica:** Análisis estadístico de la señal  

## Procedimiento y método

Las señales medidas de un entorno real, como las biomédicas, contienen:
- Información relevante: amplitud y frecuencia.
- Información contaminante: ruido.

En esta práctica:
1. Se descargó una señal fisiológica desde [PhysioNet](https://physionet.org/), una base de datos pública de señales biomédicas.
2. Se importó la señal en Python y se graficó usando `matplotlib` para su visualización.
3. Se calcularon estadísticos descriptivos de dos maneras:
   - **Método manual:** programando las fórmulas desde cero.
   - **Funciones predefinidas:** usando librerías de Python.
4. Se analizaron los siguientes parámetros:
   - **Media de la señal**: indica el valor promedio.
   - **Desviación estándar**: mide cuánto varían los valores respecto a la media.
   - **Coeficiente de variación**: relación entre la desviación estándar y la media (porcentaje de variabilidad).
   - **Histograma de la señal**: muestra la distribución de valores de voltaje.
   - **Función de probabilidad**: describe la probabilidad de que la señal tome determinados valores.
   - **Curtosis**: mide el grado de concentración de los valores de la señal en torno a la media (si tiene colas más o menos pesadas que una distribución normal).

## Código en Python (Google Colab)

```python
# Importación de librerías necesarias
import numpy as np                  # Para manejo de arreglos y cálculos matemáticos
import matplotlib.pyplot as plt     # Para graficar la señal
!pip install wfdb                   # Instalación de la librería wfdb para leer señales biomédicas
import wfdb                          # Librería para trabajar con datos en formato PhysioNet
from scipy.stats import gaussian_kde, kurtosis # Para análisis estadístico adicional

# Montar Google Drive para acceder a los archivos
from google.colab import drive
drive.mount('/content/drive')

# Ruta del archivo descargado desde PhysioNet
record_name="/content/drive/MyDrive/GITHUB/100001_ECG"

# Lectura de la señal (señal1 contiene los datos, campos contiene información de cabecera)
señal1, campos = wfdb.rdsamp(record_name)
señal1

# Graficar la señal
plt.plot(señal1)                     # Dibuja la señal en función del tiempo
plt.xlabel("Tiempo(s)")              # Etiqueta del eje X
plt.ylabel("Voltaje(V)")             # Etiqueta del eje Y
plt.axis([3.5666e7,3.5668e7,-1500,2000]) # Límite de ejes para enfocar la región de interés
plt.grid()                           # Mostrar cuadrícula para mejor lectura
plt.show()
