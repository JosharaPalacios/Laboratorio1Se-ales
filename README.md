# Laboratorio 1 - Análisis Estadístico de la Señal
**Universidad Militar Nueva Granada**  
**Asignatura:** Procesamiento Digital de Señales  
**Estudiantes:** [Maria Jose Peña, Joshara Valentina Palacios, Lina Marcela Pabuena]  
**Fecha:** Agosto 2025  
**Asignatura:** Procesamiento Digital de Señales  
**Título de la práctica:** Análisis estadístico de la señal 

## Objetivos 
- Identificar los estadísticos que describen una señal biomédica.  
- Obtener dichos estadísticos a partir de algoritmos programados en Python.  
- Comparar el cálculo de estadísticos hecho de forma manual (programando las fórmulas) con el uso de funciones predefinidas.  
- Importar, graficar y manipular señales fisiológicas en Python utilizando librerías como *matplotlib*.  
- Generar y capturar señales fisiológicas con (STM32 o DAQ).  
- Analizar el efecto del ruido en las señales mediante el cálculo de la relación señal-ruido (SNR).  
- Utilizar GitHub como herramienta de documentación y colaboración para reportar los resultados.  


# Análisis Estadístico de la Señal

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
<pre> ```
python
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

## Gráfica de la señal ECG

<p align="center">
<img width="758" height="540" alt="señal11" src="https://github.com/user-attachments/assets/a300c2d4-4562-4c9f-bf4d-69e2726b4d73" />
``` </pre>

## Interpretación de la gráfica

La gráfica anterior muestra un segmento de la señal fisiológica descargada desde la base de datos **PhysioNet**.  

- En el **eje X** se representa el tiempo (segundos).  
- En el **eje Y** se muestra el voltaje (microvoltios).  

Se observan **picos pronunciados** que corresponden a los **complejos QRS** del electrocardiograma, los cuales presentan una amplitud considerable en comparación con el resto de la señal.  

Entre los picos, se distinguen zonas de menor variación que representan las **ondas P y T**.  

Asimismo, se evidencia la presencia de **ruido** y variaciones irregulares en la línea base, lo cual es característico de señales biomédicas reales. Este ruido puede deberse a factores externos, **interferencias eléctricas** o **movimiento del paciente**.  
