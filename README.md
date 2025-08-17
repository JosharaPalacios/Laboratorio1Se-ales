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

# PARTE A
## Análisis Estadístico de la Señal

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

# Código en Python (Google Colab)
## Descarga y grafica de la señal con funciones
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
``` </pre>

## Gráfica de la señal ECG

<p align="center">
<img width="758" height="540" alt="señal11" src="https://github.com/user-attachments/assets/a300c2d4-4562-4c9f-bf4d-69e2726b4d73" />


## Interpretación de la gráfica

La gráfica anterior muestra un segmento de la señal fisiológica descargada desde la base de datos **PhysioNet**.  

- En el **eje X** se representa el tiempo (segundos).  
- En el **eje Y** se muestra el voltaje (microvoltios).  

Se observan **picos pronunciados** que corresponden a los **complejos QRS** del electrocardiograma, los cuales presentan una amplitud considerable en comparación con el resto de la señal.  

Entre los picos, se distinguen zonas de menor variación que representan las **ondas P y T**.  

Asimismo, se evidencia la presencia de **ruido** y variaciones irregulares en la línea base, lo cual es característico de señales biomédicas reales. Este ruido puede deberse a factores externos, **interferencias eléctricas** o **movimiento del paciente**. 
### Resultado: Señal fisiológica descargada

La gráfica corresponde a un segmento de una señal electrocardiográfica (ECG) obtenida desde la base de datos PhysioNet.  
El archivo fue descargado, subido a Google Drive y cargado en Python, donde fue almacenado en la variable **señal1**.  

Dado que la señal original contenía una gran cantidad de muestras, se realizó un ajuste de los ejes para **ampliar una ventana específica** y facilitar la visualización de las ondas características.  

En la figura se observan claramente los complejos QRS, con amplitudes que alcanzan valores cercanos a **±1500 µV**. Entre los picos principales se identifican segmentos isoeléctricos con variaciones menores, lo que corresponde al comportamiento normal de la señal entre cada latido.  

   
## Análisis estadístico de la señal

A partir de la señal fisiológica importada en la variable `señal1`, se calcularon los principales estadísticos descriptivos. 
La **media** permite identificar el valor promedio de la señal, mientras que la **desviación estándar** muestra el grado de variabilidad de los datos respecto a dicho promedio. 
El **coeficiente de variación** se empleó para normalizar la dispersión en relación con la magnitud de la señal, facilitando su comparación con otras señales. 
Adicionalmente, se generó el **histograma**, que permite visualizar la distribución de los valores, y la **función de probabilidad**, que describe la tendencia de ocurrencia de los datos. 
Finalmente, se obtuvo la **curtosis**, con el fin de analizar si la distribución presenta una concentración mayor o menor alrededor de la media.

<pre> ```
python
   # Media de la señal
media = np.mean(señal1)
print(f"Media= {media}")

# Desviación estándar
desviacion_muestra = np.std(señal1,ddof=1)
print(f"Desviacion estandar de la muestra = {desviacion_muestra}")

# Coeficiente de variación
coeficiente_variacion = np.std(señal1, ddof=0) / np.mean(señal1) * 100
print(f"Coefiiente de variación = {coeficiente_variacion}")

# Histograma
plt.hist(señal1,bins=100)
plt.title("Histograma de ECG")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Voltaje (mV)")
plt.grid()
plt.show()

# Función de probabilidad  
data = np.ravel(señal1)  # Convierte cualquier forma a un arreglo plano

# Crear KDE
kde = gaussian_kde(data)

# Rango de valores para evaluar la KDE
x_vals = np.linspace(min(data), max(data), 1000)

# Graficar
plt.plot(x_vals, kde(x_vals))
plt.xlabel("Valor de la señal")
plt.ylabel("Densidad de probabilidad")
plt.title("Estimación de densidad (KDE) de señal1")
plt.show()

# Curtosis
n=len(señal1)
curtosis=np.sum(((señal1-media)/desviacion_muestra)**4)/n
print(f"Curtosis = {curtosis}")
``` </pre>
## Resultados del analisis estadisticos 
## Histograma
<img width="686" height="560" alt="Captura de pantalla 2025-08-16 001720" src="https://github.com/user-attachments/assets/c63a407c-0d26-4663-a75c-06a09b496e3f" />

## Resultados numéricos

- **Media:** 0.21505595733362035 
- **Desviación estándar:** 461.2037350921682  
- **Coeficiente de variación:** 214457.54777615666  
- **Curtosis:** 122.85882529006771  

## Función de probabilidad 
<img width="732" height="565" alt="Captura de pantalla 2025-08-16 002834" src="https://github.com/user-attachments/assets/fb765971-eac3-465b-b67e-e6917cedbbcb" />

## Descarga y grafica de la señal sin funciones
<pre> ```
   senal = list(señal1)
n = len(senal)

# Media
suma = 0
for x in senal:
    suma += x
media = suma / n

# Desviación estándar
suma_cuadrados = 0
for x in senal:
    suma_cuadrados += (x - media) ** 2
desv_std = math.sqrt(suma_cuadrados / n)

# Coeficiente de variación
coef_var = (desv_std / media) * 100

# Curtosis
suma_curt = 0
for x in senal:
    suma_curt += (x - media) ** 4
curtosis = (suma_curt / n) / (desv_std ** 4)

# Mostrar resultados
print("Media:", media)
print("Desviación estándar:", desv_std)
print("Coeficiente de variación (%):", coef_var)
print("Curtosis:", curtosis)

# Histograma
plt.figure(figsize=(8,4))
plt.hist(senal, bins=20, edgecolor='black')
plt.title("Histograma de la señal")
plt.xlabel("Valor")
plt.ylabel("Frecuencia")
plt.show()

# Función de probabilidad (PMF)
valores_unicos = sorted(set(senal))
pmf = []
for v in valores_unicos:
    frecuencia = 0
    for x in senal:
        if x == v:
            frecuencia += 1
    pmf.append(frecuencia / n)

print("\nFunción de probabilidad (PMF):")
for v, p in zip(valores_unicos, pmf):
    print(f"Valor: {v}  Probabilidad: {p}")

plt.figure(figsize=(8,4))
plt.stem(valores_unicos, pmf, use_line_collection=True)
plt.title("Función de probabilidad (PMF)")
plt.xlabel("Valor")
plt.ylabel("Probabilidad")
plt.show()
``` </pre>
## Resultados del analisis estadisticos sin funciones

### Resultados numéricos

- **Media:** 0.21505595733362035 
- **Desviación estándar:** 461.2037350921682  
- **Coeficiente de variación:** 214457.54777615666  
- **Curtosis:** 122.85882529006771  

## Análisis de resultados – Parte A

En la Parte A se trabajó con una señal fisiológica obtenida desde la base de datos PhysioNet. 
Tras su importación en Python y su almacenamiento en la variable `señal1`, se realizó un recorte de la señal para mejorar la visualización de sus componentes característicos, dado que la longitud original era demasiado extensa.

Al calcular los estadísticos descriptivos mediante dos métodos (programación manual de las fórmulas y uso de funciones predefinidas de librerías como *NumPy* y *SciPy*), se obtuvieron resultados consistentes entre ambos enfoques, lo cual valida la implementación realizada.  

Los valores de la **media** y la **desviación estándar** mostraron que la señal oscila alrededor de un promedio cercano a cero, con una dispersión significativa debida a la presencia de los picos de los complejos QRS. 
El **coeficiente de variación** evidenció un alto grado de variabilidad relativa, lo que es coherente con la naturaleza no estacionaria de las señales biomédicas.  

El **histograma** permitió visualizar la distribución de amplitudes, donde la mayor parte de los datos se concentra alrededor de valores cercanos a cero, mientras que los picos de alta amplitud aparecen con menor frecuencia. 
La **función de probabilidad** confirmó esta tendencia al mostrar que las amplitudes extremas tienen baja probabilidad de ocurrencia. 
Finalmente, la **curtosis** indicó que la distribución es más apuntada que una normal estándar, reflejando la presencia de valores extremos en la señal.

# PARTE B

En esta parte del laboratorio nos dirigimos al laboratorio y realizamos la adquisición de una señal fisiológica utilizando un **generador de señales** conectado a un sistema de adquisición de datos (**DAQ**).  
La señal fue capturada y exportada en formato `.csv`, lo que permitió almacenarla y procesarla posteriormente en **Google Colab** mediante Python.  

# Código en Python (Google Colab)

## Descarga de la señal
 <pre> ```
python
    
from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
ruta = "/content/drive/MyDrive/GITHUB/medicion1.csv"
df = pd.read_csv(ruta)
df.head()
``` </pre>
## Almacenamiento de la señal en una varibale
 <pre> ```
python
tiempo = df.iloc[:,0].values
senal2 = df.iloc[:,1].values
``` </pre>
## Graficas y datos estadisticos por medio de funciones 
 <pre> ```
python
    
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
import seaborn as sns
    
plt.figure(figsize=(10,4))
plt.plot(senal2)
plt.title("Señal fisiológica medida en laboratorio")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()

media = np.mean(senal2)
desv = np.std(senal2)
cv = desv / media
curt = kurtosis(senal2)

print("Media:", media)
print("Desviación estándar:", desv)
print("Coeficiente de variación:", cv)
print("Curtosis:", curt)

# Histograma
plt.hist(senal2, bins=50, density=True, alpha=0.7, color='blue')
plt.title("Histograma de la señal medida")
plt.xlabel("Amplitud")
plt.ylabel("Frecuencia")
plt.show()


# Estimación de la densidad de probabilidad (PDF)
plt.figure(figsize=(8,4))
sns.kdeplot(senal2, fill=True, color="red", alpha=0.6)
plt.title("Función de probabilidad de la señal")
plt.xlabel("Amplitud")
plt.ylabel("Densidad de probabilidad")
plt.grid(True)
plt.show()
``` </pre>
## Resultados, gráfica de la señal 
<img width="1068" height="486" alt="Captura de pantalla 2025-08-16 235352" src="https://github.com/user-attachments/assets/d1032b27-280d-48b5-b40d-6c9eb0e07bab" />

## Resultados estadisticos de la señal fisiologica 

- Media: 1.219676066378888
- Desviación estándar: 0.4011725172544121
- Coeficiente de variación: 0.3289172660782452
- Curtosis: 4.689155028469519

## Histograma
<img width="692" height="556" alt="Captura de pantalla 2025-08-16 235301" src="https://github.com/user-attachments/assets/16286fc9-2c26-4497-b59e-bed6386c8aea" />

## Función de probabilidad
<img width="863" height="488" alt="Captura de pantalla 2025-08-16 235421" src="https://github.com/user-attachments/assets/4a3b16be-dd19-4d60-9b0b-ba242ce5cc8d" />

## Graficas y datos estadisticos por medio de funciones 
 <pre> ```
python
    import matplotlib.pyplot as plt

n = len(senal2)


# Media (valor promedio)

suma = 0
for x in senal2:
    suma += x
media_manual = suma / n

# Desviación estándar

suma_cuadrados = 0
for x in senal2:
    suma_cuadrados += (x - media_manual)**2
desv_manual = (suma_cuadrados / (n-1))**0.5   # uso n-1 para ser insesgado


# Coeficiente de variación

cv_manual = desv_manual / media_manual if media_manual != 0 else float("inf")


# Curtosis

suma_cuarta = 0
for x in senal2:
    suma_cuarta += (x - media_manual)**4
curtosis_manual = (suma_cuarta / n) / (desv_manual**4)


# Resultados

print("Media (manual):", media_manual)
print("Desviación estándar (manual):", desv_manual)
print("Coeficiente de variación (manual):", cv_manual)
print("Curtosis (manual):", curtosis_manual)

# Histograma

plt.hist(senal2, bins=50, color="skyblue", edgecolor="black")
plt.title("Histograma de la señal (manual)")
plt.xlabel("Amplitud")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.show()

# Función de probabilidad 

import numpy as np

conteo, bordes = np.histogram(senal2, bins=50, density=True)
centros = (bordes[:-1] + bordes[1:]) / 2

plt.plot(centros, conteo, marker="o", color="red")
plt.title("Función de probabilidad (manual)")
plt.xlabel("Amplitud")
plt.ylabel("Probabilidad")
plt.grid(True)
plt.show()
``` </pre>





