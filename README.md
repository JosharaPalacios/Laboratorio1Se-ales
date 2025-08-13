# Laboratorio 1 Señales
## Parte A
import numpy as np

import matplotlib.pyplot as plt

!pip install wfdb # Instalación en colab

## pip install wfdb # Instalación en python instalado
import wfdb

from scipy.stats import gaussian_kde

from google.colab import drive

drive.mount('/content/drive')

record_name="/content/drive/MyDrive/GITHUB/100001_ECG"

señal1, campos = wfdb.rdsamp(record_name)

señal1

plt.plot(señal1)

plt.xlabel("Tiempo(s)")

plt.ylabel("Voltaje(V)")

plt.axis([3.5666e7,3.5668e7,-1500,2000])

plt.grid()

plt.show()
