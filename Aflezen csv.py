# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 13:38:28 2024

@author: Stein
"""

#importeer de benodigde libs
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
import pandas as pd

#importeer je csv file
df = pd.read_csv("C:/Users/Dell/Downloads/test-1-v2.csv")
#print(df.head())
#print(df.columns)

#Geef de columns een naam
t = df['t']
x = df['x']
y = df['y']
z = df['z']

#plot rauwe data
plt.figure(figsize=(18, 16))
plt.plot(t, x)
plt.title('tril tijdomijn')
plt.xlabel('Tijd [ms]')
plt.ylabel('Versnelling [m/s^2]')
plt.grid(True)
plt.show()



