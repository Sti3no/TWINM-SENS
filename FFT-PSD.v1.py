import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal

# CSV inladen
df = pd.read_csv("C:/Users/Dell/Downloads/test-1-v2.csv")
df.columns = ['t', 'x', 'y', 'z']  # Zorg ervoor dat je kolomnamen goed zijn
t = df['t']
x = df['x']
y = df['y']
z = df['z']

# Bereken de totale accelaratie
df['total_vibration'] = np.sqrt(x**2 + y**2 + z**2)

# Verwijder de DC-component (gemiddelde van het signaal afhalen)
df['total_vibration'] = df['total_vibration'] - np.mean(df['total_vibration'])

# Bereken het tijdsverschil tussen opeenvolgende tijdstappen
dt = np.diff(t)  # Dit geeft het verschil tussen opeenvolgende tijdstappen

# Bereken de gemiddelde samplefrequentie (omgekeerde van het gemiddelde tijdsverschil)
average_dt = np.mean(dt)  # Gemiddeld tijdsinterval
fs = 1 / average_dt  # Samplefrequentie is 1 gedeeld door het gemiddelde tijdsinterval

# FFT berekenen
N = len(df['total_vibration'])  # Aantal datapunten
frequencies_fft = np.fft.fftfreq(N, 1/fs)  # Bereken de frequenties voor de FFT
fft_values = np.fft.fft(df['total_vibration'])  # Voer de FFT uit
fft_magnitude = np.abs(fft_values) / N  # Schaal de magnitude van de FFT

# PSD berekenen met de Welch methode
frequencies_psd, psd = signal.welch(df['total_vibration'], fs, nperseg=1024)

# Plot de FFT
plt.figure(figsize=(18, 10))
plt.subplot(2, 1, 1)
plt.plot(frequencies_fft[:N//2], fft_magnitude[:N//2])  # Alleen de positieve frequenties
plt.title('FFT van Totale versnelling')
plt.xlabel('Frequentie [Hz]')
plt.ylabel('Magnitude')
plt.grid(True)

# Plot de PSD
plt.subplot(2, 1, 2)
plt.semilogy(frequencies_psd, psd)
plt.title('Power Spectral Density (PSD) van Totale versnelling')
plt.xlabel('Frequentie [Hz]')
plt.ylabel('PSD [m^2/s^3/Hz]')
plt.grid(True)

plt.tight_layout()
plt.show()
