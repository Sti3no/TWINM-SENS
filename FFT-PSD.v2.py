import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal

# CSV inladen
df = pd.read_csv("C:/Users/Dell/Downloads/test-1-v2.csv")
df.columns = ['t', 'x', 'y', 'z']  # Zorg ervoor dat je kolomnamen goed zijn
t = df['t'] / 1000  # Converteer tijd naar seconden
x = df['x']
y = df['y']
z = df['z']

# Bereken de totale acceleratie
df['total_vibration'] = np.sqrt(x**2 + y**2 + z**2)

# Verwijder de DC-component (gemiddelde van het signaal afhalen)
df['total_vibration'] = df['total_vibration'] - np.mean(df['total_vibration'])

# Bereken het tijdsverschil tussen opeenvolgende tijdstappen
dt = np.diff(t)  # Dit geeft het verschil tussen opeenvolgende tijdstappen

# Bereken de gemiddelde samplefrequentie (omgekeerde van het gemiddelde tijdsverschil)
average_dt = np.mean(dt)  # Gemiddeld tijdsinterval
fs = 1 / average_dt  # Samplefrequentie is 1 gedeeld door het gemiddelde tijdsinterval

# Bereken het aantal datapunten
N = len(df['total_vibration'])

# Spectrogram berekenen (deze keer gebruiken we een schuivende FFT om een waterfall plot te maken)
nperseg = 1024  # Lengte van het segment voor de FFT
noverlap = nperseg // 2  # Overlap tussen opeenvolgende segmenten
frequencies_spec, times_spec, Sxx = signal.spectrogram(df['total_vibration'], fs, nperseg=nperseg, noverlap=noverlap)

# Maak de figuur en voeg meerdere subplots toe
fig, axs = plt.subplots(2, 2, figsize=(18, 10))

# 1. Plot de ruwe data (versnelling vs tijd)
axs[0, 0].plot(t, df['total_vibration'], label='Totale versnelling', color='blue')
axs[0, 0].set_title('Ruwe Versnelling vs Tijd')
axs[0, 0].set_xlabel('Tijd [s]')
axs[0, 0].set_ylabel('Versnelling [m/s^2]')
axs[0, 0].grid(True)

# 2. FFT Plot (Frequentie vs Magnitude)
frequencies_fft = np.fft.fftfreq(N, 1/fs)
fft_values = np.fft.fft(df['total_vibration'])
fft_magnitude = np.abs(fft_values) / N  # Schaal de magnitude van de FFT

axs[0, 1].plot(frequencies_fft[:N//2], fft_magnitude[:N//2], label='FFT van Versnelling', color='orange')
axs[0, 1].set_title('FFT van Versnelling')
axs[0, 1].set_xlabel('Frequentie [Hz]')
axs[0, 1].set_ylabel('Magnitude')
axs[0, 1].grid(True)

# 3. PSD Plot (Frequentie vs PSD)
frequencies_psd, psd = signal.welch(df['total_vibration'], fs, nperseg=1024)

axs[1, 0].semilogy(frequencies_psd, psd, label='PSD van Versnelling', color='green')
axs[1, 0].set_title('Power Spectral Density (PSD) van Versnelling')
axs[1, 0].set_xlabel('Frequentie [Hz]')
axs[1, 0].set_ylabel('PSD [m^2/s^4/Hz]')
axs[1, 0].grid(True)

# 4. Waterfall Plot
# Voor de waterfall plot gebruiken we een 3D plot (maar die komt nu in een aparte figuur)
fig2 = plt.figure(figsize=(12, 8))
ax = fig2.add_subplot(111, projection='3d')

X, Y = np.meshgrid(times_spec, frequencies_spec)
Z = Sxx

ax.plot_surface(X, Y, Z, cmap='inferno', edgecolor='none')

ax.set_title('Waterfall Plot van Versnelling', fontsize=16)
ax.set_xlabel('Tijd [s]', fontsize=14)
ax.set_ylabel('Frequentie [Hz]', fontsize=14)
ax.set_zlabel('Amplitude', fontsize=14)
ax.set_zlim(0, np.max(Z))  # Zorg ervoor dat de z-as vanaf 0 begint

# Zorg voor een strakke layout
plt.tight_layout()
plt.show()
