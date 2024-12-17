import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal

# CSV inladen
df = pd.read_csv("C:/Users/Dell/Downloads/kedeng-kedenggg.csv")
df.columns = ['t', 'x', 'y', 'z']  # Pas dit aan naar de juiste kolomnamen
t = df['t'] / 1000 # Converteer tijd naar seconden
x = df['x']
y = df['y']
z = df['z']

# Verwijder de DC-component (gemiddelde van het signaal afhalen) per component
x = x - np.mean(x)
y = y - np.mean(y)
z = z - np.mean(z)

x = np.nan_to_num(x)  # Vervang NaN of Inf met 0
y = np.nan_to_num(y)
z = np.nan_to_num(z)

# Bereken het tijdsverschil tussen opeenvolgende tijdstappen
dt = np.diff(t)  # Dit geeft het verschil tussen opeenvolgende tijdstappen

# Bereken de gemiddelde samplefrequentie (omgekeerde van het gemiddelde tijdsverschil)
average_dt = np.mean(dt)  # Gemiddeld tijdsinterval
fs = 1 / average_dt  # Samplefrequentie is 1 gedeeld door het gemiddelde tijdsinterval

# Bereken het aantal datapunten
N = len(t)

# FFT van elk signaal
frequencies_fft = np.fft.fftfreq(N, 1/fs)

fft_x = np.fft.fft(x)
fft_y = np.fft.fft(y)
fft_z = np.fft.fft(z)

# Combineer de spectrale bijdragen
# Combineer bijvoorbeeld de magnitude spectra van x, y en z
fft_combined_magnitude = (np.abs(fft_x) + np.abs(fft_y) + np.abs(fft_z)) / 3  # Gemiddelde magnitude

# Spectrogrammen voor visualisatie (optioneel)
frequencies_spec, times_spec, Sxx_x = signal.spectrogram(x, fs, nperseg=1024, noverlap=512)
_, _, Sxx_y = signal.spectrogram(y, fs, nperseg=1024, noverlap=512)
_, _, Sxx_z = signal.spectrogram(z, fs, nperseg=1024, noverlap=512)

Sxx_combined = (Sxx_x + Sxx_y + Sxx_z) / 3  # Gemiddelde spectrale power

# Plot de resultaten
fig, axs = plt.subplots(2, 2, figsize=(18, 10))

# 1. Plot de ruwe data (versnelling vs tijd)
axs[0, 0].plot(t, x, label='x-component', color='red', alpha=0.6)
axs[0, 0].plot(t, y, label='y-component', color='green', alpha=0.6)
axs[0, 0].plot(t, z, label='z-component', color='blue', alpha=0.6)
axs[0, 0].set_title('Ruwe Versnelling vs Tijd')
axs[0, 0].set_xlabel('Tijd [s]')
axs[0, 0].set_ylabel('Versnelling [m/s^2]')
axs[0, 0].legend()
axs[0, 0].grid(True)

# 2. FFT Plot (Frequentie vs Magnitude)
#axs[0, 1].plot(frequencies_fft[:N//2], np.abs(fft_x[:N//2]) / N, label='FFT x', color='red', alpha=0.6)
#axs[0, 1].plot(frequencies_fft[:N//2], np.abs(fft_y[:N//2]) / N, label='FFT y', color='green', alpha=0.6)
#axs[0, 1].plot(frequencies_fft[:N//2], np.abs(fft_z[:N//2]) / N, label='FFT z', color='blue', alpha=0.6)
axs[0, 1].plot(frequencies_fft[:N//2], fft_combined_magnitude[:N//2] / N, label='Gecombineerde FFT', color='orange', linewidth=2)
axs[0, 1].set_title('FFT van Versnelling')
axs[0, 1].set_xlabel('Frequentie [Hz]')
axs[0, 1].set_ylabel('Magnitude')
axs[0, 1].legend()
axs[0, 1].grid(True)

# 3. PSD Plot (Frequentie vs PSD)
frequencies_psd, psd_x = signal.welch(x, fs, nperseg=1024)
_, psd_y = signal.welch(y, fs, nperseg=1024)
_, psd_z = signal.welch(z, fs, nperseg=1024)

psd_combined = (psd_x + psd_y + psd_z) / 3

#axs[1, 0].semilogy(frequencies_psd, psd_x, label='PSD x', color='red', alpha=0.6)
#axs[1, 0].semilogy(frequencies_psd, psd_y, label='PSD y', color='green', alpha=0.6)
#axs[1, 0].semilogy(frequencies_psd, psd_z, label='PSD z', color='blue', alpha=0.6)
axs[1, 0].semilogy(frequencies_psd, psd_combined, label='Gecombineerde PSD', color='green', linewidth=2)
axs[1, 0].set_title('Power Spectral Density (PSD) van Versnelling')
axs[1, 0].set_xlabel('Frequentie [Hz]')
axs[1, 0].set_ylabel('PSD [m^2/s^4/Hz]')
axs[1, 0].legend()
axs[1, 0].grid(True)

# 4. Waterfall Plot (Rechtonder in de 2x2 grid)
ax = fig.add_subplot(2, 2, 4, projection='3d')  # Dit maakt de 3D plot in de rechterbenedenhoek

X, Y = np.meshgrid(times_spec, frequencies_spec)
Z = Sxx_combined

ax.plot_surface(X, Y, Z, cmap='inferno', edgecolor='none')

ax.set_title('Waterfall Plot van Frequentie', fontsize=16)
ax.set_xlabel('Tijd [s]', fontsize=14)
ax.set_ylabel('Frequentie [Hz]', fontsize=14)
ax.set_zlabel('Amplitude', fontsize=14)
ax.set_zlim(0, np.max(Z))  # Zorg ervoor dat de z-as vanaf 0 begint

# Zorg voor een strakke layout
plt.tight_layout()
plt.show()
