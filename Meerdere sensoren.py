import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal

# CSV inladen
df = pd.read_csv("C:/Users/Dell/Downloads/WIFI.csv")

# Dynamisch bepalen van het aantal sensoren
time_column = 't'
sensor_count = (len(df.columns) - 1) // 3  # Elke sensor heeft x, y, z
columns = [time_column] + [f"{ax}{i}" for i in range(1, sensor_count + 1) for ax in ['x', 'y', 'z']]
df.columns = columns

# Tijd omzetten naar seconden
df[time_column] /= 1000

# Tijd- en frequentiedomein plots per sensor
fs = 1 / np.mean(np.diff(df[time_column]))  # Samplefrequentie
N = len(df[time_column])  # Aantal datapunten
frequencies_fft = np.fft.fftfreq(N, 1/fs)  # Frequenties voor de FFT

for i in range(1, sensor_count + 1):
    # Data ophalen en de DC-component verwijderen
    x = df[f"x{i}"] - np.mean(df[f"x{i}"])
    y = df[f"y{i}"] - np.mean(df[f"y{i}"])
    z = df[f"z{i}"] - np.mean(df[f"z{i}"])
    
    # Controleer of er data is (niet alleen nullen)
    if np.all(x == 0) and np.all(y == 0) and np.all(z == 0):
        print(f"Sensor {i} heeft geen actieve data en wordt overgeslagen.")
        continue

    # Vervang eventuele NaN of Inf waarden door 0
    x = np.nan_to_num(x)
    y = np.nan_to_num(y)
    z = np.nan_to_num(z)

    # FFT voor deze sensor
    fft_x = np.fft.fft(x)
    fft_y = np.fft.fft(y)
    fft_z = np.fft.fft(z)

    fft_combined_magnitude = np.sqrt(np.abs(fft_x)**2 + np.abs(fft_y)**2 + np.abs(fft_z)**2)

    # Spectrogram berekeningen
    frequencies_spec, times_spec, Sxx_x = signal.spectrogram(x, fs, nperseg=1024, noverlap=512)
    _, _, Sxx_y = signal.spectrogram(y, fs, nperseg=1024, noverlap=512)
    _, _, Sxx_z = signal.spectrogram(z, fs, nperseg=1024, noverlap=512)
    Sxx_combined = np.sqrt(Sxx_x**2 + Sxx_y**2 + Sxx_z**2)

    # Plotten
    fig, axs = plt.subplots(2, 2, figsize=(18, 10))

    # 1. Plot ruwe data
    axs[0, 0].plot(df[time_column], x, label=f'x{i}', color='red', alpha=0.6)
    axs[0, 0].plot(df[time_column], y, label=f'y{i}', color='green', alpha=0.6)
    axs[0, 0].plot(df[time_column], z, label=f'z{i}', color='blue', alpha=0.6)
    axs[0, 0].set_title(f"Sensor {i} - Ruwe Versnelling vs Tijd")
    axs[0, 0].set_xlabel('Tijd [s]')
    axs[0, 0].set_ylabel('Versnelling [m/s^2]')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # 2. Plot FFT
    axs[0, 1].plot(frequencies_fft[:N // 2], fft_combined_magnitude[:N // 2] / N, label=f"Sensor {i}", linewidth=2)
    axs[0, 1].set_title(f"Sensor {i} - FFT van Versnelling")
    axs[0, 1].set_xlabel('Frequentie [Hz]')
    axs[0, 1].set_ylabel('Magnitude')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # 3. PSD Plot
    frequencies_psd, psd_x = signal.welch(x, fs, nperseg=1024)
    _, psd_y = signal.welch(y, fs, nperseg=1024)
    _, psd_z = signal.welch(z, fs, nperseg=1024)
    psd_combined = np.sqrt(psd_x**2 + psd_y**2 + psd_z**2)

    axs[1, 0].semilogy(frequencies_psd, psd_combined, label=f"Sensor {i}", color='orange', linewidth=2)
    axs[1, 0].set_title(f"Sensor {i} - Power Spectral Density (PSD)")
    axs[1, 0].set_xlabel('Frequentie [Hz]')
    axs[1, 0].set_ylabel('PSD [m^2/s^4/Hz]')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # 4. Spectrogram
    X, Y = np.meshgrid(times_spec, frequencies_spec)
    Z = Sxx_combined

    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='inferno', edgecolor='none')
    ax.set_title(f"Sensor {i} - Waterfall Plot")
    ax.set_xlabel('Tijd [s]')
    ax.set_ylabel('Frequentie [Hz]')
    ax.set_zlabel('Amplitude')

    plt.tight_layout()
    plt.show()
