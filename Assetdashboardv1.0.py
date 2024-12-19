import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Invoer van de console voor sensorlokaties
sensor_locations = []
sensor_count = int(input("Voer het aantal sensoren in: "))
for i in range(1, sensor_count + 1):
    location = input(f"Geef de locatie van sensor {i}: ")
    sensor_locations.append(location)

# Dynamisch bepalen van het aantal sensoren uit CSV
csv_path = "C:/Users/Dell/Downloads/kedeng-kedenggg.csv"
df = pd.read_csv(csv_path)

time_column = 't'
sensor_count_from_csv = (len(df.columns) - 1) // 3  # Elke sensor heeft x, y, z
columns = [time_column] + [f"{ax}{i}" for i in range(1, sensor_count_from_csv + 1) for ax in ['x', 'y', 'z']]
df.columns = columns

# Tijd omzetten naar seconden
df[time_column] /= 1000

# Samplefrequentie en duur van de meting
fs = 1 / np.mean(np.diff(df[time_column]))
begin_time = df[time_column].iloc[0]
end_time = df[time_column].iloc[-1]
total_duration = end_time - begin_time

# Batterijstatus berekenen
battery_capacity_mAh = 20000
sensor_power_mA = 50  # schatting: 50mA per sensor
current_consumption_mAh = sensor_count * sensor_power_mA * (total_duration / 3600)
remaining_battery_percentage = max(0, 100 - (current_consumption_mAh / battery_capacity_mAh) * 100)

# Assetmanager gegevens verzamelen
asset_data = []
global_threshold = 4.5

# Tijd CSV bestand bijwerken (tijd.csv)
tijd_csv_path = "C:/Users/Dell/Downloads/tijd.csv"
tijd_df = pd.read_csv(tijd_csv_path)

# Tijd berekenen en updaten voor elke sensor
for i in range(1, sensor_count_from_csv + 1):
    # Bepaal de tijden voor tt, tk, tm
    tt = tijd_df['tt'].iloc[0] + total_duration  # Totale tijd
    tk = tijd_df['tk'].iloc[0] + total_duration  # Tijd sinds kalibratie (veronderstel hier hetzelfde)
    tm = total_duration  # Tijd sinds laatste meting (veronderstel hier hetzelfde)

    # Update de tijd in de DataFrame
    tijd_df['tt'] = tt
    tijd_df['tk'] = tk
    tijd_df['tm'] = tm

    # Sla de bijgewerkte tijden op
    tijd_df.to_csv(tijd_csv_path, index=False)

    # Sensor data bewerken
    x = df[f"x{i}"] - np.mean(df[f"x{i}"])
    y = df[f"y{i}"] - np.mean(df[f"y{i}"])
    z = df[f"z{i}"] - np.mean(df[f"z{i}"])

    active = True
    overschreden = (
        np.max(np.abs(x)) > global_threshold or 
        np.max(np.abs(y)) > global_threshold or 
        np.max(np.abs(z)) > global_threshold
    )

    # Toevoegen van afgeronde tijden met 2 decimalen
    asset_data.append({
        "Sensor Naam": f"Sensor {i}",
        "Locatie": sensor_locations[i - 1],
        "Actief": "Ja" if active else "Nee",
        "Batterij (%)": f"{remaining_battery_percentage:.2f}%",
        "Grenswaarde Overschreden": "Ja" if overschreden else "Nee",
        "Totale Tijd (tt)": f"{tt:.2f}s",  # 2 decimalen
        "Tijd Sinds Kalibratie (tk)": f"{tk:.2f}s",  # 2 decimalen
        "Tijd Sinds Laatste Meting (tm)": f"{tm:.2f}s"  # 2 decimalen
    })

# Maak een DataFrame van assetmanager gegevens
asset_df = pd.DataFrame(asset_data)

# Tabel in een apart venster weergeven
fig, ax = plt.subplots(figsize=(10, len(asset_data) * 0.6))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=asset_df.values, colLabels=asset_df.columns, cellLoc='center', loc='center')

# Pas de lettergrootte aan voor de tabel en zorg voor een strakke layout
table.auto_set_font_size(False)
table.set_fontsize(10)
plt.tight_layout()

plt.show()
