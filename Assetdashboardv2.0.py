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

# Batterijstatus berekenen met totale tijd
battery_capacity_mAh = 20000
sensor_power_mA = 50  # schatting: 50mA per sensor

# Leest de totale tijd (tt) uit het tijd.csv bestand
tijd_csv_path = "C:/Users/Dell/Downloads/tijd.csv"
tijd_df = pd.read_csv(tijd_csv_path)

# Haal de totale tijd (tt) en tijd sinds onderhoud (td) op uit het bestand
total_time_from_csv = tijd_df['tt'].iloc[0]
tijd_since_maintenance = tijd_df['td'].iloc[0]

# Nieuwe totale tijd is de oude totale tijd + de huidige metingstijd
new_total_time = total_time_from_csv + total_duration

# Voeg de meettijd toe aan de tijd sinds onderhoud (td)
new_tijd_since_maintenance = tijd_since_maintenance + total_duration  # voeg meettijd toe aan tijd sinds onderhoud

# Bereken het energieverbruik met de totale tijd
current_consumption_mAh = sensor_count * sensor_power_mA * (new_total_time / 3600)  # in mAh
remaining_battery_percentage = max(0, 100 - (current_consumption_mAh / battery_capacity_mAh) * 100)

# Assetmanager gegevens verzamelen
asset_data = []
global_threshold = 4.5

# Tijd CSV bestand bijwerken (tijd.csv)
# Werk de totale tijd (tt), tijd sinds kalibratie (tk), tijd sinds laatste meting (tm) en tijd sinds onderhoud (td) bij
tijd_df['tt'] = new_total_time
tijd_df['tk'] = tijd_df['tk'].iloc[0] + total_duration  # Tijd sinds kalibratie wordt ook aangepast
tijd_df['tm'] = total_duration  # Tijd sinds laatste meting (veronderstel hier hetzelfde)
tijd_df['td'] = new_tijd_since_maintenance  # Tijd sinds onderhoud bijwerken door meettijd toe te voegen

# Sla de bijgewerkte tijden op
tijd_df.to_csv(tijd_csv_path, index=False)

# Verzamel sensor gegevens
for i in range(1, sensor_count_from_csv + 1):
    x = df[f"x{i}"] - np.mean(df[f"x{i}"])
    y = df[f"y{i}"] - np.mean(df[f"y{i}"])
    z = df[f"z{i}"] - np.mean(df[f"z{i}"])

    active = True
    overschreden = (
        np.max(np.abs(x)) > global_threshold or 
        np.max(np.abs(y)) > global_threshold or 
        np.max(np.abs(z)) > global_threshold
    )

    # Voeg de gegevens van de sensor toe, inclusief de bijgewerkte tijden
    asset_data.append({
        "Sensor Naam": f"Sensor {i}",
        "Locatie": sensor_locations[i - 1],
        "Actief": "Ja" if active else "Nee",
        "Batterij (%)": f"{remaining_battery_percentage:.2f}%",
        "Grenswaarde Overschreden": "Ja" if overschreden else "Nee",
        "Totale Tijd (tt)": f"{new_total_time:.2f}s",  # 2 decimalen
        "Tijd Sinds Kalibratie (tk)": f"{tijd_df['tk'].iloc[0]:.2f}s",  # 2 decimalen
        "Tijd Sinds Laatste Meting (tm)": f"{total_duration:.2f}s",  # 2 decimalen
        "Tijd Sinds Onderhoud (td)": f"{new_tijd_since_maintenance / 3600:.2f}u"  # Converteer naar uren en 2 decimalen
    })

# Maak een DataFrame van assetmanager gegevens
asset_df = pd.DataFrame(asset_data)

# Tabel in een apart venster weergeven
fig, ax = plt.subplots(figsize=(10, len(asset_data) * 0.6))
ax.axis('tight')
ax.axis('off')

# Voeg de tabel toe en pas tekstinstellingen aan voor leesbaarheid
table = ax.table(cellText=asset_df.values, colLabels=asset_df.columns, cellLoc='center', loc='center')

# Pas de lettergrootte aan voor de tabel en zorg voor een strakke layout
table.auto_set_font_size(False)
table.set_fontsize(10)

# Zorg ervoor dat de tekst in de cellen terugloopt
for (i, j), cell in table.get_celld().items():
    if i == 0:
        # Pas de kolombreedte aan voor de header (optie)
        cell.set_text_props(weight='bold')
    else:
        cell.set_text_props(ha='center', va='center', wrap=True)  # Terugloop instellen
    
    # Maak de cellen breder en zorg ervoor dat lange teksten kunnen afbreken
    cell.set_edgecolor('black')
    cell.set_linewidth(0.5)

# Pas de rijenhoogte aan voor tekstterugloop
table.auto_set_column_width([i for i in range(len(asset_df.columns))])

# Verbeter de lay-out zodat de tekst goed wordt weergegeven
plt.tight_layout()

plt.show()
