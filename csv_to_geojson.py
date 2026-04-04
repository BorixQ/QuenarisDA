import pandas as pd
import json

# 1. Cargar los datos planos (Estructura Tabular)
# Pandas es ideal aquí por eficiencia, similar a manejar arrays en Física computacional.
df = pd.read_csv('ubicacion_500_plantulas_wgs84.csv')

# 2. Transformación a Estructura Jerárquica (FeatureCollection)
features = []

for _, row in df.iterrows():
    # Cada fila se convierte en un objeto 'Feature'
    feature = {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            # GeoJSON exige el orden [Longitud, Latitud] (X, Y), no al revés.
            "coordinates": [row['Longitude'], row['Latitude']]
        },
        "properties": {
            # Los metadatos van encapsulados aquí
            "ID": row['ID']
        }
    }
    features.append(feature)

# El objeto raíz que contiene todo
geojson_structure = {
    "type": "FeatureCollection",
    "features": features
}

# 3. Serialización a archivo
with open('ubicacion_500_plantulas.geojson', 'w') as f:
    json.dump(geojson_structure, f, indent=2)