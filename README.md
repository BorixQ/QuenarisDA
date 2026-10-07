# Queñaris — análisis geoespacial para reforestación

Pipeline en Python que procesa imágenes Sentinel-2 y topografía SRTM mediante Google Earth Engine para obtener índices ambientales y seleccionar sitios candidatos para plantación.

La implementación web del análisis geoespacial se encuentra en [Terranode / Queñaris MVP](https://github.com/BorixQ/quenaris_mvp).

## Flujo de análisis

1. Definir el área de estudio y el período de imágenes.
2. Extraer índices de vegetación, humedad y suelo, junto con variables topográficas.
3. Agrupar sitios mediante DBSCAN y ordenarlos con un score calculado a partir de sus variables.
4. Exportar coordenadas, índices y resultados como CSV y GeoJSON.

El score expresa el criterio del algoritmo; este repositorio no incluye evidencia de validación de supervivencia en campo.

## Tecnologías

Python, Google Earth Engine, pandas, NumPy, GeoPandas, Shapely y scikit-learn. Los proyectos QGIS y notebooks acompañan el análisis.

## Archivos principales

| Ruta | Función |
| --- | --- |
| `4run_full_pipeline.py` | Configuración y ejecución del flujo completo. |
| `modulo1_extraccion_indices.py` | Extracción de índices satelitales y topografía. |
| `modulo3_seleccion_ia.py` | Clustering, puntuación y selección de sitios. |
| `datos_indices_puros/` | Salidas de la extracción. |
| `datos_scoring/` | Resultados de puntuación. |
| `sitios_plantacion/` | Exportaciones para plantación. |
| `Quenaris Web/` | Recursos de visualización web. |

## Preparar una ejecución

```bash
git clone https://github.com/BorixQ/QuenarisDA.git
cd QuenarisDA
python -m venv .venv
```

Activa el entorno según tu sistema e instala las dependencias importadas por el pipeline:

```bash
python -m pip install earthengine-api geopandas scikit-learn shapely pandas numpy
earthengine authenticate
```

Configura `ConfiguracionReforestacion` en `4run_full_pipeline.py` con tu propio proyecto habilitado en Earth Engine, polígono WGS84, fechas y parámetros de selección. Después ejecuta:

```bash
python 4run_full_pipeline.py
```

El repositorio contiene un área de ejemplo y salidas previas. Ante errores de extracción, el pipeline puede recurrir a archivos existentes; comprueba su procedencia y fecha antes de interpretar el resultado.

## Estado y próximos pasos

El repositorio reúne scripts, notebooks y resultados de investigación. Aún no tiene dependencias fijadas ni una suite de pruebas automatizadas. Una ejecución reproducible requiere configurar Earth Engine y verificar el entorno geoespacial.

Mejoras pendientes: fijar dependencias, separar configuración de datos de ejemplo y documentar la validación de los criterios de selección.
