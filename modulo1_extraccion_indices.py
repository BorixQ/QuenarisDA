import ee
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
import json
import numpy as np
from datetime import datetime
import os
from typing import Dict, List, Tuple

# ============================================================================
# MÓDULO 1: EXTRACCIÓN DE ÍNDICES ECOFISIOLÓGICOS Y TOPOGRÁFICOS
# ============================================================================

class SatelliteIndicesExtractor:
    """
    Extractor de índices espectrales y topográficos sin procesamiento multicriterio.
    Genera datos puros para análisis posterior con IA/clustering.
    """
    
    def __init__(self, project_id: str, coords: List[List[float]], 
                 fecha_inicio: str, fecha_fin: str):
        self.project_id = project_id
        self.coords = coords
        self.fecha_inicio = fecha_inicio
        self.fecha_fin = fecha_fin
        self.output_dir = 'datos_indices_puros'
        
        # Inicializar GEE
        self._initialize_gee()
        
        # Crear ROI
        self.roi = ee.Geometry.Polygon([coords], None, False)
        
        # Bandas espectrales Sentinel-2
        self.bandas = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 
                       'B8', 'B8A', 'B9', 'B11', 'B12']
        
        # Crear directorio de salida
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _initialize_gee(self):
        """Inicializa conexión con Google Earth Engine"""
        try:
            ee.Initialize(
                project=self.project_id,
                opt_url='https://earthengine-highvolume.googleapis.com'
            )
            print(f"✅ Conectado a Google Earth Engine (Proyecto: {self.project_id})")
        except Exception as e:
            print("⚠️ Error al inicializar GEE")
            raise

    # ========================================================================
    # PROCESAMIENTO DE IMÁGENES SATELITALES
    # ========================================================================
    
    def mask_s2_clouds(self, image):
        """Máscara de nubes mejorada para Sentinel-2"""
        qa = image.select('QA60')
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
               qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        
        scl = image.select('SCL')
        scl_mask = scl.gte(4).And(scl.lte(6))
        combined_mask = mask.And(scl_mask)
        
        return image.updateMask(combined_mask).divide(10000).select(self.bandas) \
                    .copyProperties(image, ['system:time_start', 'CLOUDY_PIXEL_PERCENTAGE'])
    
    def add_spectral_indices(self, image):
        """Añade todos los índices espectrales ecofisiológicos"""
        
        # ---- ÍNDICES DE VEGETACIÓN (Vigor) ----
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        
        evi = image.expression(
            '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
                'NIR': image.select('B8'),
                'RED': image.select('B4'),
                'BLUE': image.select('B2')
            }).rename('EVI')
        
        savi = image.expression(
            '((NIR - RED) / (NIR + RED + 0.428)) * 1.428', {
                'NIR': image.select('B8'),
                'RED': image.select('B4')
            }).rename('SAVI')
        
        ndre = image.normalizedDifference(['B8', 'B5']).rename('NDRE')
        gndvi = image.normalizedDifference(['B8', 'B3']).rename('GNDVI')
        
        cire = image.expression(
            '(NIR / RE1) - 1', {
                'NIR': image.select('B8'),
                'RE1': image.select('B5')
            }).rename('CIre')
        
        # ---- ÍNDICES DE HUMEDAD ----
        ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
        ndmi = image.normalizedDifference(['B8', 'B11']).rename('NDMI')
        
        msi = image.expression(
            'SWIR1 / NIR', {
                'SWIR1': image.select('B11'),
                'NIR': image.select('B8')
            }).rename('MSI')
        
        # ---- ÍNDICES DE ESTRÉS HÍDRICO ----
        moisture_stress = image.expression(
            '(SWIR1 - NIR) / (SWIR1 + NIR)', {
                'SWIR1': image.select('B11'),
                'NIR': image.select('B8')
            }).rename('MoistureStress')
        
        wdi = image.expression(
            '(RED + GREEN) / (NIR + SWIR1)', {
                'RED': image.select('B4'),
                'GREEN': image.select('B3'),
                'NIR': image.select('B8'),
                'SWIR1': image.select('B11')
            }).rename('WDI')
        
        # ---- ÍNDICES DE SUELO ----
        nbr = image.normalizedDifference(['B8', 'B12']).rename('NBR')
        
        bsi = image.expression(
            '((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))', {
                'SWIR1': image.select('B11'),
                'RED': image.select('B4'),
                'NIR': image.select('B8'),
                'BLUE': image.select('B2')
            }).rename('BSI')
        
        ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI')
        
        # ---- ÍNDICES ADICIONALES ----
        lai = image.expression(
            '3.618 * EVI - 0.118', {
                'EVI': evi
            }).rename('LAI')
        
        arvi = image.expression(
            '(NIR - (2 * RED) + BLUE) / (NIR + (2 * RED) + BLUE)', {
                'NIR': image.select('B8'),
                'RED': image.select('B4'),
                'BLUE': image.select('B2')
            }).rename('ARVI')
        
        sipi = image.expression(
            '(NIR - BLUE) / (NIR - RED)', {
                'NIR': image.select('B8'),
                'BLUE': image.select('B2'),
                'RED': image.select('B4')
            }).rename('SIPI')
        
        psri = image.expression(
            '(RED - GREEN) / RE1', {
                'RED': image.select('B4'),
                'GREEN': image.select('B3'),
                'RE1': image.select('B6')
            }).rename('PSRI')
        
        return image.addBands([
            ndvi, evi, savi, ndre, gndvi, cire,
            ndwi, ndmi, msi, moisture_stress, wdi,
            nbr, bsi, ndbi,
            lai, arvi, sipi, psri
        ])
    
    # ========================================================================
    # ANÁLISIS TOPOGRÁFICO
    # ========================================================================
    
    def add_topographic_features(self):
        """Añade características topográficas del terreno"""
        dem = ee.Image('USGS/SRTMGL1_003').clip(self.roi)
        
        elevation = dem.select('elevation').rename('Elevation')
        slope = ee.Terrain.slope(dem).rename('Slope')
        aspect = ee.Terrain.aspect(dem).rename('Aspect')
        
        tri = dem.subtract(
            dem.focal_mean(radius=1, kernelType='square', units='pixels')
        ).abs().rename('TRI')
        
        tpi = dem.subtract(
            dem.focal_mean(radius=3, kernelType='circle', units='pixels')
        ).rename('TPI')
        
        # Exposición solar (Norte vs Sur en Hemisferio Sur)
        solar_exposure = aspect.expression(
            '(cos((aspect - 0) * 3.14159 / 180) + 1) / 2', {
                'aspect': aspect
            }).rename('SolarExposure')
        
        return ee.Image.cat([
            elevation, slope, aspect, tri, tpi, solar_exposure
        ])
    
    # ========================================================================
    # PROCESAMIENTO PRINCIPAL
    # ========================================================================
    
    def process_satellite_data(self):
        """Procesa datos satelitales y genera composición con índices"""
        print("📡 Descargando imágenes Sentinel-2...")
        
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(self.roi) \
            .filterDate(self.fecha_inicio, self.fecha_fin) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
            .map(self.mask_s2_clouds) \
            .map(self.add_spectral_indices)
        
        # Composición mediana
        composite = s2.median().clip(self.roi)
        
        # Añadir topografía
        topo = self.add_topographic_features()
        composite = composite.addBands(topo)
        
        print(f"✅ Procesadas {s2.size().getInfo()} imágenes")
        
        return composite
    
    # ========================================================================
    # GENERACIÓN DE GRID DE MUESTREO
    # ========================================================================
    
    def create_sampling_grid(self, cell_size_meters: int = 50, max_points: int = None):
        """Crea grid regular de puntos de muestreo"""
        
        bounds = self.roi.bounds().getInfo()['coordinates'][0]
        min_lon = min([p[0] for p in bounds])
        max_lon = max([p[0] for p in bounds])
        min_lat = min([p[1] for p in bounds])
        max_lat = max([p[1] for p in bounds])
        
        # Convertir metros a grados
        lat_center = (min_lat + max_lat) / 2
        meters_per_degree = 111320 * np.cos(np.radians(lat_center))
        cell_size_deg = cell_size_meters / meters_per_degree
        
        # Crear grid
        lons = np.arange(min_lon, max_lon, cell_size_deg)
        lats = np.arange(min_lat, max_lat, cell_size_deg)
        
        polygon = Polygon(self.coords)
        points = []
        
        for lon in lons:
            for lat in lats:
                point = Point(lon, lat)
                if polygon.contains(point):
                    points.append(point)
                    
                    if max_points and len(points) >= max_points:
                        print(f"⚠️ Alcanzado límite de {max_points:,} puntos")
                        return points
        
        print(f"📍 Generados {len(points):,} puntos de muestreo ({cell_size_meters}m)")
        return points
    
    # ========================================================================
    # EXTRACCIÓN Y EXPORTACIÓN DE DATOS
    # ========================================================================
    
    def extract_point_values(self, composite, points: List[Point], 
                            scale: int = 10, batch_size: int = 5000) -> gpd.GeoDataFrame:
        """Extrae valores de todas las bandas en puntos de muestreo"""
        
        total_points = len(points)
        print(f"🔍 Extrayendo valores en {total_points} puntos...")
        
        if total_points <= batch_size:
            return self._extract_batch(composite, points, scale)
        
        # Procesar en lotes
        print(f"⚙️ Procesando en lotes de {batch_size} puntos...")
        
        all_gdfs = []
        num_batches = int(np.ceil(total_points / batch_size))
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_points)
            batch_points = points[start_idx:end_idx]
            
            print(f"   Lote {i+1}/{num_batches}: puntos {start_idx+1}-{end_idx}")
            
            try:
                batch_gdf = self._extract_batch(composite, batch_points, scale, 
                                               start_id=start_idx+1)
                all_gdfs.append(batch_gdf)
                
            except Exception as e:
                print(f"   ⚠️ Error en lote {i+1}: {e}")
                # Reintentar con sub-lotes
                sub_batch_size = len(batch_points) // 2
                for j in range(2):
                    sub_start = j * sub_batch_size
                    sub_end = (j + 1) * sub_batch_size if j == 0 else len(batch_points)
                    sub_points = batch_points[sub_start:sub_end]
                    
                    try:
                        sub_gdf = self._extract_batch(composite, sub_points, scale,
                                                     start_id=start_idx+sub_start+1)
                        all_gdfs.append(sub_gdf)
                    except Exception as e2:
                        print(f"   ❌ Fallo sub-lote {j+1}: {e2}")
        
        if all_gdfs:
            gdf_combined = pd.concat(all_gdfs, ignore_index=True)
            gdf_combined = gpd.GeoDataFrame(gdf_combined, geometry='geometry', crs='EPSG:4326')
            print(f"✅ Extraídos datos para {len(gdf_combined)} puntos")
            return gdf_combined
        else:
            raise Exception("No se pudieron extraer datos")
    
    def _extract_batch(self, composite, points: List[Point], scale: int = 10,
                      start_id: int = 1) -> gpd.GeoDataFrame:
        """Extrae un lote de puntos"""
        
        features = [ee.Feature(ee.Geometry.Point([p.x, p.y])) for p in points]
        fc = ee.FeatureCollection(features)
        
        sampled = composite.sampleRegions(
            collection=fc,
            scale=scale,
            geometries=True
        )
        
        data = sampled.getInfo()['features']
        
        records = []
        for i, feature in enumerate(data):
            props = feature['properties']
            coords = feature['geometry']['coordinates']
            
            record = {
                'point_id': start_id + i,
                'longitude': coords[0],
                'latitude': coords[1],
                **props
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        geometry = [Point(xy) for xy in zip(df.longitude, df.latitude)]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
        
        return gdf
    
    def export_to_geojson(self, gdf: gpd.GeoDataFrame, filename: str = None):
        """Exporta GeoDataFrame a GeoJSON con metadatos"""
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'indices_puros_{timestamp}.geojson'
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Lista de índices extraídos
        indices_disponibles = {
            "vegetacion": ["NDVI", "EVI", "SAVI", "NDRE", "GNDVI", "CIre", "LAI", "ARVI", "SIPI"],
            "humedad": ["NDWI", "NDMI", "MSI", "MoistureStress", "WDI"],
            "suelo": ["NBR", "BSI", "NDBI", "PSRI"],
            "topografia": ["Elevation", "Slope", "Aspect", "TRI", "TPI", "SolarExposure"]
        }
        
        metadata = {
            "type": "FeatureCollection",
            "metadata": {
                "title": "Índices Ecofisiológicos y Topográficos - Datos Puros",
                "description": "Datos sin procesamiento multicriterio para análisis con IA/clustering",
                "created": datetime.now().isoformat(),
                "region": "Arequipa, Perú",
                "date_range": f"{self.fecha_inicio} a {self.fecha_fin}",
                "source": "Sentinel-2 L2A, SRTM DEM",
                "crs": "EPSG:4326",
                "total_sites": len(gdf),
                "indices_disponibles": indices_disponibles,
                "uso_recomendado": "Clustering (K-means, DBSCAN), PCA, Random Forest"
            },
            "features": json.loads(gdf.to_json())['features']
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Exportado: {filepath}")
        return filepath
    
    def export_to_csv(self, gdf: gpd.GeoDataFrame, filename: str = None):
        """Exporta a CSV (sin geometría) para análisis en Python/R"""
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'indices_puros_{timestamp}.csv'
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Crear DataFrame sin geometría
        df = pd.DataFrame(gdf.drop(columns='geometry'))
        df.to_csv(filepath, index=False)
        
        print(f"📊 Exportado CSV: {filepath}")
        return filepath
    
    def export_summary_stats(self, gdf: gpd.GeoDataFrame):
        """Exporta estadísticas descriptivas"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        stats_file = os.path.join(self.output_dir, f'estadisticas_{timestamp}.csv')
        
        numeric_cols = gdf.select_dtypes(include=[np.number]).columns
        stats = gdf[numeric_cols].describe()
        stats.to_csv(stats_file)
        
        print(f"📈 Estadísticas exportadas: {stats_file}")
        
        # Matriz de correlación
        corr_file = os.path.join(self.output_dir, f'correlacion_{timestamp}.csv')
        corr = gdf[numeric_cols].corr()
        corr.to_csv(corr_file)
        
        print(f"🔗 Matriz de correlación: {corr_file}")
        
        return stats_file, corr_file
    
    # ========================================================================
    # FLUJO PRINCIPAL
    # ========================================================================
    
    def run_extraction(self, grid_size_m: int = 50, export_formats: List[str] = ['geojson', 'csv']):
        """
        Ejecuta extracción completa de índices
        
        Args:
            grid_size_m: Tamaño de celda en metros
            export_formats: Lista con 'geojson', 'csv', o ambos
        
        Returns:
            GeoDataFrame con todos los índices
        """
        
        print("="*60)
        print("🛰️  EXTRACCIÓN DE ÍNDICES ECOFISIOLÓGICOS")
        print("="*60)
        
        # 1. Procesar datos satelitales
        composite = self.process_satellite_data()
        
        # 2. Crear grid de muestreo
        points = self.create_sampling_grid(cell_size_meters=grid_size_m)
        
        # 3. Extraer valores
        gdf = self.extract_point_values(composite, points)
        
        # 4. Exportar en formatos solicitados
        if 'geojson' in export_formats:
            self.export_to_geojson(gdf)
        
        if 'csv' in export_formats:
            self.export_to_csv(gdf)
        
        # 5. Exportar estadísticas
        self.export_summary_stats(gdf)
        
        print("\n" + "="*60)
        print("✅ EXTRACCIÓN COMPLETADA")
        print(f"📦 Total de variables: {len(gdf.columns) - 3}")  # -3 por point_id, lon, lat
        print(f"📍 Total de puntos: {len(gdf)}")
        print("="*60)
        
        return gdf


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    
    PROJECT_ID = 'gen-lang-client-0247558984'
    
    coords = [
        [-71.3385389,  -16.43038162],
        [-71.33685562, -16.43528126],
        [-71.32475405, -16.43134153],
        [-71.3256297,  -16.42287724],
        [-71.32898083, -16.42349868],
        [-71.33447786, -16.42826592],
        [-71.3385389,  -16.43038162]
    ]
    
    fecha_inicio = '2024-01-01'
    fecha_fin = '2026-01-14'
    
    # Crear extractor
    extractor = SatelliteIndicesExtractor(
        project_id=PROJECT_ID,
        coords=coords,
        fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin
    )
    
    # Ejecutar extracción
    datos_puros = extractor.run_extraction(
        grid_size_m=5,
        export_formats=['geojson', 'csv']
    )
    
    print("\n📊 Primeras 5 filas del dataset:")
    print(datos_puros.head())
    
    print("\n📋 Variables disponibles:")
    print(datos_puros.columns.tolist())