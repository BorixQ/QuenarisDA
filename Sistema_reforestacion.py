import ee
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point, box
import json
import numpy as np
from datetime import datetime
import os
from typing import Dict, List, Tuple

# ============================================================================
# 1. CONFIGURACIÓN E INICIALIZACIÓN
# ============================================================================

class ReforestationEvaluator:
    """Sistema de evaluación ecofisiológica para priorización de micro-sitios"""
    
    def __init__(self, project_id: str, coords: List[List[float]], 
                 fecha_inicio: str, fecha_fin: str):
        self.project_id = project_id
        self.coords = coords
        self.fecha_inicio = fecha_inicio
        self.fecha_fin = fecha_fin
        self.output_dir = 'datos_reforestacion_gis'
        
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
    # 2. PROCESAMIENTO DE IMÁGENES SATELITALES
    # ========================================================================
    
    def mask_s2_clouds(self, image):
        """Máscara de nubes mejorada para Sentinel-2"""
        qa = image.select('QA60')
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
               qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        
        scl = image.select('SCL')
        # 4=Vegetation, 5=Not-vegetated, 6=Water
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
        # Moisture Stress Index (valores altos = estrés)
        moisture_stress = image.expression(
            '(SWIR1 - NIR) / (SWIR1 + NIR)', {
                'SWIR1': image.select('B11'),
                'NIR': image.select('B8')
            }).rename('MoistureStress')
        
        # Water Deficit Index
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
    # 3. ANÁLISIS TOPOGRÁFICO
    # ========================================================================
    
    def add_topographic_features(self):
        """Añade características topográficas del terreno"""
        dem = ee.Image('USGS/SRTMGL1_003').clip(self.roi)
        
        # Elevación
        elevation = dem.select('elevation').rename('Elevation')
        
        # Pendiente (grados)
        slope = ee.Terrain.slope(dem).rename('Slope')
        
        # Aspecto (orientación en grados)
        aspect = ee.Terrain.aspect(dem).rename('Aspect')
        
        # Índice de Rugosidad del Terreno (TRI)
        tri = dem.subtract(
            dem.focal_mean(radius=1, kernelType='square', units='pixels')
        ).abs().rename('TRI')
        
        # Índice de Posición Topográfica (TPI)
        tpi = dem.subtract(
            dem.focal_mean(radius=3, kernelType='circle', units='pixels')
        ).rename('TPI')
        
        # Clasificación de exposición solar (Norte vs Sur en Hemisferio Sur)
        # En Arequipa (Hemisferio Sur), laderas norte reciben más sol
        solar_exposure = aspect.expression(
            '(cos((aspect - 0) * 3.14159 / 180) + 1) / 2', {
                'aspect': aspect
            }).rename('SolarExposure')
        
        return ee.Image.cat([
            elevation, slope, aspect, tri, tpi, solar_exposure
        ])
    
    # ========================================================================
    # 4. SISTEMA DE PUNTUACIÓN MULTI-CRITERIO
    # ========================================================================
    
    def calculate_suitability_score(self, composite):
        """
        Calcula puntuación de idoneidad para reforestación (0-100)
        
        Criterios:
        - Vigor vegetal (30%): NDVI, EVI, LAI
        - Disponibilidad hídrica (25%): NDMI, NDWI, MoistureStress
        - Condiciones de suelo (20%): BSI, NBR
        - Topografía favorable (15%): Pendiente, Exposición solar
        - Bajo estrés (10%): WDI, PSRI
        """
        
        # Normalización de índices (0-1)
        def normalize(band, min_val, max_val):
            return band.subtract(min_val).divide(max_val - min_val).clamp(0, 1)
        
        # 1. VIGOR VEGETAL (30 puntos)
        ndvi_score = normalize(composite.select('NDVI'), 0, 0.8).multiply(10)
        evi_score = normalize(composite.select('EVI'), 0, 0.6).multiply(10)
        lai_score = normalize(composite.select('LAI'), 0, 5).multiply(10)
        vigor_score = ndvi_score.add(evi_score).add(lai_score)
        
        # 2. DISPONIBILIDAD HÍDRICA (25 puntos)
        ndmi_score = normalize(composite.select('NDMI'), -0.2, 0.4).multiply(10)
        ndwi_score = normalize(composite.select('NDWI'), -0.3, 0.3).multiply(8)
        # Invertir MoistureStress (bajo estrés = bueno)
        moisture_score = normalize(composite.select('MoistureStress'), -0.5, 0.5).multiply(-1).add(1).multiply(7)
        water_score = ndmi_score.add(ndwi_score).add(moisture_score)
        
        # 3. CONDICIONES DE SUELO (20 puntos)
        # Bajo BSI es mejor (menos suelo desnudo)
        bsi_score = normalize(composite.select('BSI'), -0.5, 0.5).multiply(-1).add(1).multiply(10)
        nbr_score = normalize(composite.select('NBR'), 0, 0.6).multiply(10)
        soil_score = bsi_score.add(nbr_score)
        
        # 4. TOPOGRAFÍA (15 puntos)
        # Pendiente óptima: 5-25 grados (evitar muy plano o muy empinado)
        slope_optimal = composite.select('Slope').subtract(15).abs().multiply(-1).add(15).clamp(0, 15).divide(15).multiply(8)
        # Exposición solar moderada (no extremos)
        solar_score = composite.select('SolarExposure').subtract(0.5).abs().multiply(-1).add(0.5).multiply(2).multiply(7)
        topo_score = slope_optimal.add(solar_score)
        
        # 5. BAJO ESTRÉS (10 puntos)
        wdi_score = normalize(composite.select('WDI'), 0, 2).multiply(-1).add(1).multiply(5)
        psri_score = normalize(composite.select('PSRI'), -0.2, 0.2).multiply(-1).add(1).multiply(5)
        stress_score = wdi_score.add(psri_score)
        
        # PUNTUACIÓN TOTAL (0-100)
        total_score = vigor_score.add(water_score).add(soil_score).add(topo_score).add(stress_score)
        
        return total_score.rename('SuitabilityScore')
    
    # ========================================================================
    # 5. PROCESAMIENTO PRINCIPAL
    # ========================================================================
    
    def process_satellite_data(self):
        """Procesa datos satelitales y genera composición"""
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
        
        # Calcular puntuación de idoneidad
        suitability = self.calculate_suitability_score(composite)
        composite = composite.addBands(suitability)
        
        print(f"✅ Procesadas {s2.size().getInfo()} imágenes")
        
        return composite
    
    # ========================================================================
    # 6. GENERACIÓN DE GRID DE MUESTREO
    # ========================================================================
    
    def create_sampling_grid(self, cell_size_meters: int = 50, max_points: int = None):
        """
        Crea grid regular de puntos de muestreo
        
        Args:
            cell_size_meters: Tamaño de celda en metros
            max_points: Límite máximo de puntos (None = sin límite)
        """
        
        # Obtener bounds del ROI
        bounds = self.roi.bounds().getInfo()['coordinates'][0]
        min_lon = min([p[0] for p in bounds])
        max_lon = max([p[0] for p in bounds])
        min_lat = min([p[1] for p in bounds])
        max_lat = max([p[1] for p in bounds])
        
        # Convertir metros a grados (aproximado)
        lat_center = (min_lat + max_lat) / 2
        meters_per_degree = 111320 * np.cos(np.radians(lat_center))
        cell_size_deg = cell_size_meters / meters_per_degree
        
        # Estimar número de puntos
        width_deg = max_lon - min_lon
        height_deg = max_lat - min_lat
        estimated_points = int((width_deg / cell_size_deg) * (height_deg / cell_size_deg))
        
        # # Advertencia si hay demasiados puntos
        # if estimated_points > 10000:
        #     print(f"⚠️ ADVERTENCIA: Se generarían ~{estimated_points:,} puntos")
        #     print(f"   Esto puede ser muy lento. Recomendaciones:")
        #     print(f"   - Usar cell_size_meters >= 25m (actual: {cell_size_meters}m)")
        #     print(f"   - Establecer max_points (ej: max_points=5000)")
            
        #     if max_points is None and estimated_points > 50000:
        #         print(f"   🛑 Limitando automáticamente a 10,000 puntos")
        #         max_points = 10000
        
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
                    
                    # Límite de puntos
                    if max_points and len(points) >= max_points:
                        print(f"⚠️ Alcanzado límite de {max_points:,} puntos")
                        print(f"📍 Generados {len(points):,} puntos de muestreo ({cell_size_meters}m)")
                        return points
        
        print(f"📍 Generados {len(points):,} puntos de muestreo ({cell_size_meters}m)")
        
        # Sugerencia si hay muy pocos puntos
        if len(points) < 50:
            print(f"   💡 Hay pocos puntos. Considera usar cell_size_meters < {cell_size_meters}m")
        
        return points
    
    # ========================================================================
    # 7. EXTRACCIÓN Y EXPORTACIÓN DE DATOS
    # ========================================================================
    
    def extract_point_values(self, composite, points: List[Point], 
                            scale: int = 10, batch_size: int = 5000) -> gpd.GeoDataFrame:
        """
        Extrae valores de todas las bandas en puntos de muestreo
        Procesa en lotes para evitar límites de GEE
        
        Args:
            composite: Imagen compuesta con todas las bandas
            points: Lista de puntos a muestrear
            scale: Escala de muestreo en metros
            batch_size: Tamaño de lote (máx recomendado: 5000)
        """
        
        total_points = len(points)
        print(f"🔍 Extrayendo valores en {total_points} puntos de muestreo...")
        
        # Si hay pocos puntos, procesar de una vez
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
                print(f"   Intentando con lotes más pequeños...")
                
                # Si falla, dividir el lote a la mitad
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
        
        # Combinar todos los lotes
        if all_gdfs:
            gdf_combined = pd.concat(all_gdfs, ignore_index=True)
            gdf_combined = gpd.GeoDataFrame(gdf_combined, geometry='geometry', crs='EPSG:4326')
            print(f"✅ Extraídos datos para {len(gdf_combined)} puntos")
            return gdf_combined
        else:
            raise Exception("No se pudieron extraer datos de ningún lote")
    
    def _extract_batch(self, composite, points: List[Point], scale: int = 10,
                      start_id: int = 1) -> gpd.GeoDataFrame:
        """Extrae un lote de puntos"""
        
        # Crear FeatureCollection de puntos
        features = [ee.Feature(ee.Geometry.Point([p.x, p.y])) for p in points]
        fc = ee.FeatureCollection(features)
        
        # Extraer valores
        sampled = composite.sampleRegions(
            collection=fc,
            scale=scale,
            geometries=True
        )
        
        # Convertir a DataFrame
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
        
        # Convertir a GeoDataFrame
        geometry = [Point(xy) for xy in zip(df.longitude, df.latitude)]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
        
        return gdf
    
    def classify_priority(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Clasifica sitios por prioridad de reforestación"""
        
        if 'SuitabilityScore' in gdf.columns:
            conditions = [
                (gdf['SuitabilityScore'] >= 75),
                (gdf['SuitabilityScore'] >= 50),
                (gdf['SuitabilityScore'] >= 25),
                (gdf['SuitabilityScore'] < 25)
            ]
            choices = ['Muy Alta', 'Alta', 'Media', 'Baja']
            gdf['Prioridad'] = np.select(conditions, choices, default='Sin Datos')
        
        return gdf
    
    def export_to_geojson(self, gdf: gpd.GeoDataFrame, filename: str = None):
        """Exporta GeoDataFrame a GeoJSON con metadatos"""
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'micrositios_reforestacion_{timestamp}.geojson'
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Añadir metadatos
        metadata = {
            "type": "FeatureCollection",
            "metadata": {
                "title": "Evaluación de Micro-sitios para Reforestación",
                "description": "Sistema de evaluación ecofisiológica multi-criterio",
                "created": datetime.now().isoformat(),
                "region": "Arequipa, Perú",
                "date_range": f"{self.fecha_inicio} a {self.fecha_fin}",
                "source": "Sentinel-2, SRTM DEM",
                "crs": "EPSG:4326",
                "total_sites": len(gdf),
                "criteria": {
                    "vigor_vegetal": "NDVI, EVI, LAI (30%)",
                    "disponibilidad_hidrica": "NDMI, NDWI, Moisture Stress (25%)",
                    "condiciones_suelo": "BSI, NBR (20%)",
                    "topografia": "Pendiente, Exposición Solar (15%)",
                    "bajo_estres": "WDI, PSRI (10%)"
                }
            },
            "features": json.loads(gdf.to_json())['features']
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Exportado: {filepath}")
        
        return filepath
    
    def export_summary_stats(self, gdf: gpd.GeoDataFrame):
        """Exporta estadísticas resumen"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Estadísticas descriptivas
        stats_file = os.path.join(self.output_dir, f'estadisticas_{timestamp}.csv')
        
        numeric_cols = gdf.select_dtypes(include=[np.number]).columns
        stats = gdf[numeric_cols].describe()
        stats.to_csv(stats_file)
        
        print(f"📊 Estadísticas exportadas: {stats_file}")
        
        # Resumen por prioridad
        if 'Prioridad' in gdf.columns:
            priority_summary = gdf.groupby('Prioridad').size()
            print("\n📋 Resumen de Prioridades:")
            print(priority_summary)
        
        return stats_file
    
    # ========================================================================
    # 8. FLUJO PRINCIPAL
    # ========================================================================
    
    def run_full_analysis(self, grid_size_m: int = 50, export_geojson: bool = True):
        """Ejecuta análisis completo"""
        
        print("="*60)
        print("🌳 SISTEMA DE EVALUACIÓN ECOFISIOLÓGICA PARA REFORESTACIÓN")
        print("="*60)
        
        # 1. Procesar datos satelitales
        composite = self.process_satellite_data()
        
        # 2. Crear grid de muestreo
        points = self.create_sampling_grid(cell_size_meters=grid_size_m)
        
        # 3. Extraer valores
        gdf = self.extract_point_values(composite, points)
        
        # 4. Clasificar prioridades
        gdf = self.classify_priority(gdf)
        
        # 5. Exportar resultados
        if export_geojson:
            geojson_path = self.export_to_geojson(gdf)
        
        stats_path = self.export_summary_stats(gdf)
        
        print("\n" + "="*60)
        print("✅ ANÁLISIS COMPLETADO")
        print("="*60)
        
        return gdf
    
    # ========================================================================
    # 9. ANÁLISIS TEMPORAL (NUEVO)
    # ========================================================================
    
    def run_temporal_analysis(self, grid_size_m: int = 50, 
                             temporal_step: str = 'monthly',
                             fixed_points: List[Point] = None):
        """
        Ejecuta análisis temporal sobre los mismos puntos
        
        Args:
            grid_size_m: Tamaño del grid en metros
            temporal_step: 'monthly', 'seasonal', o 'yearly'
            fixed_points: Lista de puntos fijos (si None, genera nuevos)
        
        Returns:
            Dict con GeoDataFrames por período
        """
        
        print("="*60)
        print("📅 ANÁLISIS TEMPORAL DE MICRO-SITIOS")
        print("="*60)
        
        # Generar o usar puntos fijos
        if fixed_points is None:
            points = self.create_sampling_grid(cell_size_meters=grid_size_m)
            print(f"📍 Generados {len(points)} puntos fijos para todo el análisis")
        else:
            points = fixed_points
            print(f"📍 Usando {len(points)} puntos previamente definidos")
        
        # Definir períodos temporales
        periods = self._create_time_periods(temporal_step)
        
        results = {}
        
        for period_name, (start, end) in periods.items():
            print(f"\n🔍 Procesando período: {period_name} ({start} a {end})")
            
            # Actualizar fechas temporalmente
            original_start = self.fecha_inicio
            original_end = self.fecha_fin
            
            self.fecha_inicio = start
            self.fecha_fin = end
            
            # Procesar datos
            try:
                composite = self.process_satellite_data()
                gdf = self.extract_point_values(composite, points)
                gdf = self.classify_priority(gdf)
                gdf['periodo'] = period_name
                
                results[period_name] = gdf
                
            except Exception as e:
                print(f"⚠️ Error en período {period_name}: {e}")
            
            # Restaurar fechas
            self.fecha_inicio = original_start
            self.fecha_fin = original_end
        
        # Exportar resultados temporales
        self._export_temporal_results(results)
        
        print("\n" + "="*60)
        print(f"✅ ANÁLISIS TEMPORAL COMPLETADO - {len(results)} períodos")
        print("="*60)
        
        return results
    
    def _create_time_periods(self, step: str) -> Dict[str, Tuple[str, str]]:
        """Crea períodos temporales según el paso"""
        
        from datetime import datetime
        from dateutil.relativedelta import relativedelta
        
        start = datetime.strptime(self.fecha_inicio, '%Y-%m-%d')
        end = datetime.strptime(self.fecha_fin, '%Y-%m-%d')
        
        periods = {}
        
        if step == 'monthly':
            current = start
            while current < end:
                next_month = current + relativedelta(months=1)
                period_name = current.strftime('%Y-%m')
                periods[period_name] = (
                    current.strftime('%Y-%m-%d'),
                    min(next_month, end).strftime('%Y-%m-%d')
                )
                current = next_month
        
        elif step == 'seasonal':
            # Hemisferio Sur: verano (dic-feb), otoño (mar-may), 
            # invierno (jun-ago), primavera (sep-nov)
            seasons = {
                'Verano_2024': ('2024-01-01', '2024-02-29'),
                'Otoño_2024': ('2024-03-01', '2024-05-31'),
                'Invierno_2024': ('2024-06-01', '2024-08-31'),
                'Primavera_2024': ('2024-09-01', '2024-11-30'),
                'Verano_2024_2025': ('2024-12-01', '2025-02-28'),
                'Otoño_2025': ('2025-03-01', '2025-05-31'),
                'Invierno_2025': ('2025-06-01', '2025-08-31'),
                'Primavera_2025': ('2025-09-01', '2025-11-30'),
                'Verano_2025_2026': ('2025-12-01', '2026-01-14')
            }
            periods = seasons
        
        elif step == 'yearly':
            periods = {
                '2024': ('2024-01-01', '2024-12-31'),
                '2025': ('2025-01-01', '2025-12-31'),
                '2026': ('2026-01-01', '2026-01-14')
            }
        
        return periods
    
    def _export_temporal_results(self, results: Dict):
        """Exporta resultados del análisis temporal"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Combinar todos los períodos en un solo GeoDataFrame
        all_periods = pd.concat(results.values(), ignore_index=True)
        gdf_combined = gpd.GeoDataFrame(all_periods, geometry='geometry', crs='EPSG:4326')
        
        # Exportar GeoJSON temporal
        temporal_file = os.path.join(
            self.output_dir, 
            f'analisis_temporal_{timestamp}.geojson'
        )
        
        metadata = {
            "type": "FeatureCollection",
            "metadata": {
                "title": "Análisis Temporal de Micro-sitios",
                "description": "Evolución temporal de indicadores ecofisiológicos",
                "created": datetime.now().isoformat(),
                "periods": list(results.keys()),
                "total_observations": len(gdf_combined)
            },
            "features": json.loads(gdf_combined.to_json())['features']
        }
        
        with open(temporal_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Exportado análisis temporal: {temporal_file}")
        
        # Exportar estadísticas por período
        stats_file = os.path.join(
            self.output_dir,
            f'estadisticas_temporales_{timestamp}.csv'
        )
        
        temporal_stats = gdf_combined.groupby('periodo').agg({
            'SuitabilityScore': ['mean', 'std', 'min', 'max'],
            'NDVI': ['mean', 'std'],
            'NDMI': ['mean', 'std'],
            'MoistureStress': ['mean', 'std']
        }).round(3)
        
        temporal_stats.to_csv(stats_file)
        print(f"📊 Exportadas estadísticas temporales: {stats_file}")
        
        return temporal_file


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    
    # Configuración
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
    
    # Crear evaluador
    evaluator = ReforestationEvaluator(
        project_id=PROJECT_ID,
        coords=coords,
        fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin
    )
    
    # ========================================================================
    # RECOMENDACIONES DE TAMAÑO DE GRID
    # ========================================================================
    # 
    # Para un área de ~1.5 km²:
    # - grid_size_m=50  → ~600 puntos   (ÓPTIMO para análisis detallado)
    # - grid_size_m=25  → ~2,400 puntos (detalle muy fino, más lento)
    # - grid_size_m=10  → ~15,000 puntos (extremadamente denso, muy lento)
    # - grid_size_m=5   → ~60,000 puntos (NO RECOMENDADO - demasiados)
    #
    # Si necesitas alta densidad, usa max_points para limitar:
    # ========================================================================
    
    # ========================================================================
    # OPCIÓN 1: Análisis único (snapshot)
    # ========================================================================
    print("\n" + "="*60)
    print("OPCIÓN 1: ANÁLISIS ÚNICO")
    print("="*60)
    
    resultados = evaluator.run_full_analysis(
        grid_size_m=5,  # CAMBIADO: 50m es más razonable que 5m
        export_geojson=True
    )
    
    print("\n📊 Top 5 sitios con mayor puntuación:")
    top_sites = resultados.nlargest(5, 'SuitabilityScore')[
        ['point_id', 'SuitabilityScore', 'Prioridad', 'NDVI', 'NDMI', 'Slope']
    ]
    print(top_sites)
    
    # ========================================================================
    # OPCIÓN ALTERNATIVA: Alta densidad con límite de puntos
    # ========================================================================
    # Descomentar si necesitas grid de 5m pero limitado a 5000 puntos:
    
    """
    print("\n" + "="*60)
    print("ANÁLISIS DE ALTA DENSIDAD (limitado)")
    print("="*60)
    
    # Modifica temporalmente el método
    points = evaluator.create_sampling_grid(
        cell_size_meters=5,
        max_points=5000  # Limita a 5000 puntos mejor distribuidos
    )
    
    composite = evaluator.process_satellite_data()
    gdf = evaluator.extract_point_values(composite, points, batch_size=1000)
    gdf = evaluator.classify_priority(gdf)
    evaluator.export_to_geojson(gdf, filename='alta_densidad_5m.geojson')
    """
    
    # ========================================================================
    # OPCIÓN 2: Análisis temporal (monitoreo)
    # ========================================================================
    # Descomentar para ejecutar análisis temporal:
    
    """
    print("\n" + "="*60)
    print("OPCIÓN 2: ANÁLISIS TEMPORAL")
    print("="*60)
    
    # Usar los mismos puntos del análisis anterior
    points_fixed = [Point(x, y) for x, y in 
                    zip(resultados.longitude, resultados.latitude)]
    
    # Análisis estacional
    temporal_results = evaluator.run_temporal_analysis(
        grid_size_m=50,
        temporal_step='seasonal',  # 'monthly', 'seasonal', 'yearly'
        fixed_points=points_fixed  # Usa los mismos puntos
    )
    
    # Comparar cambios entre estaciones
    print("\n📈 Cambios en NDVI promedio por estación:")
    for periodo, gdf in temporal_results.items():
        ndvi_mean = gdf['NDVI'].mean()
        print(f"{periodo}: {ndvi_mean:.3f}")
    """