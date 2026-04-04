import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import json
import os
from datetime import datetime
from typing import List, Dict, Tuple
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# MÓDULO 3: SELECCIÓN ÓPTIMA DE SITIOS CON IA
# ============================================================================

class OptimalSiteSelector:
    """
    Sistema de selección inteligente de sitios para plantación usando:
    1. Clustering espacial (DBSCAN)
    2. Ranking por condiciones ecofisiológicas
    3. Distribución espacial equilibrada
    4. Detección de outliers (sitios excepcionales)
    """
    
    def __init__(self, output_dir: str = 'sitios_plantacion'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Configuración de características para análisis
        self.feature_groups = {
            'vigor': ['NDVI', 'EVI', 'LAI', 'GNDVI'],
            'agua': ['NDMI', 'NDWI', 'MoistureStress'],
            'suelo': ['BSI', 'NBR'],
            'topografia': ['Slope', 'Elevation', 'SolarExposure']
        }
    
    # ========================================================================
    # 1. PREPARACIÓN DE DATOS
    # ========================================================================
    
    def prepare_features(self, gdf: gpd.GeoDataFrame, 
                        feature_importance: Dict[str, float] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Prepara matriz de características para análisis
        
        Args:
            gdf: GeoDataFrame con índices del Módulo 1
            feature_importance: Pesos personalizados por grupo
                               Ejemplo: {'vigor': 0.4, 'agua': 0.3, 'suelo': 0.2, 'topografia': 0.1}
        
        Returns:
            X_scaled: Matriz normalizada
            feature_names: Nombres de características usadas
        """
        
        if feature_importance is None:
            # Pesos por defecto
            feature_importance = {
                'vigor': 0.35,       # 35% - Más importante
                'agua': 0.30,        # 30% - Crítico para supervivencia
                'suelo': 0.20,       # 20% - Importante
                'topografia': 0.15   # 15% - Moderadamente importante
            }
        
        print("📊 Preparando características...")
        print(f"   Pesos: {feature_importance}")
        
        # Recolectar todas las características disponibles
        all_features = []
        feature_names = []
        
        for group, features in self.feature_groups.items():
            available = [f for f in features if f in gdf.columns]
            if available:
                all_features.extend(available)
                feature_names.extend(available)
        
        # Crear matriz de características
        X = gdf[all_features].values
        
        # Manejar valores NaN
        if np.any(np.isnan(X)):
            print("   ⚠️ Detectados valores NaN - imputando con medianas")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        # Normalizar (StandardScaler)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print(f"   ✅ {len(feature_names)} características preparadas")
        
        return X_scaled, feature_names
    
    # ========================================================================
    # 2. CLUSTERING ESPACIAL (DBSCAN)
    # ========================================================================
    
    def apply_dbscan_clustering(self, gdf: gpd.GeoDataFrame, 
                                eps_meters: float = 100,
                                min_samples: int = 5) -> gpd.GeoDataFrame:
        """
        Aplica clustering DBSCAN basado en coordenadas geográficas
        
        Args:
            eps_meters: Radio de vecindad en metros
            min_samples: Mínimo de puntos para formar cluster
        
        Returns:
            GeoDataFrame con columna 'cluster_spatial'
        """
        
        print(f"\n🗺️  Aplicando DBSCAN espacial (ε={eps_meters}m, min={min_samples})...")
        
        # Convertir coordenadas geográficas a metros (aproximado)
        coords = gdf[['longitude', 'latitude']].values
        
        # Convertir grados a metros (factor aproximado)
        lat_center = gdf['latitude'].mean()
        meters_per_deg_lat = 111320
        meters_per_deg_lon = 111320 * np.cos(np.radians(lat_center))
        
        coords_meters = coords.copy()
        coords_meters[:, 0] = coords[:, 0] * meters_per_deg_lon
        coords_meters[:, 1] = coords[:, 1] * meters_per_deg_lat
        
        # DBSCAN
        dbscan = DBSCAN(eps=eps_meters, min_samples=min_samples, metric='euclidean')
        gdf['cluster_spatial'] = dbscan.fit_predict(coords_meters)
        
        n_clusters = len(set(gdf['cluster_spatial'])) - (1 if -1 in gdf['cluster_spatial'] else 0)
        n_noise = list(gdf['cluster_spatial']).count(-1)
        
        print(f"   ✅ Identificados {n_clusters} clusters espaciales")
        print(f"   ⚠️ {n_noise} puntos clasificados como ruido (aislados)")
        
        return gdf
    
    # ========================================================================
    # 3. SCORING ECOFISIOLÓGICO
    # ========================================================================
    
    def calculate_survival_score(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Calcula puntuación de probabilidad de supervivencia (0-100)
        Basado en condiciones óptimas para cada índice
        """
        
        print("\n🌱 Calculando scores de supervivencia...")
        
        # Función auxiliar de normalización
        def normalize(series, optimal_min, optimal_max, hard_min=None, hard_max=None):
            """
            Normaliza dando máxima puntuación al rango óptimo
            """
            if hard_min is None:
                hard_min = series.min()
            if hard_max is None:
                hard_max = series.max()
            
            # Crear score basado en proximidad al rango óptimo
            score = np.zeros(len(series))
            
            for i, val in enumerate(series):
                if optimal_min <= val <= optimal_max:
                    score[i] = 1.0  # Óptimo
                elif val < optimal_min:
                    # Penalizar proporcionalmente
                    score[i] = max(0, (val - hard_min) / (optimal_min - hard_min))
                else:  # val > optimal_max
                    score[i] = max(0, (hard_max - val) / (hard_max - optimal_max))
            
            return score
        
        # VIGOR VEGETAL (25 puntos)
        # NDVI óptimo: 0.4-0.7 (vegetación moderada-vigorosa)
        ndvi_score = normalize(gdf['NDVI'], 0.4, 0.7, 0, 0.9) * 15
        # LAI óptimo: 2-4 (cobertura adecuada sin competencia excesiva)
        lai_score = normalize(gdf['LAI'], 2, 4, 0, 6) * 10
        
        vigor_total = ndvi_score + lai_score
        
        # DISPONIBILIDAD HÍDRICA (30 puntos) - MÁS CRÍTICO
        # NDMI óptimo: 0.2-0.4 (humedad moderada-alta)
        ndmi_score = normalize(gdf['NDMI'], 0.2, 0.4, -0.5, 0.6) * 15
        # MoistureStress óptimo: -0.2-0.1 (bajo estrés)
        moisture_score = normalize(gdf['MoistureStress'], -0.2, 0.1, -0.5, 0.5) * 15
        
        agua_total = ndmi_score + moisture_score
        
        # CONDICIONES DE SUELO (20 puntos)
        # BSI óptimo: -0.3-0.1 (poco suelo desnudo)
        bsi_score = normalize(gdf['BSI'], -0.3, 0.1, -0.5, 0.5) * 10
        # NBR óptimo: 0.3-0.6 (vegetación saludable)
        nbr_score = normalize(gdf['NBR'], 0.3, 0.6, -0.5, 0.8) * 10
        
        suelo_total = bsi_score + nbr_score
        
        # TOPOGRAFÍA (25 puntos)
        # Pendiente óptima: 5-20 grados (evita erosión, permite drenaje)
        slope_score = normalize(gdf['Slope'], 5, 20, 0, 45) * 15
        # Exposición solar óptima: 0.3-0.7 (ni muy norte ni muy sur)
        solar_score = normalize(gdf['SolarExposure'], 0.3, 0.7, 0, 1) * 10
        
        topo_total = slope_score + solar_score
        
        # SCORE TOTAL (0-100)
        gdf['survival_score'] = (vigor_total + agua_total + suelo_total + topo_total).round(2)
        
        print(f"   ✅ Score mín: {gdf['survival_score'].min():.1f}, máx: {gdf['survival_score'].max():.1f}")
        print(f"   📊 Promedio: {gdf['survival_score'].mean():.1f} ± {gdf['survival_score'].std():.1f}")
        
        return gdf
    
    # ========================================================================
    # 4. DETECCIÓN DE SITIOS EXCEPCIONALES
    # ========================================================================
    
    def detect_exceptional_sites(self, gdf: gpd.GeoDataFrame, X_scaled: np.ndarray,
                                 contamination: float = 0.05) -> gpd.GeoDataFrame:
        """
        Detecta sitios excepcionales usando Isolation Forest
        
        Args:
            contamination: Fracción esperada de outliers (ej: 0.05 = 5% mejores)
        """
        
        print(f"\n🌟 Detectando sitios excepcionales (top {contamination*100:.0f}%)...")
        
        # Isolation Forest (detecta outliers = sitios excepcionales)
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        gdf['is_exceptional'] = iso_forest.fit_predict(X_scaled)
        
        # Convertir: -1 (outlier/excepcional) → True, 1 (normal) → False
        gdf['is_exceptional'] = gdf['is_exceptional'] == -1
        
        n_exceptional = gdf['is_exceptional'].sum()
        print(f"   ✅ Identificados {n_exceptional} sitios excepcionales")
        
        return gdf
    
    # ========================================================================
    # 5. SELECCIÓN INTELIGENTE DE N MEJORES SITIOS
    # ========================================================================
    
    def select_top_sites(self, gdf: gpd.GeoDataFrame, n_sites: int = 1000,
                        strategy: str = 'balanced',
                        min_distance_m: float = 3) -> gpd.GeoDataFrame:
        """
        Selecciona los N mejores sitios para plantación
        
        Args:
            n_sites: Número de sitios a seleccionar
            strategy: Estrategia de selección:
                - 'best': Top N por score directo
                - 'balanced': Balance entre score y distribución espacial
                - 'clustered': Prioriza clusters densos con buen score
            min_distance_m: Distancia mínima entre sitios (metros)
        
        Returns:
            GeoDataFrame con N sitios seleccionados
        """
        
        print(f"\n🎯 Seleccionando {n_sites} sitios óptimos (estrategia: {strategy})...")
        
        if len(gdf) <= n_sites:
            print(f"   ⚠️ Hay menos puntos ({len(gdf)}) que sitios solicitados")
            return gdf.copy()
        
        if strategy == 'best':
            # Estrategia 1: Top N directo por score
            selected = gdf.nlargest(n_sites, 'survival_score').copy()
        
        elif strategy == 'balanced':
            # Estrategia 2: Balance score + distribución espacial
            selected = self._balanced_selection(gdf, n_sites, min_distance_m)
        
        elif strategy == 'clustered':
            # Estrategia 3: Priorizar clusters buenos
            selected = self._clustered_selection(gdf, n_sites)
        
        else:
            raise ValueError(f"Estrategia desconocida: {strategy}")
        
        # Asignar IDs de plántula
        selected = selected.reset_index(drop=True)
        selected['plantula_id'] = [f"PLANT_{i+1:04d}" for i in range(len(selected))]
        
        print(f"   ✅ Seleccionados {len(selected)} sitios")
        print(f"   📊 Score promedio: {selected['survival_score'].mean():.1f}")
        
        return selected
    
    def _balanced_selection(self, gdf: gpd.GeoDataFrame, n_sites: int, 
                           min_distance_m: float) -> gpd.GeoDataFrame:
        """Selección balanceada con distribución espacial"""
        
        # Ordenar por score descendente
        gdf_sorted = gdf.sort_values('survival_score', ascending=False).copy()
        
        # Convertir coordenadas a metros
        lat_center = gdf['latitude'].mean()
        meters_per_deg_lon = 111320 * np.cos(np.radians(lat_center))
        meters_per_deg_lat = 111320
        
        selected_indices = []
        selected_coords_m = []
        
        for idx, row in gdf_sorted.iterrows():
            if len(selected_indices) >= n_sites:
                break
            
            # Coordenadas en metros
            x_m = row['longitude'] * meters_per_deg_lon
            y_m = row['latitude'] * meters_per_deg_lat
            
            # Verificar distancia mínima con sitios ya seleccionados
            too_close = False
            for sx, sy in selected_coords_m:
                dist = np.sqrt((x_m - sx)**2 + (y_m - sy)**2)
                if dist < min_distance_m:
                    too_close = True
                    break
            
            if not too_close:
                selected_indices.append(idx)
                selected_coords_m.append((x_m, y_m))
        
        return gdf.loc[selected_indices].copy()
    
    def _clustered_selection(self, gdf: gpd.GeoDataFrame, n_sites: int) -> gpd.GeoDataFrame:
        """Selección priorizando clusters con buen score"""
        
        # Calcular score promedio por cluster
        cluster_stats = gdf[gdf['cluster_spatial'] >= 0].groupby('cluster_spatial').agg({
            'survival_score': ['mean', 'count'],
            'point_id': 'first'
        })
        
        cluster_stats.columns = ['mean_score', 'count', 'first_point']
        cluster_stats['weighted_score'] = cluster_stats['mean_score'] * np.log1p(cluster_stats['count'])
        cluster_stats = cluster_stats.sort_values('weighted_score', ascending=False)
        
        # Seleccionar de clusters buenos
        selected = []
        remaining = n_sites
        
        for cluster_id in cluster_stats.index:
            cluster_data = gdf[gdf['cluster_spatial'] == cluster_id].nlargest(
                min(remaining, len(gdf[gdf['cluster_spatial'] == cluster_id])),
                'survival_score'
            )
            selected.append(cluster_data)
            remaining -= len(cluster_data)
            
            if remaining <= 0:
                break
        
        # Si faltan sitios, agregar de puntos aislados (ruido) con mejor score
        if remaining > 0:
            noise_points = gdf[gdf['cluster_spatial'] == -1].nlargest(remaining, 'survival_score')
            selected.append(noise_points)
        
        return pd.concat(selected, ignore_index=True)
    
    # ========================================================================
    # 6. EXPORTACIÓN
    # ========================================================================
    
    def export_planting_sites(self, gdf: gpd.GeoDataFrame, 
                             filename: str = None,
                             include_metadata: bool = True):
        """
        Exporta GeoJSON con ubicaciones para plantación
        
        Formato optimizado para uso en campo con GPS
        """
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'sitios_plantacion_{len(gdf)}_{timestamp}.geojson'
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Simplificar columnas para campo
        export_cols = ['plantula_id', 'longitude', 'latitude', 'survival_score',
                      'cluster_spatial', 'is_exceptional']
        
        # Añadir índices clave
        key_indices = ['NDVI', 'NDMI', 'Slope', 'Elevation']
        for col in key_indices:
            if col in gdf.columns:
                export_cols.append(col)
        
        gdf_export = gdf[export_cols + ['geometry']].copy()
        
        if include_metadata:
            metadata = {
                "type": "FeatureCollection",
                "metadata": {
                    "title": "Sitios Óptimos para Plantación",
                    "description": "Ubicaciones seleccionadas con IA para máxima supervivencia",
                    "created": datetime.now().isoformat(),
                    "total_sites": len(gdf_export),
                    "selection_method": "DBSCAN + Survival Score Ranking",
                    "average_survival_score": float(gdf_export['survival_score'].mean().round(2)),
                    "exceptional_sites": int(gdf_export['is_exceptional'].sum()),
                    "spatial_clusters": int(gdf_export['cluster_spatial'].max() + 1),
                    "coordinate_system": "EPSG:4326 (WGS84)",
                    "usage": "Usar plantula_id y coordenadas (longitude, latitude) para navegación GPS"
                },
                "features": json.loads(gdf_export.to_json())['features']
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        else:
            gdf_export.to_file(filepath, driver='GeoJSON')
        
        print(f"\n💾 Exportado: {filepath}")
        return filepath
    
    def export_field_summary(self, gdf: gpd.GeoDataFrame):
        """Exporta resumen ejecutivo para campo"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # CSV simple para campo
        csv_file = os.path.join(self.output_dir, f'listado_plantacion_{timestamp}.csv')
        
        field_data = gdf[['plantula_id', 'latitude', 'longitude', 'survival_score', 
                         'cluster_spatial', 'is_exceptional']].copy()
        field_data = field_data.sort_values('plantula_id')
        field_data.to_csv(csv_file, index=False)
        
        print(f"📋 CSV de campo: {csv_file}")
        
        # Resumen estadístico
        summary_file = os.path.join(self.output_dir, f'resumen_campana_{timestamp}.txt')
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("RESUMEN DE CAMPAÑA DE REFORESTACIÓN\n")
            f.write("="*70 + "\n\n")
            f.write(f"📅 Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"🌱 Total de sitios: {len(gdf)}\n\n")
            
            f.write("📊 ESTADÍSTICAS DE SUPERVIVENCIA:\n")
            f.write(f"   • Score promedio: {gdf['survival_score'].mean():.1f}/100\n")
            f.write(f"   • Score mínimo: {gdf['survival_score'].min():.1f}/100\n")
            f.write(f"   • Score máximo: {gdf['survival_score'].max():.1f}/100\n\n")
            
            f.write("🌟 SITIOS EXCEPCIONALES:\n")
            n_exceptional = gdf['is_exceptional'].sum()
            f.write(f"   • Total: {n_exceptional} ({n_exceptional/len(gdf)*100:.1f}%)\n\n")
            
            f.write("🗺️  DISTRIBUCIÓN ESPACIAL:\n")
            n_clusters = gdf['cluster_spatial'].max() + 1
            f.write(f"   • Clusters identificados: {n_clusters}\n")
            f.write(f"   • Sitios por cluster (promedio): {len(gdf)/n_clusters:.0f}\n\n")
            
            f.write("💡 RECOMENDACIONES:\n")
            f.write("   1. Priorizar sitios con is_exceptional = True\n")
            f.write("   2. Plantar en orden descendente de survival_score\n")
            f.write("   3. Usar coordenadas (latitude, longitude) con GPS de precisión\n")
            f.write("   4. Considerar cluster_spatial para logística de brigadas\n\n")
            
            f.write("="*70 + "\n")
        
        print(f"📄 Resumen ejecutivo: {summary_file}")
        
        return csv_file, summary_file
    
    # ========================================================================
    # 7. FLUJO PRINCIPAL
    # ========================================================================
    
    def run_selection(self, gdf: gpd.GeoDataFrame, 
                     n_sites: int = 1000,
                     strategy: str = 'balanced',
                     eps_meters: float = 100,
                     min_distance_m: float = 3) -> gpd.GeoDataFrame:
        """
        Pipeline completo de selección de sitios
        
        Args:
            gdf: GeoDataFrame del Módulo 1 (con índices puros)
            n_sites: Número de sitios a seleccionar
            strategy: 'best', 'balanced', o 'clustered'
            eps_meters: Radio DBSCAN para clustering espacial
            min_distance_m: Distancia mínima entre sitios (solo en 'balanced')
        
        Returns:
            GeoDataFrame con sitios seleccionados
        """
        
        print("="*70)
        print("🤖 SELECCIÓN INTELIGENTE DE SITIOS CON IA")
        print("="*70)
        
        # 1. Preparar features
        X_scaled, feature_names = self.prepare_features(gdf)
        
        # 2. Clustering espacial
        gdf = self.apply_dbscan_clustering(gdf, eps_meters=eps_meters)
        
        # 3. Calcular scores de supervivencia
        gdf = self.calculate_survival_score(gdf)
        
        # 4. Detectar sitios excepcionales
        gdf = self.detect_exceptional_sites(gdf, X_scaled)
        
        # 5. Seleccionar top N sitios
        selected_sites = self.select_top_sites(
            gdf, 
            n_sites=n_sites, 
            strategy=strategy,
            min_distance_m=min_distance_m
        )
        
        # 6. Exportar resultados
        self.export_planting_sites(selected_sites)
        self.export_field_summary(selected_sites)
        
        print("\n" + "="*70)
        print("✅ SELECCIÓN COMPLETADA")
        print("="*70)
        
        return selected_sites


# ============================================================================
# EJEMPLO DE USO COMPLETO
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("EJEMPLO: FLUJO COMPLETO MÓDULO 1 → MÓDULO 3")
    print("="*70)
    
    # ========================================================================
    # PASO 1: Ejecutar Módulo 1 para obtener datos reales
    # ========================================================================
    
    print("\n📡 Ejecutando Módulo 1: Extracción de índices satelitales...")
    
    # Importar Módulo 1
    import sys
    # sys.path.append('ruta/a/modulo1')  # Si está en otra carpeta
    
    # OPCIÓN A: Ejecutar Módulo 1 completo (recomendado)
    try:
        from modulo1_extraccion_indices import SatelliteIndicesExtractor
        
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
        
        extractor = SatelliteIndicesExtractor(
            project_id=PROJECT_ID,
            coords=coords,
            fecha_inicio='2024-01-01',
            fecha_fin='2026-01-14'
        )
        
        # Extraer datos reales
        gdf_indices = extractor.run_extraction(
            grid_size_m=5,  # Grid de 50m
            export_formats=['geojson', 'csv']
        )
        
        print(f"✅ Obtenidos {len(gdf_indices)} sitios candidatos del Módulo 1")
    
    except ImportError:
        print("⚠️ Módulo 1 no encontrado. Intentando cargar desde archivos...")
        
        # OPCIÓN B: Cargar desde archivos ya generados
        import glob
        import os
        
        # Buscar el archivo más reciente
        geojson_files = glob.glob('datos_indices_puros/indices_puros_*.geojson')
        csv_files = glob.glob('datos_indices_puros/indices_puros_*.csv')
        
        if geojson_files:
            latest_file = max(geojson_files, key=os.path.getctime)
            print(f"📂 Cargando: {latest_file}")
            gdf_indices = gpd.read_file(latest_file)
            print(f"✅ Cargados {len(gdf_indices)} sitios candidatos")
        
        elif csv_files:
            latest_file = max(csv_files, key=os.path.getctime)
            print(f"📂 Cargando: {latest_file}")
            df = pd.read_csv(latest_file)
            gdf_indices = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df.longitude, df.latitude),
                crs='EPSG:4326'
            )
            print(f"✅ Cargados {len(gdf_indices)} sitios candidatos")
        
        else:
            print("❌ No se encontraron datos del Módulo 1")
            print("   Por favor, ejecuta primero el Módulo 1 o proporciona la ruta del archivo")
            print("\n💡 Para ejecutar con datos de ejemplo, descomentar el bloque siguiente:")
            print("""
    # DATOS DE EJEMPLO (solo para testing sin GEE)
    # np.random.seed(42)
    # n_points = 5000
    # data = {
    #     'point_id': range(1, n_points + 1),
    #     'longitude': np.random.uniform(-71.34, -71.32, n_points),
    #     'latitude': np.random.uniform(-16.44, -16.42, n_points),
    #     'NDVI': np.random.uniform(0.1, 0.8, n_points),
    #     # ... resto de índices
    # }
    # gdf_indices = gpd.GeoDataFrame(data, geometry=..., crs='EPSG:4326')
            """)
            sys.exit(1)
    
    # ========================================================================
    # PASO 2: Ejecutar selección con IA
    # ========================================================================
    
    selector = OptimalSiteSelector()
    
    sitios_seleccionados = selector.run_selection(
        gdf=gdf_indices,
        n_sites=1000,              # 1000 plántulas
        strategy='balanced',        # Distribución espacial equilibrada
        eps_meters=100,            # Clusters de ~100m de radio
        min_distance_m=3           # Mínimo 3m entre plántulas
    )
    
    # ========================================================================
    # PASO 3: Análisis de resultados
    # ========================================================================
    
    print("\n" + "="*70)
    print("📊 ANÁLISIS DE RESULTADOS")
    print("="*70)
    
    print("\n🏆 Top 10 sitios por score de supervivencia:")
    top10 = sitios_seleccionados.nlargest(10, 'survival_score')[
        ['plantula_id', 'survival_score', 'NDVI', 'NDMI', 'Slope', 'is_exceptional']
    ]
    print(top10.to_string(index=False))
    
    print("\n🌟 Sitios excepcionales:")
    exceptional = sitios_seleccionados[sitios_seleccionados['is_exceptional']]
    print(f"   Total: {len(exceptional)}")
    print(f"   Score promedio: {exceptional['survival_score'].mean():.1f}")
    
    print("\n🗺️  Distribución por cluster:")
    cluster_dist = sitios_seleccionados['cluster_spatial'].value_counts().head(5)
    print(cluster_dist)
    
    print("\n" + "="*70)
    print("✅ LISTOS PARA PLANTACIÓN")
    print(f"📍 Usa el archivo GeoJSON con GPS en campo")
    print(f"📋 {len(sitios_seleccionados)} sitios optimizados")
    print("="*70)