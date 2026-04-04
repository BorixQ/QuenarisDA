"""
═══════════════════════════════════════════════════════════════════════════════
PIPELINE COMPLETO DE REFORESTACIÓN
═══════════════════════════════════════════════════════════════════════════════

Flujo de trabajo:
1. Módulo 1: Extracción de índices satelitales (Sentinel-2 + SRTM)
2. Módulo 3: Selección inteligente de sitios con IA (DBSCAN + Ranking)
3. Exportación de ubicaciones GPS para plantación

Autor: Sistema de Evaluación Ecofisiológica
Fecha: 2025
"""

import sys
import os
from datetime import datetime

# ============================================================================
# CONFIGURACIÓN DEL PROYECTO
# ============================================================================

class ConfiguracionReforestacion:
    """Configuración centralizada del proyecto"""
    
    # Google Earth Engine
    PROJECT_ID = 'gen-lang-client-0247558984'
    
    # Área de estudio (polígono en coordenadas WGS84)
    COORDS = [
        [-71.3385389,  -16.43038162],
        [-71.33685562, -16.43528126],
        [-71.32475405, -16.43134153],
        [-71.3256297,  -16.42287724],
        [-71.32898083, -16.42349868],
        [-71.33447786, -16.42826592],
        [-71.3385389,  -16.43038162]
    ]
    
    # Período temporal
    FECHA_INICIO = '2024-01-01'
    FECHA_FIN = '2026-01-14'
    
    # Parámetros de muestreo
    GRID_SIZE_M = 50  # Tamaño de celda en metros (50m = ~600 puntos para esta área)
    
    # Parámetros de selección
    N_PLANTULAS = 1000  # Número de plántulas a plantar
    ESTRATEGIA = 'balanced'  # 'best', 'balanced', 'clustered'
    EPS_METERS = 100  # Radio de clustering DBSCAN
    MIN_DISTANCE_M = 5  # Distancia mínima entre plántulas (metros)
    
    # Opciones de exportación
    EXPORT_FORMATS = ['geojson', 'csv']


# ============================================================================
# PASO 1: EXTRACCIÓN DE ÍNDICES SATELITALES
# ============================================================================

def ejecutar_modulo1(config):
    """
    Ejecuta el Módulo 1: Extracción de índices ecofisiológicos
    
    Returns:
        GeoDataFrame con índices puros
    """
    
    print("\n" + "="*80)
    print("PASO 1: EXTRACCIÓN DE ÍNDICES SATELITALES Y TOPOGRÁFICOS")
    print("="*80)
    
    try:
        # Importar Módulo 1
        from modulo1_extraccion_indices import SatelliteIndicesExtractor
        
        # Crear extractor
        extractor = SatelliteIndicesExtractor(
            project_id=config.PROJECT_ID,
            coords=config.COORDS,
            fecha_inicio=config.FECHA_INICIO,
            fecha_fin=config.FECHA_FIN
        )
        
        # Ejecutar extracción
        print(f"\n⚙️ Configuración:")
        print(f"   • Grid: {config.GRID_SIZE_M}m")
        print(f"   • Período: {config.FECHA_INICIO} a {config.FECHA_FIN}")
        print(f"   • Área: {len(config.COORDS)-1} vértices\n")
        
        gdf_indices = extractor.run_extraction(
            grid_size_m=config.GRID_SIZE_M,
            export_formats=config.EXPORT_FORMATS
        )
        
        print(f"\n✅ Módulo 1 completado exitosamente")
        print(f"   • Total de puntos candidatos: {len(gdf_indices)}")
        print(f"   • Variables extraídas: {len(gdf_indices.columns) - 3}")
        
        return gdf_indices
    
    except ImportError as e:
        print(f"\n❌ Error: No se pudo importar el Módulo 1")
        print(f"   {str(e)}")
        print(f"\n💡 Solución:")
        print(f"   1. Verifica que 'modulo1_extraccion_indices.py' esté en el mismo directorio")
        print(f"   2. O ajusta sys.path para incluir la ruta del módulo")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Error al ejecutar Módulo 1: {str(e)}")
        print(f"\n💡 Intentando cargar datos previamente generados...")
        
        # Intentar cargar desde archivos existentes
        return cargar_datos_existentes()


# ============================================================================
# PASO 2: SELECCIÓN DE SITIOS CON IA
# ============================================================================

def ejecutar_modulo3(gdf_indices, config):
    """
    Ejecuta el Módulo 3: Selección inteligente de sitios
    
    Args:
        gdf_indices: GeoDataFrame del Módulo 1
        config: Configuración del proyecto
    
    Returns:
        GeoDataFrame con sitios seleccionados
    """
    
    print("\n" + "="*80)
    print("PASO 2: SELECCIÓN INTELIGENTE DE SITIOS CON IA")
    print("="*80)
    
    try:
        # Importar Módulo 3
        from modulo3_seleccion_ia import OptimalSiteSelector
        
        # Crear selector
        selector = OptimalSiteSelector()
        
        # Ejecutar selección
        print(f"\n⚙️ Configuración de selección:")
        print(f"   • Plántulas objetivo: {config.N_PLANTULAS}")
        print(f"   • Estrategia: {config.ESTRATEGIA}")
        print(f"   • Radio clustering: {config.EPS_METERS}m")
        print(f"   • Distancia mínima: {config.MIN_DISTANCE_M}m\n")
        
        sitios_seleccionados = selector.run_selection(
            gdf=gdf_indices,
            n_sites=config.N_PLANTULAS,
            strategy=config.ESTRATEGIA,
            eps_meters=config.EPS_METERS,
            min_distance_m=config.MIN_DISTANCE_M
        )
        
        print(f"\n✅ Módulo 3 completado exitosamente")
        print(f"   • Sitios seleccionados: {len(sitios_seleccionados)}")
        print(f"   • Score promedio: {sitios_seleccionados['survival_score'].mean():.1f}/100")
        
        return sitios_seleccionados
    
    except ImportError as e:
        print(f"\n❌ Error: No se pudo importar el Módulo 3")
        print(f"   {str(e)}")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Error al ejecutar Módulo 3: {str(e)}")
        raise


# ============================================================================
# PASO 3: ANÁLISIS Y REPORTES
# ============================================================================

def generar_reporte_final(sitios_seleccionados, config):
    """Genera reporte final de la campaña"""
    
    print("\n" + "="*80)
    print("PASO 3: GENERACIÓN DE REPORTES FINALES")
    print("="*80)
    
    print("\n📊 ESTADÍSTICAS DE LA CAMPAÑA:")
    print(f"   • Total de sitios: {len(sitios_seleccionados)}")
    print(f"   • Score promedio: {sitios_seleccionados['survival_score'].mean():.2f}")
    print(f"   • Score mínimo: {sitios_seleccionados['survival_score'].min():.2f}")
    print(f"   • Score máximo: {sitios_seleccionados['survival_score'].max():.2f}")
    
    n_exceptional = sitios_seleccionados['is_exceptional'].sum()
    print(f"\n🌟 SITIOS EXCEPCIONALES:")
    print(f"   • Total: {n_exceptional} ({n_exceptional/len(sitios_seleccionados)*100:.1f}%)")
    print(f"   • Score promedio: {sitios_seleccionados[sitios_seleccionados['is_exceptional']]['survival_score'].mean():.2f}")
    
    n_clusters = len(sitios_seleccionados['cluster_spatial'].unique())
    print(f"\n🗺️  DISTRIBUCIÓN ESPACIAL:")
    print(f"   • Clusters identificados: {n_clusters}")
    print(f"   • Promedio por cluster: {len(sitios_seleccionados)/n_clusters:.0f} sitios")
    
    print(f"\n🏆 TOP 10 SITIOS:")
    top10 = sitios_seleccionados.nlargest(10, 'survival_score')[
        ['plantula_id', 'survival_score', 'NDVI', 'NDMI', 'Slope', 'is_exceptional']
    ]
    print(top10.to_string(index=False))
    
    print(f"\n📋 DISTRIBUCIÓN POR RANGOS DE SCORE:")
    bins = [0, 50, 70, 85, 100]
    labels = ['Bajo (0-50)', 'Medio (50-70)', 'Alto (70-85)', 'Muy Alto (85-100)']
    sitios_seleccionados['rango_score'] = pd.cut(
        sitios_seleccionados['survival_score'], 
        bins=bins, 
        labels=labels
    )
    print(sitios_seleccionados['rango_score'].value_counts().sort_index())


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def cargar_datos_existentes():
    """Intenta cargar datos del Módulo 1 previamente generados"""
    
    import glob
    import geopandas as gpd
    import pandas as pd
    
    print("\n🔍 Buscando datos previamente generados...")
    
    # Buscar GeoJSON
    geojson_files = glob.glob('datos_indices_puros/indices_puros_*.geojson')
    
    if geojson_files:
        latest_file = max(geojson_files, key=os.path.getctime)
        print(f"   ✅ Encontrado: {latest_file}")
        gdf = gpd.read_file(latest_file)
        print(f"   • Puntos cargados: {len(gdf)}")
        return gdf
    
    # Buscar CSV
    csv_files = glob.glob('datos_indices_puros/indices_puros_*.csv')
    
    if csv_files:
        latest_file = max(csv_files, key=os.path.getctime)
        print(f"   ✅ Encontrado: {latest_file}")
        df = pd.read_csv(latest_file)
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.longitude, df.latitude),
            crs='EPSG:4326'
        )
        print(f"   • Puntos cargados: {len(gdf)}")
        return gdf
    
    print("   ❌ No se encontraron datos previos")
    print("\n💡 Debes ejecutar primero el Módulo 1 para obtener datos satelitales")
    sys.exit(1)


def verificar_dependencias():
    """Verifica que todas las dependencias estén instaladas"""
    
    dependencias = {
        'ee': 'earthengine-api',
        'geopandas': 'geopandas',
        'sklearn': 'scikit-learn',
        'shapely': 'shapely',
        'pandas': 'pandas',
        'numpy': 'numpy'
    }
    
    faltantes = []
    
    for modulo, paquete in dependencias.items():
        try:
            __import__(modulo)
        except ImportError:
            faltantes.append(paquete)
    
    if faltantes:
        print("\n❌ Faltan dependencias:")
        for pkg in faltantes:
            print(f"   • {pkg}")
        print("\n💡 Instalar con:")
        print(f"   pip install {' '.join(faltantes)}")
        sys.exit(1)


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Ejecuta el pipeline completo de reforestación"""
    
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║          🌳 SISTEMA DE EVALUACIÓN ECOFISIOLÓGICA PARA REFORESTACIÓN 🌳    ║
║                                                                           ║
║  Pipeline completo: Extracción satelital → Selección con IA → GPS        ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Verificar dependencias
    print("🔍 Verificando dependencias...")
    verificar_dependencias()
    print("   ✅ Todas las dependencias disponibles\n")
    
    # Cargar configuración
    config = ConfiguracionReforestacion()
    
    print("📋 CONFIGURACIÓN DEL PROYECTO:")
    print(f"   • Área: {config.COORDS[0]} (primer vértice)")
    print(f"   • Período: {config.FECHA_INICIO} a {config.FECHA_FIN}")
    print(f"   • Grid: {config.GRID_SIZE_M}m")
    print(f"   • Plántulas: {config.N_PLANTULAS}")
    print(f"   • Estrategia: {config.ESTRATEGIA}")
    
    # Confirmar ejecución
    print("\n" + "="*80)
    respuesta = input("¿Deseas continuar con esta configuración? (s/n): ")
    
    if respuesta.lower() not in ['s', 'si', 'sí', 'y', 'yes']:
        print("\n❌ Ejecución cancelada por el usuario")
        sys.exit(0)
    
    # Timestamp de inicio
    inicio = datetime.now()
    print(f"\n⏱️  Inicio: {inicio.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # PASO 1: Extracción de índices
        gdf_indices = ejecutar_modulo1(config)
        
        # PASO 2: Selección de sitios
        sitios_finales = ejecutar_modulo3(gdf_indices, config)
        
        # PASO 3: Reportes finales
        generar_reporte_final(sitios_finales, config)
        
        # Tiempo total
        fin = datetime.now()
        duracion = fin - inicio
        
        print("\n" + "="*80)
        print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
        print("="*80)
        print(f"\n⏱️  Duración total: {duracion}")
        print(f"📁 Archivos generados:")
        print(f"   • datos_indices_puros/*.geojson")
        print(f"   • datos_indices_puros/*.csv")
        print(f"   • sitios_plantacion/*.geojson ← USAR PARA GPS")
        print(f"   • sitios_plantacion/*.csv ← LISTADO DE CAMPO")
        print(f"   • sitios_plantacion/*.txt ← RESUMEN EJECUTIVO")
        
        print(f"\n🎯 PRÓXIMOS PASOS:")
        print(f"   1. Abrir 'sitios_plantacion/*.geojson' en tu app GPS")
        print(f"   2. Importar waypoints usando plantula_id")
        print(f"   3. Navegar a cada coordenada para plantación")
        print(f"   4. Priorizar sitios con is_exceptional = True")
        
        return sitios_finales
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Ejecución interrumpida por el usuario")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n\n❌ ERROR CRÍTICO: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    # Añadir pandas al inicio
    import pandas as pd
    
    # Ejecutar pipeline completo
    sitios = main()
    
    # Opcional: Guardar objeto para análisis interactivo
    print(f"\n💡 Los sitios seleccionados están en la variable 'sitios'")
    print(f"   Puedes analizarlos con: sitios.head(), sitios.describe(), etc.")