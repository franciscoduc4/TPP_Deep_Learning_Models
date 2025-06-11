from collections import defaultdict
import re
from typing import Dict, List, Optional, Union
import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
import os
from joblib import Parallel, delayed
import time
from datetime import timedelta, datetime
from tqdm import tqdm
import matplotlib

from custom.printer import print_debug, print_error, print_info, print_warning
matplotlib.use('Agg')
import xml.etree.ElementTree as ET
import glob
from zoneinfo import ZoneInfo

# Constantes para evitar repetición de strings
PROJECT_ROOT: str = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(PROJECT_ROOT)
DATA_PATH_SUBJECTS: str = os.path.join(os.getcwd(), "data", "subjects")
OHIO_DATA_DIRS: list[str] = [
    os.path.join(os.getcwd(), "data", "OhioT1DM","2018","train"), 
    os.path.join(os.getcwd(), "data", "OhioT1DM","2018","test"), 
    os.path.join(os.getcwd(), "data", "OhioT1DM","2020","train"), 
    os.path.join(os.getcwd(), "data", "OhioT1DM","2020","test")
]
OUTPUT_DIR: str = 'new_ohio/processed_data'
PLOTS_DIR: str = 'new_ohio/processed_data/plots'
MGDL_UNIT: str = "mg/dl"

from constants.constants import CONST_DEFAULT_SEED, DATE_FORMAT, TIMESTAMP_COL, SUBJECT_ID_COL, GLUCOSE_COL, BOLUS_COL, MEAL_COL, BASAL_COL, TEMP_BASAL_COL
from config.params import CONFIG_PROCESSING, USE_EXCEL_DATA

def _verify_excel_file(subject_path: str) -> bool:
    """
    Verifica si un archivo Excel existe, es accesible y no está vacío.
    
    Parámetros:
    -----------
    subject_path : str
        Ruta al archivo Excel del sujeto.
        
    Retorna:
    --------
    bool
        True si el archivo es válido, False en caso contrario.
    """
    # Verificar si el archivo existe y es accesible
    if not os.path.exists(subject_path) or not os.access(subject_path, os.R_OK):
        print_error(f"Archivo no existe o no es accesible: {subject_path}")
        return False
        
    # Verificar el tamaño del archivo
    file_size = os.path.getsize(subject_path)
    if file_size == 0:
        print_error(f"Archivo vacío: {subject_path}")
        return False
    
    return True

def _load_cgm_sheet(subject_path: str) -> Optional[pl.DataFrame]:
    """
    Carga la hoja CGM de un archivo Excel.
    
    Parámetros:
    -----------
    subject_path : str
        Ruta al archivo Excel del sujeto.
        
    Retorna:
    --------
    Optional[pl.DataFrame]
        DataFrame con datos CGM o None si hubo error.
    """
    try:
        cgm_df = pl.read_excel(subject_path, sheet_name="CGM")
        if cgm_df.is_empty():
            print_warning(f"Hoja CGM vacía en {os.path.basename(subject_path)}")
            return None
        
        # Verificar columnas requeridas
        if "date" not in cgm_df.columns or MGDL_UNIT not in cgm_df.columns:
            missing_cols = []
            if "date" not in cgm_df.columns: missing_cols.append("date")
            if MGDL_UNIT not in cgm_df.columns: missing_cols.append(MGDL_UNIT)
            print_warning(f"Faltan columnas en hoja CGM de {os.path.basename(subject_path)}: {missing_cols}")
            print_warning(f"Columnas disponibles: {cgm_df.columns}")
            return None
        
        # Procesamiento de CGM
        cgm_df = cgm_df.with_columns(
            pl.col("date").cast(pl.Datetime(time_unit="us")).alias(TIMESTAMP_COL)
        )
        return cgm_df.sort(TIMESTAMP_COL).rename({MGDL_UNIT: GLUCOSE_COL})
    except Exception as e:
        print_error(f"Error cargando hoja CGM de {os.path.basename(subject_path)}: {e}")
        return None

def _load_bolus_sheet(subject_path: str) -> Optional[pl.DataFrame]:
    """
    Carga la hoja Bolus de un archivo Excel.
    
    Parámetros:
    -----------
    subject_path : str
        Ruta al archivo Excel del sujeto.
        
    Retorna:
    --------
    Optional[pl.DataFrame]
        DataFrame con datos Bolus o None si hubo error.
    """
    try:
        bolus_df = pl.read_excel(subject_path, sheet_name="Bolus")
        if bolus_df.is_empty():
            print_warning(f"Hoja Bolus vacía en {os.path.basename(subject_path)}")
            return None
        
        # Verificar columnas requeridas
        if "date" not in bolus_df.columns:
            print_warning(f"Falta columna 'date' en hoja Bolus de {os.path.basename(subject_path)}")
            print_warning(f"Columnas disponibles: {bolus_df.columns}")
            return None
        
        # Procesamiento de Bolus
        return bolus_df.with_columns(
            pl.col("date").cast(pl.Datetime(time_unit="us")).alias(TIMESTAMP_COL)
        )
    except Exception as e:
        print_error(f"Error cargando hoja Bolus de {os.path.basename(subject_path)}: {e}")
        return None

def _load_basal_sheet(subject_path: str) -> Optional[pl.DataFrame]:
    """
    Carga la hoja Basal de un archivo Excel.
    
    Parámetros:
    -----------
    subject_path : str
        Ruta al archivo Excel del sujeto.
        
    Retorna:
    --------
    Optional[pl.DataFrame]
        DataFrame con datos Basal o None si hubo error.
    """
    try:
        basal_df = pl.read_excel(subject_path, sheet_name="Basal")
        if not basal_df.is_empty() and "date" in basal_df.columns:
            return basal_df.with_columns(
                pl.col("date").cast(pl.Datetime(time_unit="us")).alias(TIMESTAMP_COL)
            )
        return None
    except Exception:
        return None

def load_excel_data(subject_path: str) -> tuple[Optional[pl.DataFrame], Optional[pl.DataFrame], Optional[pl.DataFrame]]:
    """
    Carga datos de un sujeto desde un archivo Excel con hojas CGM, Bolus y Basal.

    Parámetros:
    -----------
    subject_path : str
        Ruta al archivo Excel del sujeto.

    Retorna:
    --------
    tuple[Optional[pl.DataFrame], Optional[pl.DataFrame], Optional[pl.DataFrame]]
        Tupla con (cgm_df, bolus_df, basal_df), donde cada elemento es un DataFrame
        o None si hubo error en la carga.
    """
    try:
        # Verificar archivo
        if not _verify_excel_file(subject_path):
            return None, None, None
        
        # Cargar hojas
        cgm_df = _load_cgm_sheet(subject_path)
        bolus_df = _load_bolus_sheet(subject_path)
        basal_df = _load_basal_sheet(subject_path)
        
        # Verificar si se pudo cargar al menos una de las hojas principales
        if cgm_df is None and bolus_df is None:
            print_error(f"No se pudo cargar ninguna hoja útil de {os.path.basename(subject_path)}")
            return None, None, None
        
        return cgm_df, bolus_df, basal_df
    
    except Exception as e:
        print_error(f"Error general al cargar {os.path.basename(subject_path)}: {e}")
        return None, None, None

def _get_expected_subjects_by_year(data_dir: str) -> tuple[str, list[str]]:
    """
    Determina el año y obtiene la lista de sujetos esperados.
    
    Parámetros:
    -----------
    data_dir : str
        Directorio con archivos XML.
        
    Retorna:
    --------
    tuple[str, list[str]]
        Tupla con (año, lista_de_sujetos_esperados)
    """
    expected_subjects = {
        '2018': ['559-ws-training', '563-ws-training', '570-ws-training', '575-ws-training', '588-ws-training', '591-ws-training'],
        '2020': ['540-ws-training', '544-ws-training', '552-ws-training', '567-ws-training', '584-ws-training', '596-ws-training']
    }
    
    if '2018' in data_dir:
        year = '2018'
    elif '2020' in data_dir:
        year = '2020'
    else:
        year = None
    if year is None:
        raise ValueError(f"No se pudo determinar el año del directorio: {data_dir}")
    
    suffix = '-ws-training' if 'train' in data_dir else '-ws-testing'
    expected_subjects[year] = [s.replace('-ws-training', suffix).replace('-ws-testing', suffix) for s in expected_subjects[year]]
    
    return year, expected_subjects[year]

def _validate_xml_files(data_dir: str, expected_subjects: list[str], year: str) -> list[str]:
    """
    Valida la existencia de archivos XML y reporta sujetos faltantes.
    
    Parámetros:
    -----------
    data_dir : str
        Directorio con archivos XML.
    expected_subjects : list[str]
        Lista de sujetos esperados.
    year : str
        Año de los datos.
        
    Retorna:
    --------
    list[str]
        Lista de archivos XML encontrados.
    """
    xml_files = glob.glob(os.path.join(data_dir, "*.xml"))
    found_subjects = [os.path.basename(f).split('.')[0] for f in xml_files]
    missing_subjects = [s for s in expected_subjects if s not in found_subjects]
    
    if missing_subjects:
        print_error(f"Faltan datos para sujetos del año {year}: {missing_subjects}")
        if not found_subjects:
            raise ValueError(f"No se encontraron archivos XML en {data_dir}")
    
    return xml_files

def _process_xml_records(data_type_elem, subject_id: str, year: str) -> list[dict]:
    """
    Procesa los registros de un tipo de dato específico desde un elemento XML.
    
    Parámetros:
    -----------
    data_type_elem : ET.Element
        Elemento XML que contiene los registros.
    subject_id : str
        ID del sujeto.
    year : str
        Año de los datos.
        
    Retorna:
    --------
    list[dict]
        Lista de registros procesados.
    """
    records = []
    for event in data_type_elem:
        record_dict = dict(event.attrib)
        record_dict[SUBJECT_ID_COL] = extract_numeric_id(subject_id)
        record_dict['Year'] = year
        records.append(record_dict)
    return records

def _process_single_xml_file(xml_file: str, year: str, expected_types: list[str]) -> tuple[dict[str, pl.DataFrame], dict[str, dict[str, int]]]:
    """
    Procesa un único archivo XML y extrae todos los tipos de datos.
    
    Parámetros:
    -----------
    xml_file : str
        Ruta al archivo XML.
    year : str
        Año de los datos.
    expected_types : list[str]
        Lista de tipos de datos esperados.
        
    Retorna:
    --------
    tuple[dict[str, pl.DataFrame], dict[str, dict[str, int]]]
        Tupla con (diccionario_de_dataframes, estadísticas_del_sujeto)
    """
    subject_id = os.path.basename(xml_file).split('.')[0]
    print_info(f"Procesando SubjectID: {subject_id} (Año {year})")
    
    data_dict = {}
    subject_stats = defaultdict(int)
    
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        for data_type_elem in root:
            data_type = data_type_elem.tag
            if data_type == 'patient' or data_type not in expected_types:
                continue
            
            records = _process_xml_records(data_type_elem, subject_id, year)
            
            if records:
                df = pl.DataFrame(records)
                if 'value' in df.columns:
                    df = df.with_columns(pl.col('value').cast(pl.Float64))
                data_dict[data_type] = df
                print_info(f"SubjectID {subject_id}: {data_type}={len(records)} registros")
                subject_stats[data_type] = len(records)
                
    except Exception as e:
        print_error(f"Error procesando {xml_file}: {e}")
    
    return data_dict, {subject_id: subject_stats}

def _consolidate_data_dicts(all_data_dicts: list[dict[str, pl.DataFrame]]) -> dict[str, pl.DataFrame]:
    """
    Consolida múltiples diccionarios de DataFrames en uno solo.
    
    Parámetros:
    -----------
    all_data_dicts : list[dict[str, pl.DataFrame]]
        Lista de diccionarios de DataFrames.
        
    Retorna:
    --------
    dict[str, pl.DataFrame]
        Diccionario consolidado con todos los DataFrames concatenados.
    """
    consolidated = {}
    
    for data_dict in all_data_dicts:
        for data_type, df in data_dict.items():
            if data_type in consolidated:
                consolidated[data_type] = pl.concat([consolidated[data_type], df])
            else:
                consolidated[data_type] = df
    
    return consolidated

def _print_final_statistics(subject_stats: dict, expected_subjects: list[str], expected_types: list[str], year: str):
    """
    Imprime estadísticas finales del procesamiento.
    
    Parámetros:
    -----------
    subject_stats : dict
        Estadísticas por sujeto.
    expected_subjects : list[str]
        Lista de sujetos esperados.
    expected_types : list[str]
        Lista de tipos de datos esperados.
    year : str
        Año de los datos.
    """
    print_info(f"\nEstadísticas por sujeto (Año {year}):")
    for subject_id in sorted(subject_stats.keys()):
        stats = subject_stats[subject_id]
        stat_str = ", ".join([f"{k}={v}" for k, v in stats.items()])
        print_info(f"SubjectID {subject_id}: {stat_str}")

def load_xml_data(data_dir: str) -> dict[str, pl.DataFrame]:
    """
    Carga datos desde archivos XML en el directorio especificado, con validación de sujetos y soporte para múltiples tipos de datos.

    Parámetros:
    -----------
    data_dir : str
        Directorio con archivos XML.

    Retorna:
    --------
    dict[str, pl.DataFrame]
        Diccionario con DataFrames por tipo de dato (glucose_level, bolus, meal, basal, temp_basal, etc.).
    """
    print_info(f"Cargando datos desde {data_dir}")
    
    # Determinar año y sujetos esperados
    year, expected_subjects = _get_expected_subjects_by_year(data_dir)
    
    # Validar archivos XML
    xml_files = _validate_xml_files(data_dir, expected_subjects, year)
    
    # Definir tipos de datos esperados
    expected_types = [
        'glucose_level', 'bolus', 'meal', 'basal', 'temp_basal', 'exercise', 'basis_steps', 'hypo_event',
        'finger_stick', 'sleep', 'work', 'stressors', 'illness', 'basis_heart_rate', 'basis_gsr',
        'basis_skin_temperature', 'basis_air_temperature', 'basis_sleep', 'acceleration'
    ]
    
    # Procesar todos los archivos XML
    all_data_dicts = []
    all_subject_stats = {}
    
    for xml_file in xml_files:
        file_data_dict, file_subject_stats = _process_single_xml_file(xml_file, year, expected_types)
        all_data_dicts.append(file_data_dict)
        all_subject_stats.update(file_subject_stats)
    
    # Consolidar todos los DataFrames
    data_dict = _consolidate_data_dicts(all_data_dicts)
    
    # Imprimir estadísticas finales
    _print_final_statistics(all_subject_stats, expected_subjects, expected_types, year)
    
    # Verificar tipos de datos faltantes
    missing_types = [t for t in expected_types if t not in data_dict]
    if missing_types:
        print_warning(f"Faltan tipos de datos: {missing_types}")
    
    # Verificar número de sujetos procesados
    if len(all_subject_stats) != len(expected_subjects):
        print_error(f"Se encontraron datos para {len(all_subject_stats)}/{len(expected_subjects)} sujetos")
    
    return data_dict

def _process_bolus_data(data: dict[str, pl.DataFrame], processed: dict[str, pl.DataFrame]) -> None:
    """Procesa datos de bolus."""
    if "bolus" not in data:
        return
    
    bolus = data["bolus"].clone()
    if "dose" in bolus.columns:
        bolus = bolus.rename({"dose": BOLUS_COL})
        bolus = bolus.with_columns(pl.col(BOLUS_COL).cast(pl.Float64))
    if "ts_begin" in bolus.columns:
        bolus = bolus.with_columns(
            pl.col("ts_begin")
            .str.strptime(pl.Datetime(time_unit="us"), DATE_FORMAT)
            .alias(TIMESTAMP_COL)
        )
    valid_bolus = bolus.filter(pl.col(BOLUS_COL).is_not_null() & (pl.col(BOLUS_COL) > 0))
    processed["bolus"] = valid_bolus
    print_info(f"Eventos bolus válidos: {valid_bolus.height}")

def _process_meal_data(data: dict[str, pl.DataFrame], processed: dict[str, pl.DataFrame]) -> None:
    """Procesa datos de comida."""
    if "meal" not in data:
        return
    
    meal = data["meal"].clone()
    if "carbs" in meal.columns:
        meal = meal.rename({"carbs": MEAL_COL})
        meal = meal.with_columns(pl.col(MEAL_COL).cast(pl.Float64))
    if "ts" in meal.columns:
        meal = meal.with_columns(
            pl.col("ts")
            .str.strptime(pl.Datetime(time_unit="us"), DATE_FORMAT)
            .alias(TIMESTAMP_COL)
        )
    valid_meal = meal.filter(pl.col(MEAL_COL).is_not_null() & (pl.col(MEAL_COL) > 0))
    processed["meal"] = valid_meal
    print_info(f"Eventos meal válidos: {valid_meal.height}")

def _process_basal_data(data: dict[str, pl.DataFrame], processed: dict[str, pl.DataFrame]) -> None:
    """Procesa datos de insulina basal."""
    if "basal" not in data:
        return
    
    basal = data["basal"].clone()
    basal = basal.rename({"value": BASAL_COL}).with_columns(
        pl.col(BASAL_COL).cast(pl.Float64),
        pl.col("ts").str.strptime(pl.Datetime(time_unit="us"), DATE_FORMAT).alias(TIMESTAMP_COL)
    )
    processed["basal"] = basal.filter(pl.col(BASAL_COL).is_not_null())
    print_info(f"Eventos basal válidos: {processed['basal'].height}")

def _process_temp_basal_data(data: dict[str, pl.DataFrame], processed: dict[str, pl.DataFrame]) -> None:
    """Procesa datos de insulina basal temporal."""
    if "temp_basal" not in data:
        return
    
    temp_basal = data["temp_basal"].clone()
    temp_basal = temp_basal.rename({"value": TEMP_BASAL_COL}).with_columns(
        pl.col(TEMP_BASAL_COL).cast(pl.Float64),
        pl.col("ts_begin").str.strptime(pl.Datetime(time_unit="us"), DATE_FORMAT).alias(TIMESTAMP_COL)
    )
    processed["temp_basal"] = temp_basal.filter(pl.col(TEMP_BASAL_COL).is_not_null())
    print_info(f"Eventos temp_basal válidos: {processed['temp_basal'].height}")

def _process_exercise_data(data: dict[str, pl.DataFrame], processed: dict[str, pl.DataFrame]) -> None:
    """Procesa datos de ejercicio."""
    if "exercise" not in data:
        return
    
    exercise = data["exercise"].clone()
    exercise = exercise.with_columns(
        pl.col("intensity").cast(pl.Float64),
        pl.col("duration").cast(pl.Float64),
        pl.col("ts").str.strptime(pl.Datetime(time_unit="us"), DATE_FORMAT).alias(TIMESTAMP_COL)
    )
    processed["exercise"] = exercise.filter(pl.col("intensity").is_not_null())
    print_info(f"Eventos exercise válidos: {processed['exercise'].height}")

def _process_steps_data(data: dict[str, pl.DataFrame], processed: dict[str, pl.DataFrame]) -> None:
    """Procesa datos de pasos."""
    if "basis_steps" not in data:
        return
    
    steps = data["basis_steps"].clone()
    steps = steps.rename({"value": "steps"}).with_columns(
        pl.col("steps").cast(pl.Float64),
        pl.col("ts").str.strptime(pl.Datetime(time_unit="us"), DATE_FORMAT).alias(TIMESTAMP_COL)
    )
    processed["basis_steps"] = steps.filter(pl.col("steps").is_not_null())
    print_info(f"Eventos steps válidos: {processed['basis_steps'].height}")

def _process_event_data(data: dict[str, pl.DataFrame], processed: dict[str, pl.DataFrame]) -> None:
    """Procesa datos de eventos simples (hypo_event, stressors, illness)."""
    simple_events = ["hypo_event", "stressors", "illness"]
    
    for event_type in simple_events:
        if event_type not in data:
            continue
        
        event_df = data[event_type].clone()
        if "ts" in event_df.columns:
            event_df = event_df.with_columns(
                pl.col("ts").str.strptime(pl.Datetime(time_unit="us"), DATE_FORMAT).alias(TIMESTAMP_COL)
            )
        processed[event_type] = event_df
        print_info(f"Eventos {event_type} válidos: {processed[event_type].height}")

def _process_finger_stick_data(data: dict[str, pl.DataFrame], processed: dict[str, pl.DataFrame]) -> None:
    """Procesa datos de medición digital de glucosa."""
    if "finger_stick" not in data:
        return
    
    finger_stick = data["finger_stick"].clone()
    finger_stick = finger_stick.rename({"value": "finger_stick_bg"}).with_columns(
        pl.col("finger_stick_bg").cast(pl.Float64),
        pl.col("ts").str.strptime(pl.Datetime(time_unit="us"), DATE_FORMAT).alias(TIMESTAMP_COL)
    )
    processed["finger_stick"] = finger_stick.filter(pl.col("finger_stick_bg").is_not_null())
    print_info(f"Eventos finger_stick válidos: {processed['finger_stick'].height}")

def _process_sleep_work_data(data: dict[str, pl.DataFrame], processed: dict[str, pl.DataFrame]) -> None:
    """Procesa datos de sueño y trabajo que pueden tener timestamps begin/end."""
    data_types = ["sleep", "work", "basis_sleep"]
    
    for data_type in data_types:
        if data_type not in data:
            continue
        
        df = data[data_type].clone()
        
        # Determinar columna de calidad/intensidad
        quality_col = "quality" if data_type in ["sleep", "basis_sleep"] else "intensity"
        
        if "ts_begin" in df.columns and "ts_end" in df.columns:
            # Para datos con begin/end, usar ts_begin como Timestamp principal
            df = df.with_columns([
                pl.col(quality_col).cast(pl.Float64),
                pl.col("ts_begin").str.strptime(pl.Datetime(time_unit="us"), DATE_FORMAT).alias(TIMESTAMP_COL),
                pl.col("ts_end").str.strptime(pl.Datetime(time_unit="us"), DATE_FORMAT).alias("Timestamp_end")
            ])
        elif "ts" in df.columns:
            df = df.with_columns([
                pl.col(quality_col).cast(pl.Float64),
                pl.col("ts").str.strptime(pl.Datetime(time_unit="us"), DATE_FORMAT).alias(TIMESTAMP_COL)
            ])
        
        processed[data_type] = df.filter(pl.col(quality_col).is_not_null())
        print_info(f"Eventos {data_type} válidos: {processed[data_type].height}")

def _process_sensor_data(data: dict[str, pl.DataFrame], processed: dict[str, pl.DataFrame]) -> None:
    """Procesa datos de sensores (heart rate, GSR, temperatura, aceleración)."""
    sensor_mappings = {
        "basis_heart_rate": "heart_rate",
        "basis_gsr": "gsr",
        "basis_skin_temperature": "skin_temperature",
        "basis_air_temperature": "air_temperature",
        "acceleration": "acceleration"
    }
    
    for sensor_type, new_col_name in sensor_mappings.items():
        if sensor_type not in data:
            continue
        
        sensor_df = data[sensor_type].clone()
        sensor_df = sensor_df.rename({"value": new_col_name}).with_columns(
            pl.col(new_col_name).cast(pl.Float64),
            pl.col("ts").str.strptime(pl.Datetime(time_unit="us"), DATE_FORMAT).alias(TIMESTAMP_COL)
        )
        processed[sensor_type] = sensor_df.filter(pl.col(new_col_name).is_not_null())
        print_info(f"Eventos {sensor_type} válidos: {processed[sensor_type].height}")

def preprocess_xml_bolus_meal(data: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """
    Preprocesa los datos de bolus, meal, basal, temp_basal, exercise, basis_steps, hypo_event, etc., renombrando columnas y convirtiendo timestamps.

    Parámetros:
    -----------
    data : dict[str, pl.DataFrame]
        Diccionario con DataFrames de datos XML.

    Retorna:
    --------
    dict[str, pl.DataFrame]
        Diccionario con DataFrames preprocesados.
    """
    processed: dict[str, pl.DataFrame] = {}
    
    # Procesar cada tipo de datos usando funciones especializadas
    _process_bolus_data(data, processed)
    _process_meal_data(data, processed)
    _process_basal_data(data, processed)
    _process_temp_basal_data(data, processed)
    _process_exercise_data(data, processed)
    _process_steps_data(data, processed)
    _process_event_data(data, processed)
    _process_finger_stick_data(data, processed)
    _process_sleep_work_data(data, processed)
    _process_sensor_data(data, processed)
    
    return processed

def align_events_to_cgm(cgm_df: pl.DataFrame, event_df: pl.DataFrame, event_time_col: str = TIMESTAMP_COL, tolerance_minutes: int = 5) -> pl.DataFrame:
    """
    Alinea eventos (bolus, meal, etc.) con el timestamp de CGM más cercano dentro de una tolerancia, con mejor print_

    Parámetros:
    -----------
    cgm_df : pl.DataFrame
        DataFrame con datos CGM.
    event_df : pl.DataFrame
        DataFrame con eventos a alinear.
    event_time_col : str, opcional
        Columna de timestamp en event_df (default: "Timestamp").
    tolerance_minutes : int, opcional
        Tolerancia en minutos para la alineación (default: 5).

    Retorna:
    --------
    pl.DataFrame
        DataFrame de eventos con timestamps alineados.
    """
    if cgm_df.is_empty() or event_df.is_empty():
        return event_df

    # Asegurar que las columnas de tiempo son datetime
    cgm_df = ensure_timestamp_datetime(cgm_df)
    event_df = ensure_timestamp_datetime(event_df, event_time_col)
    
    # Convertir tolerancia a timedelta
    tolerance = timedelta(minutes=tolerance_minutes)
    
    # Obtener timestamps de CGM
    cgm_times = cgm_df[TIMESTAMP_COL].to_list()
    
    # Función para encontrar el timestamp CGM más cercano
    def find_nearest_cgm(event_time: datetime) -> Optional[datetime]:
        if not cgm_times:
            return None
            
        # Encontrar el timestamp CGM más cercano
        nearest_time = min(cgm_times, key=lambda x: abs(x - event_time))
        time_diff = abs(nearest_time - event_time)
        
        # Verificar si está dentro de la tolerancia
        if time_diff <= tolerance:
            return nearest_time
        return None
    
    # Aplicar la función a cada evento
    aligned_times = []
    for event_time in event_df[event_time_col]:
        nearest_cgm = find_nearest_cgm(event_time)
        if nearest_cgm is not None:
            aligned_times.append(nearest_cgm)
        else:
            aligned_times.append(event_time)
    
    # Crear nuevo DataFrame con tiempos alineados
    aligned_df = event_df.with_columns(
        pl.Series(name=TIMESTAMP_COL, values=aligned_times)
    )
    
    # Filtrar eventos que no se pudieron alinear
    aligned_df = aligned_df.filter(
        pl.col(TIMESTAMP_COL).is_in(cgm_times)
    )
    
    # Calcular estadísticas de alineación
    total_events = len(event_df)
    aligned_events = len(aligned_df)
    lost_events = total_events - aligned_events
    
    if lost_events > 0:
        print_warning(f"Eventos descartados por estar fuera de tolerancia: {lost_events}")
        print_warning(f"Eventos sin CGM correspondiente: {lost_events}")
    
    print_info(f"Eventos alineados: {aligned_events}/{total_events} ({aligned_events/total_events*100:.1f}%)")
    
    return aligned_df

def preprocess_cgm(cgm: pl.DataFrame) -> pl.DataFrame:
    """
    Preprocesa los datos de CGM, convirtiendo la columna de timestamp.

    Parámetros:
    -----------
    cgm : pl.DataFrame
        DataFrame con datos CGM.

    Retorna:
    --------
    pl.DataFrame
        DataFrame con la columna de timestamp convertida.
    """
    if "ts" in cgm.columns:
        cgm = cgm.with_columns(
            pl.col("ts")
            .str.strptime(pl.Datetime(time_unit="us"), DATE_FORMAT)
            .alias(TIMESTAMP_COL)
        )
    return cgm

def _join_contextual_data(joined_df: pl.DataFrame, processed_data: Dict[str, pl.DataFrame]) -> pl.DataFrame:
    """
    Une datos contextuales (sleep, work, exercise) al DataFrame principal.

    Parámetros:
    -----------
    joined_df : pl.DataFrame
        DataFrame principal con datos ya unidos.
    processed_data : Dict[str, pl.DataFrame]
        Diccionario con DataFrames procesados.

    Retorna:
    --------
    pl.DataFrame
        DataFrame con datos contextuales unidos.
    """
    # Definir mapeo de datos contextuales y sus columnas de interés
    contextual_mappings = {
        'sleep': {
            'value_cols': ['quality'],
            'rename_map': {'quality': 'sleep_quality'},
            'description': 'datos de sueño'
        },
        'work': {
            'value_cols': ['intensity'],
            'rename_map': {'intensity': 'work_intensity'},
            'description': 'datos de trabajo'
        },
        'exercise': {
            'value_cols': ['intensity', 'duration'],
            'rename_map': {'intensity': 'exercise_intensity', 'duration': 'exercise_duration'},
            'description': 'datos de ejercicio'
        }
    }
    
    for data_type, config in contextual_mappings.items():
        if data_type in processed_data and not processed_data[data_type].is_empty():
            contextual_df = processed_data[data_type].clone()
            
            # Estandarizar el DataFrame contextual
            try:
                contextual_df = _standardize_contextual_dataframe(contextual_df, data_type)
            except Exception as e:
                print_warning(f"Error estandarizando DataFrame {data_type}: {e}")
                continue
            
            # Verificar que las columnas de valor existen
            available_cols = [col for col in config['value_cols'] if col in contextual_df.columns]
            if not available_cols:
                print_warning(f"No se encontraron columnas de valor para {data_type}: {config['value_cols']}")
                continue
            
            # Preparar DataFrame para join
            contextual_df_for_join = _prepare_contextual_df_for_join(
                contextual_df, available_cols, config['rename_map']
            )
            
            # Realizar join asof
            joined_df = _perform_contextual_join_asof(
                joined_df, 
                contextual_df_for_join, 
                data_type.capitalize()
            )
            
            print_info(f"Unión exitosa con {config['description']}: {len(available_cols)} columnas añadidas")
    
    return joined_df

def _standardize_contextual_dataframe(contextual_df: pl.DataFrame, data_type: str) -> pl.DataFrame:
    """
    Estandariza un DataFrame contextual para prepararlo para join.

    Parámetros:
    -----------
    contextual_df : pl.DataFrame
        DataFrame contextual original.
    data_type : str
        Tipo de datos contextuales.

    Retorna:
    --------
    pl.DataFrame
        DataFrame contextual estandarizado.
    """
    # Verificar y estandarizar columna de timestamp
    timestamp_candidates = ['Timestamp', 'ts', 'ts_begin', 'Time']
    timestamp_col = None
    
    for candidate in timestamp_candidates:
        if candidate in contextual_df.columns:
            timestamp_col = candidate
            break
    
    if timestamp_col is None:
        raise ValueError(f"No se encontró columna de timestamp en DataFrame {data_type}")
    
    # Renombrar a Timestamp si es necesario
    if timestamp_col != 'Timestamp':
        contextual_df = contextual_df.rename({timestamp_col: 'Timestamp'})
    
    # Asegurar que Timestamp es datetime
    if contextual_df['Timestamp'].dtype not in [pl.Datetime, pl.Datetime(time_unit="us"), pl.Datetime(time_unit="ms")]:
        try:
            contextual_df = contextual_df.with_columns(
                pl.col('Timestamp').cast(pl.Datetime(time_unit="us"))
            )
        except Exception as e:
            print_error(f"No se pudo convertir Timestamp a datetime en {data_type}: {e}")
            raise
    
    # Verificar que SubjectID existe
    if 'SubjectID' not in contextual_df.columns:
        raise ValueError(f"Columna 'SubjectID' no encontrada en DataFrame {data_type}")
    
    return contextual_df.sort(['SubjectID', 'Timestamp'])


def _prepare_contextual_df_for_join(
    contextual_df: pl.DataFrame, 
    value_cols: List[str], 
    rename_map: Dict[str, str]
) -> pl.DataFrame:
    """
    Prepara un DataFrame contextual para join renombrando columnas apropiadamente.

    Parámetros:
    -----------
    contextual_df : pl.DataFrame
        DataFrame contextual original.
    value_cols : List[str]
        Lista de columnas de valor disponibles.
    rename_map : Dict[str, str]
        Mapeo de nombres de columnas originales a nuevos nombres.

    Retorna:
    --------
    pl.DataFrame
        DataFrame preparado para join.
    """
    # Seleccionar columnas necesarias para el join
    required_cols = ['Timestamp', 'SubjectID'] + value_cols
    available_cols = [col for col in required_cols if col in contextual_df.columns]
    
    prepared_df = contextual_df.select(available_cols)
    
    # Renombrar columnas según el mapeo
    rename_expressions = []
    for old_name, new_name in rename_map.items():
        if old_name in prepared_df.columns:
            rename_expressions.append(pl.col(old_name).alias(new_name))
    
    # Mantener columnas que no necesitan renombre
    keep_cols = ['Timestamp', 'SubjectID']
    for col in keep_cols:
        if col in prepared_df.columns:
            rename_expressions.append(pl.col(col))
    
    if rename_expressions:
        prepared_df = prepared_df.select(rename_expressions)
    
    return prepared_df


def _perform_contextual_join_asof(
    base_df: pl.DataFrame,
    contextual_df: pl.DataFrame,
    data_type_name: str
) -> pl.DataFrame:
    """
    Realiza un join_asof seguro con datos contextuales.

    Parámetros:
    -----------
    base_df : pl.DataFrame
        DataFrame base para el join.
    contextual_df : pl.DataFrame
        DataFrame contextual a unir.
    data_type_name : str
        Nombre descriptivo del tipo de datos.

    Retorna:
    --------
    pl.DataFrame
        DataFrame resultante del join.
    """
    if contextual_df.is_empty():
        print_warning(f"DataFrame {data_type_name} está vacío, saltando join")
        return base_df
    
    try:
        # Realizar join_asof
        joined_df = base_df.join_asof(
            contextual_df,
            on='Timestamp',
            by='SubjectID',
            tolerance='15m'  # Tolerancia mayor para datos contextuales
        )
        
        print_debug(f"Join exitoso con {data_type_name}: {joined_df.shape}")
        return joined_df
        
    except Exception as e:
        print_error(f"Error en join_asof con {data_type_name}: {e}")
        # En caso de error, devolver el DataFrame base sin modificar
        return base_df


def _create_contextual_event_indicators(joined_df: pl.DataFrame) -> pl.DataFrame:
    """
    Crea indicadores binarios para eventos contextuales.

    Parámetros:
    -----------
    joined_df : pl.DataFrame
        DataFrame con datos unidos.

    Retorna:
    --------
    pl.DataFrame
        DataFrame con indicadores de eventos añadidos.
    """
    # Definir umbrales para crear indicadores binarios
    event_thresholds = {
        'sleep_quality': {'threshold': 0, 'indicator_name': 'sleep_event'},
        'work_intensity': {'threshold': 0, 'indicator_name': 'work_event'},
        'exercise_intensity': {'threshold': 0, 'indicator_name': 'exercise_event'}
    }
    
    expressions = []
    
    for col, config in event_thresholds.items():
        if col in joined_df.columns:
            # Crear indicador binario (1 si valor > umbral, 0 si no)
            indicator_expr = (
                pl.col(col).is_not_null() & (pl.col(col) > config['threshold'])
            ).cast(pl.Float64).alias(config['indicator_name'])
            
            expressions.append(indicator_expr)
    
    if expressions:
        joined_df = joined_df.with_columns(expressions)
    
    return joined_df

def _standardize_timestamp_column(df: pl.DataFrame, df_name: str) -> pl.DataFrame:
    """
    Estandariza el nombre de la columna de timestamp y su tipo de datos de forma segura,
    evitando duplicados.
    
    Parámetros:
    -----------
    df : pl.DataFrame
        DataFrame a estandarizar.
    df_name : str
        Nombre descriptivo del DataFrame para logging.
        
    Retorna:
    --------
    pl.DataFrame
        DataFrame con columna Timestamp estandarizada sin duplicados.
    """
    # Verificar si ya tiene 'Timestamp' y eliminar duplicados si existen
    timestamp_cols = [col for col in df.columns if 'Timestamp' in col]
    
    if len(timestamp_cols) > 1:
        print_warning(f"Múltiples columnas Timestamp en {df_name}: {timestamp_cols}")
        # Mantener solo 'Timestamp' principal, eliminar otras
        cols_to_drop = [col for col in timestamp_cols if col != 'Timestamp']
        if cols_to_drop:
            df = df.drop(cols_to_drop)
    
    # Si no tiene 'Timestamp', buscar alternativas
    if 'Timestamp' not in df.columns:
        if 'ts' in df.columns:
            df = df.rename({'ts': 'Timestamp'})
        elif 'ts_begin' in df.columns:
            df = df.rename({'ts_begin': 'Timestamp'})
        elif 'Time' in df.columns:
            df = df.rename({'Time': 'Timestamp'})
        else:
            raise ValueError(f"No se encontró columna de timestamp válida en DataFrame {df_name}. Columnas disponibles: {df.columns}")
    
    # Asegurar que Timestamp es datetime
    if df['Timestamp'].dtype != pl.Datetime(time_unit="us"):
        df = _convert_timestamp_to_datetime(df)
    
    return df

def _convert_timestamp_to_datetime(df: pl.DataFrame) -> pl.DataFrame:
    """Convierte la columna Timestamp a datetime probando diferentes formatos"""
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%d-%m-%Y %H:%M:%S",
        "%m-%d-%Y %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S"
    ]
    
    for fmt in formats:
        try:
            return df.with_columns(
                pl.col('Timestamp').str.strptime(pl.Datetime, fmt)
            )
        except Exception:
            continue
    
    raise ValueError("No se pudo convertir Timestamp a datetime con ningún formato conocido")

def _validate_critical_columns(cgm_df: pl.DataFrame, bolus_df: pl.DataFrame) -> None:
    """Valida que existan las columnas críticas necesarias"""
    if 'value' not in cgm_df.columns:
        raise ValueError("Columna 'value' no encontrada en cgm_df")
    if 'bolus' not in bolus_df.columns:
        raise ValueError("Columna 'bolus' no encontrada en bolus_df")

def _join_bolus_data(cgm_df: pl.DataFrame, bolus_df: pl.DataFrame) -> pl.DataFrame:
    """
    Une datos de CGM con datos de bolus de forma segura evitando duplicados de columnas.
    
    Parámetros:
    -----------
    cgm_df : pl.DataFrame
        DataFrame CGM.
    bolus_df : pl.DataFrame
        DataFrame bolus.
        
    Retorna:
    --------
    pl.DataFrame
        DataFrame con datos unidos sin duplicados de columnas.
    """
    # Verificar columnas comunes antes del join (excluyendo las de join)
    common_cols = set(cgm_df.columns) & set(bolus_df.columns)
    join_cols = {'Timestamp', 'SubjectID'}
    conflicting_cols = common_cols - join_cols
    
    if conflicting_cols:
        print_debug(f"Columnas en conflicto CGM-Bolus: {conflicting_cols}")
        # Eliminar columnas conflictivas del DataFrame de bolus
        bolus_for_join = bolus_df.drop(list(conflicting_cols))
    else:
        bolus_for_join = bolus_df
    
    joined_df = cgm_df.join_asof(
        bolus_for_join,
        on='Timestamp',
        by='SubjectID',
        tolerance='5m'
    )
    
    # Verificar que las columnas críticas se mantuvieron
    if 'value' not in joined_df.columns:
        raise ValueError("Columna 'value' perdida después de join con bolus")
    if 'bolus' not in joined_df.columns:
        raise ValueError("Columna 'bolus' perdida después de join con bolus")
    
    return joined_df

def _join_meal_data(joined_df: pl.DataFrame, meal_df: pl.DataFrame) -> pl.DataFrame:
    """
    Une datos de comidas al DataFrame ya unido de forma segura.
    
    Parámetros:
    -----------
    joined_df : pl.DataFrame
        DataFrame con datos ya unidos.
    meal_df : pl.DataFrame
        DataFrame meal.
        
    Retorna:
    --------
    pl.DataFrame
        DataFrame con datos de meal unidos sin duplicados.
    """
    # Verificar columnas comunes antes del join (excluyendo las de join)
    common_cols = set(joined_df.columns) & set(meal_df.columns)
    join_cols = {'Timestamp', 'SubjectID'}
    conflicting_cols = common_cols - join_cols
    
    if conflicting_cols:
        print_debug(f"Columnas en conflicto Joined-Meal: {conflicting_cols}")
        # Eliminar columnas conflictivas del DataFrame de meal
        meal_for_join = meal_df.drop(list(conflicting_cols))
    else:
        meal_for_join = meal_df
    
    joined_df = joined_df.join_asof(
        meal_for_join,
        on='Timestamp',
        by='SubjectID',
        tolerance='5m'
    )
    
    # Verificar que las columnas críticas se mantuvieron
    if 'value' not in joined_df.columns:
        raise ValueError("Columna 'value' perdida después de join con meal")
    if 'bolus' not in joined_df.columns:
        raise ValueError("Columna 'bolus' perdida después de join con bolus")
    
    return joined_df

def _join_physiological_data(joined_df: pl.DataFrame, physiological_df: pl.DataFrame) -> pl.DataFrame:
    """Une datos fisiológicos al DataFrame ya unido"""
    # Filtrar valores nulos antes de unir
    physiological_df = physiological_df.filter(pl.col('value').is_not_null())
    
    # Renombrar columna value a physiological_value antes de unir
    physiological_df = physiological_df.rename({'value': 'physiological_value'})
    
    joined_df = joined_df.join_asof(
        physiological_df,
        on='Timestamp',
        by='SubjectID',
        tolerance='5m',
        suffix='_physio'
    )
    
    return joined_df

def _prepare_dataframes_for_join(cgm_df: pl.DataFrame, bolus_df: pl.DataFrame, 
                                     meal_df: pl.DataFrame, physiological_df: Optional[pl.DataFrame]) -> tuple:
    """
    Prepara todos los DataFrames estandarizando timestamps y ordenando de forma segura,
    eliminando duplicados de columnas Timestamp.
    
    Parámetros:
    -----------
    cgm_df : pl.DataFrame
        DataFrame CGM.
    bolus_df : pl.DataFrame
        DataFrame bolus.
    meal_df : pl.DataFrame
        DataFrame meal.
    physiological_df : Optional[pl.DataFrame]
        DataFrame fisiológico opcional.
        
    Retorna:
    --------
    tuple
        Tupla con DataFrames preparados sin duplicados de Timestamp.
    """
    # Estandarizar nombres de columnas de timestamp y tipos de datos
    cgm_df = _standardize_timestamp_column(cgm_df, "CGM")
    bolus_df = _standardize_timestamp_column(bolus_df, "Bolus")
    meal_df = _standardize_timestamp_column(meal_df, "Meal")
    
    if physiological_df is not None:
        physiological_df = _standardize_timestamp_column(physiological_df, "Physiological")
    
    # Asegurar que los DataFrames estén ordenados por timestamp y SubjectID
    cgm_df = cgm_df.sort(['SubjectID', 'Timestamp'])
    bolus_df = bolus_df.sort(['SubjectID', 'Timestamp'])
    meal_df = meal_df.sort(['SubjectID', 'Timestamp'])
    
    if physiological_df is not None:
        physiological_df = physiological_df.sort(['SubjectID', 'Timestamp'])
    
    return cgm_df, bolus_df, meal_df, physiological_df

def join_signals(cgm_df: pl.DataFrame, 
                bolus_df: pl.DataFrame, 
                meal_df: pl.DataFrame,
                physiological_df: Optional[pl.DataFrame] = None,
                processed_data: Optional[Dict[str, pl.DataFrame]] = None) -> pl.DataFrame:
    """
    Une las diferentes señales temporales usando join_asof de forma segura, incluyendo 
    datos de sueño, trabajo y ejercicio, evitando duplicados de columnas Timestamp.

    Parámetros:
    -----------
    cgm_df : pl.DataFrame
        DataFrame con datos CGM.
    bolus_df : pl.DataFrame
        DataFrame con datos de bolus.
    meal_df : pl.DataFrame
        DataFrame con datos de comidas.
    physiological_df : Optional[pl.DataFrame], opcional
        DataFrame con datos fisiológicos (default: None).
    processed_data : Optional[Dict[str, pl.DataFrame]], opcional
        Diccionario con DataFrames procesados que incluye sleep, work, exercise (default: None).

    Retorna:
    --------
    pl.DataFrame
        DataFrame con todas las señales unidas.
    """
    try:
        print_debug(f"Iniciando join_signals - CGM: {cgm_df.shape}, Bolus: {bolus_df.shape}, Meal: {meal_df.shape}")
        
        # Preparar DataFrames eliminando duplicados de Timestamp
        cgm_df, bolus_df, meal_df, physiological_df = _prepare_dataframes_for_join(
            cgm_df, bolus_df, meal_df, physiological_df
        )
        
        # Validar columnas críticas
        _validate_critical_columns(cgm_df, bolus_df)
        
        # Unir datos paso a paso usando funciones seguras
        joined_df = _join_bolus_data(cgm_df, bolus_df)
        print_debug(f"Después de join bolus: {joined_df.shape}, columnas: {len(joined_df.columns)}")
        
        joined_df = _join_meal_data(joined_df, meal_df)
        print_debug(f"Después de join meal: {joined_df.shape}, columnas: {len(joined_df.columns)}")
        
        # Unir datos fisiológicos si existen
        if physiological_df is not None:
            joined_df = _join_physiological_data(joined_df, physiological_df)
            print_debug(f"Después de join physiological: {joined_df.shape}")
        
        # Unir con datos contextuales si están disponibles
        if processed_data is not None:
            joined_df = _join_contextual_data(joined_df, processed_data)
            print_debug(f"Después de join contextual: {joined_df.shape}")
        
        # Verificación final de columnas críticas
        if 'value' not in joined_df.columns or 'bolus' not in joined_df.columns:
            raise ValueError("Columnas críticas perdidas durante el proceso de join")
        
        # Verificación final de duplicados de Timestamp
        timestamp_cols = [col for col in joined_df.columns if 'Timestamp' in col]
        if len(timestamp_cols) > 1:
            print_warning(f"Múltiples columnas Timestamp detectadas al final: {timestamp_cols}")
            # Mantener solo la principal
            cols_to_drop = [col for col in timestamp_cols if col != 'Timestamp']
            joined_df = joined_df.drop(cols_to_drop)
        
        print_debug(f"Join signals completado exitosamente: {joined_df.shape}")
        return joined_df
        
    except Exception as e:
        print_error(f"Error en join_signals: {e}")
        print_error(f"CGM columnas: {cgm_df.columns}")
        print_error(f"Bolus columnas: {bolus_df.columns}")
        print_error(f"Meal columnas: {meal_df.columns}")
    
def ensure_timestamp_datetime(df: pl.DataFrame, col: str = TIMESTAMP_COL) -> pl.DataFrame:
    """
    Convierte la columna de timestamp a pl.Datetime, tolerando diferentes formatos.

    Parámetros:
    -----------
    df : pl.DataFrame
        DataFrame con la columna de timestamp.
    col : str, opcional
        Nombre de la columna de timestamp (default: "Timestamp").

    Retorna:
    --------
    pl.DataFrame
        DataFrame con la columna de timestamp convertida a pl.Datetime.
    """
    if col in df.columns:
        if df[col].dtype == pl.Object:
            df = df.with_columns(
                pl.col(col).map_elements(
                    lambda x: str(x) if x is not None else None,
                    return_dtype=pl.Utf8
                ).alias(col)
            )
        df = df.with_columns(
            pl.col(col).cast(pl.Datetime(time_unit="us"))
        )
    return df

def calculate_iob(bolus_time: datetime, basal_df: pl.DataFrame, half_life_hours: float = 4.0) -> float:
    """
    Calcula la insulina activa en el cuerpo (IOB).

    Parámetros:
    -----------
    bolus_time : datetime
        Tiempo del bolo de insulina.
    basal_df : pl.DataFrame
        DataFrame con datos de insulina basal.
    half_life_hours : float, opcional
        Vida media de la insulina en horas (default: 4.0).

    Retorna:
    --------
    float
        Cantidad de insulina activa en el organismo.
    """
    if basal_df is None or basal_df.is_empty():
        return 0.0
    
    iob: float = 0.0
    for row in basal_df.iter_rows(named=True):
        start_time: datetime = row[TIMESTAMP_COL]
        duration_hours: float = row["duration"] / (1000 * 3600) if "duration" in row else 0.0
        end_time: datetime = start_time + timedelta(hours=duration_hours)
        rate: float = row[BASAL_COL] if BASAL_COL in row and row[BASAL_COL] is not None else 0.9
        rate = min(rate, 2.0)
        
        if start_time <= bolus_time <= end_time:
            time_since_start: float = (bolus_time - start_time).total_seconds() / 3600
            remaining: float = rate * (1 - (time_since_start / half_life_hours))
            iob += max(0.0, remaining)
    
    return min(iob, CONFIG_PROCESSING["cap_iob"])

def get_cgm_window(bolus_time: datetime, cgm_df: pl.DataFrame, window_hours: int = CONFIG_PROCESSING["window_hours"]) -> np.ndarray:
    """
    Obtiene la ventana de datos CGM para un tiempo de bolo específico.

    Parámetros:
    -----------
    bolus_time : datetime
        Tiempo del bolo de insulina.
    cgm_df : pl.DataFrame
        DataFrame con datos CGM.
    window_hours : int, opcional
        Horas de la ventana de datos (default: 2).

    Retorna:
    --------
    np.ndarray
        Ventana de datos CGM o None si no hay suficientes datos.
    """
    window_start: datetime = bolus_time - timedelta(hours=window_hours)
    window: pl.DataFrame = cgm_df.filter(
        (pl.col(TIMESTAMP_COL) >= window_start) & (pl.col(TIMESTAMP_COL) <= bolus_time)
    ).sort(TIMESTAMP_COL).tail(24)
    
    if window.height < 24:
        return None
    return window.get_column(GLUCOSE_COL).to_numpy()

def generate_windows(df: pl.DataFrame, window_size: int = CONFIG_PROCESSING["window_steps"]) -> pl.DataFrame:
    """
    Genera ventanas de CGM de tamaño fijo antes de cada evento bolus.

    Parámetros:
    -----------
    df : pl.DataFrame
        DataFrame con datos de CGM y bolus.
    window_size : int, opcional
        Cantidad de pasos en la ventana (default: 24 para 2 horas con datos cada 5 min).

    Retorna:
    --------
    pl.DataFrame
        DataFrame con ventanas generadas.
    """
    windows: list[dict] = []
    bolus_events: pl.DataFrame = df.filter(pl.col(BOLUS_COL) > 0)
    for row in bolus_events.iter_rows(named=True):
        ts: datetime = row[TIMESTAMP_COL]
        subject_id: str = row[SUBJECT_ID_COL]
        cgm_window: pl.DataFrame = df.filter(
            (pl.col(SUBJECT_ID_COL) == subject_id) &
            (pl.col(TIMESTAMP_COL) <= ts) &
            (pl.col(TIMESTAMP_COL) > ts - timedelta(minutes=window_size*5))
        ).sort(TIMESTAMP_COL)
        if cgm_window.height == window_size:
            windows.append({
                SUBJECT_ID_COL: subject_id,
                TIMESTAMP_COL: ts,
                "cgm_window": cgm_window[GLUCOSE_COL].to_list(),
                BOLUS_COL: row[BOLUS_COL],
                MEAL_COL: row.get(MEAL_COL, 0.0)
            })
    return pl.DataFrame(windows)

def calculate_medians(bolus_df: pl.DataFrame, basal_df: pl.DataFrame) -> tuple[float, float]:
    """
    Calcula valores medianos para imputación de datos faltantes.

    Parámetros:
    -----------
    bolus_df : pl.DataFrame
        DataFrame con datos de bolos de insulina.
    basal_df : pl.DataFrame
        DataFrame con datos de insulina basal.

    Retorna:
    --------
    tuple[float, float]
        Tupla con (carb_median, iob_median).
    """
    non_zero_carbs: pl.Series = bolus_df.filter(pl.col("carbInput") > 0).get_column("carbInput")
    carb_median: float = non_zero_carbs.median() if len(non_zero_carbs) > 0 else 10.0
    
    iob_values: list[float] = []
    for row in bolus_df.iter_rows(named=True):
        iob: float = calculate_iob(row[TIMESTAMP_COL], basal_df)
        iob_values.append(iob)
    
    non_zero_iob: list[float] = [iob for iob in iob_values if iob > 0]
    iob_median: float = np.median(non_zero_iob) if non_zero_iob else 0.5
    
    return carb_median, iob_median

def compute_glucose_patterns_24h(cgm_values: list[float]) -> dict[str, float]:
    """
    Calcula patrones de glucosa de 24 horas para análisis clínico.
    """
    if not cgm_values or len(cgm_values) == 0:
        return {
            'cgm_mean_24h': 120.0,
            'cgm_std_24h': 0.0,
            'cgm_median_24h': 120.0,
            'cgm_range_24h': 0.0,
            'hypo_episodes_24h': 0,
            'hypo_percentage_24h': 0.0,
            'hyper_episodes_24h': 0,
            'hyper_percentage_24h': 0.0,
            'time_in_range_24h': 100.0,
            'cv_24h': 0.0,
            'mage_24h': 0.0,
            'glucose_trend_24h': 0.0
        }
    
    values_array = np.array(cgm_values)
    
    # Basic statistics
    patterns = {
        'cgm_mean_24h': float(np.mean(values_array)),
        'cgm_std_24h': float(np.std(values_array)),
        'cgm_median_24h': float(np.median(values_array)),
        'cgm_range_24h': float(np.max(values_array) - np.min(values_array))
    }
    
    # Hypoglycemia analysis (< 70 mg/dL)
    hypo_episodes = np.sum(values_array < CONFIG_PROCESSING['hypoglycemia_threshold'])
    patterns['hypo_episodes_24h'] = int(hypo_episodes)
    patterns['hypo_percentage_24h'] = float((hypo_episodes / len(values_array)) * 100)
    
    # Hyperglycemia analysis (> 180 mg/dL)
    hyper_episodes = np.sum(values_array > CONFIG_PROCESSING['hyperglycemia_threshold'])
    patterns['hyper_episodes_24h'] = int(hyper_episodes)
    patterns['hyper_percentage_24h'] = float((hyper_episodes / len(values_array)) * 100)
    
    # Time in Range (70-180 mg/dL)
    in_range = np.sum((values_array >= CONFIG_PROCESSING['tir_lower']) & (values_array <= CONFIG_PROCESSING['tir_upper']))
    patterns['time_in_range_24h'] = float((in_range / len(values_array)) * 100)
    
    # Glucose variability
    if len(values_array) > 1 and patterns['cgm_mean_24h'] > 0:
        patterns['cv_24h'] = float((patterns['cgm_std_24h'] / patterns['cgm_mean_24h']) * 100)
        glucose_changes = np.abs(np.diff(values_array))
        patterns['mage_24h'] = float(np.mean(glucose_changes))
        if len(values_array) >= 3:
            time_points = np.arange(len(values_array))
            slope, _ = np.polyfit(time_points, values_array, 1)
            patterns['glucose_trend_24h'] = float(slope)
        else:
            patterns['glucose_trend_24h'] = 0.0
    else:
        patterns['cv_24h'] = 0.0
        patterns['mage_24h'] = 0.0
        patterns['glucose_trend_24h'] = 0.0
    
    return patterns

def encode_time_cyclical(timestamp: datetime) -> dict[str, float]:
    """
    Codifica características de tiempo cíclicamente usando seno y coseno.
    """
    hour_decimal = timestamp.hour + timestamp.minute / 60.0 + timestamp.second / 3600.0
    hour_normalized = hour_decimal / 24.0
    hour_radians = 2 * np.pi * hour_normalized
    day_of_week = timestamp.weekday() / 7.0
    day_radians = 2 * np.pi * day_of_week
    return {
        'hour_sin': float(np.sin(hour_radians)),
        'hour_cos': float(np.cos(hour_radians)),
        'day_sin': float(np.sin(day_radians)),
        'day_cos': float(np.cos(day_radians)),
        'hour_of_day_normalized': float(hour_normalized),
        'day_of_week_normalized': float(day_of_week)
    }

def compute_enhanced_meal_context(bolus_time: datetime, meal_df: pl.DataFrame, 
                                window_hours: float = 2.0) -> dict[str, Union[float, int]]:
    """
    Calcula características mejoradas del contexto de comidas alrededor del tiempo del bolo.
    
    Parámetros:
    -----------
    bolus_time : datetime
        Tiempo del bolo de insulina
    meal_df : pl.DataFrame
        DataFrame con datos de comidas
    window_hours : float, opcional
        Horas de la ventana alrededor del bolo (default: 2.0)
        
    Retorna:
    --------
    Dict[str, Union[float, int]]
        Diccionario con características del contexto de comidas
    """
    if meal_df.is_empty():
        return {
            'meal_carbs': 0.0,
            'meal_time_diff_minutes': 0.0,
            'meal_time_diff_hours': 0.0,
            'has_meal': 0.0,
            'meals_in_window': 0,
            'significant_meal': 0.0,
            'total_carbs_window': 0.0,
            'largest_meal_carbs': 0.0,
            'meal_timing_score': 0.0
        }
    
    start_time = bolus_time - timedelta(hours=window_hours/2)
    end_time = bolus_time + timedelta(hours=window_hours)
    meals_in_window = meal_df.filter(
        (pl.col(TIMESTAMP_COL) >= start_time) & 
        (pl.col(TIMESTAMP_COL) <= end_time)
    )
    
    if meals_in_window.is_empty():
        return {
            'meal_carbs': 0.0,
            'meal_time_diff_minutes': 0.0,
            'meal_time_diff_hours': 0.0,
            'has_meal': 0.0,
            'meals_in_window': 0,
            'significant_meal': 0.0,
            'total_carbs_window': 0.0,
            'largest_meal_carbs': 0.0,
            'meal_timing_score': 0.0
        }
    
    meals_with_diff = meals_in_window.with_columns(
        (pl.col(TIMESTAMP_COL) - bolus_time).abs().alias("time_diff_abs")
    ).sort("time_diff_abs")
    
    closest_meal = meals_with_diff.row(0, named=True)
    time_diff = (closest_meal[TIMESTAMP_COL] - bolus_time).total_seconds()
    meal_carbs = float(closest_meal.get(MEAL_COL, 0.0))
    
    all_meals = meals_in_window.to_dicts()
    total_carbs = sum(float(meal.get(MEAL_COL, 0.0)) for meal in all_meals)
    largest_meal = max((float(meal.get(MEAL_COL, 0.0)) for meal in all_meals), default=0.0)
    
    max_diff_seconds = window_hours * 3600
    timing_score = max(0.0, 1.0 - abs(time_diff) / max_diff_seconds)
    
    return {
        'meal_carbs': meal_carbs,
        'meal_time_diff_minutes': float(time_diff / 60.0),
        'meal_time_diff_hours': float(time_diff / 3600.0),
        'has_meal': 1.0,
        'meals_in_window': len(all_meals),
        'significant_meal': 1.0 if meal_carbs > CONFIG_PROCESSING['significant_meal_threshold'] else 0.0,
        'total_carbs_window': total_carbs,
        'largest_meal_carbs': largest_meal,
        'meal_timing_score': timing_score
    }

def _get_default_risk_indicators() -> Dict[str, float]:
    """Retorna indicadores de riesgo por defecto cuando no hay datos de glucosa."""
    return {
        'hypo_risk': 0.0,
        'hyper_risk': 0.0,
        'variability_risk': 0.0,
        'sleep_hypo_risk': 0.0,
        'activity_hypo_risk': 0.0,
        'stress_hyper_risk': 0.0,
        'overall_risk': 0.0
    }

def _calculate_basic_glucose_risks(glucose_values: List[float]) -> tuple[float, float, float]:
    """Calcula riesgos básicos de hipoglucemia, hiperglucemia y variabilidad."""
    hypo_threshold = CONFIG_PROCESSING.get('hypo_threshold', 70)
    hyper_threshold = CONFIG_PROCESSING.get('hyper_threshold', 180)
    
    hypo_count = sum(1 for g in glucose_values if g < hypo_threshold)
    hyper_count = sum(1 for g in glucose_values if g > hyper_threshold)
    
    hypo_risk = hypo_count / len(glucose_values)
    hyper_risk = hyper_count / len(glucose_values)
    
    # Calcular riesgo de variabilidad
    if len(glucose_values) > 1:
        glucose_std = np.std(glucose_values)
        variability_risk = min(1.0, glucose_std / 50.0)
    else:
        variability_risk = 0.0
    
    return hypo_risk, hyper_risk, variability_risk

def _calculate_contextual_risk(event_name: str, physiological_data: Dict[str, List[float]], 
                              glucose_values: List[float], threshold: float, 
                              is_hypo_risk: bool = True) -> float:
    """Calcula riesgo contextual para un tipo específico de evento."""
    if event_name not in physiological_data:
        return 0.0
    
    event_indices = [i for i, event in enumerate(physiological_data[event_name]) 
                    if event > 0 and i < len(glucose_values)]
    
    if not event_indices:
        return 0.0
    
    event_glucose = [glucose_values[i] for i in event_indices]
    
    if is_hypo_risk:
        risk_count = sum(1 for g in event_glucose if g < threshold)
    else:
        risk_count = sum(1 for g in event_glucose if g > threshold)
    
    return risk_count / len(event_glucose)

def _calculate_physiological_risks(physiological_data: Optional[Dict[str, List[float]]], 
                                  glucose_values: List[float]) -> tuple[float, float, float]:
    """Calcula riesgos fisiológicos específicos."""
    if physiological_data is None:
        return 0.0, 0.0, 0.0
    
    hypo_threshold = CONFIG_PROCESSING.get('hypo_threshold', 70)
    hyper_threshold = CONFIG_PROCESSING.get('hyper_threshold', 180)
    
    sleep_hypo_risk = _calculate_contextual_risk(
        'sleep_event', physiological_data, glucose_values, hypo_threshold, True
    )
    
    activity_hypo_risk = _calculate_contextual_risk(
        'work_event', physiological_data, glucose_values, hypo_threshold, True
    )
    
    stress_hyper_risk = _calculate_contextual_risk(
        'stressors_event', physiological_data, glucose_values, hyper_threshold, False
    )
    
    return sleep_hypo_risk, activity_hypo_risk, stress_hyper_risk

def _calculate_overall_risk(hypo_risk: float, hyper_risk: float, variability_risk: float,
                           sleep_hypo_risk: float, activity_hypo_risk: float, 
                           stress_hyper_risk: float) -> float:
    """Calcula el riesgo general ponderado."""
    risk_weights = CONFIG_PROCESSING.get('risk_weights', {
        'hypo': 0.3,
        'hyper': 0.3,
        'variability': 0.2,
        'sleep_hypo': 0.1,
        'activity_hypo': 0.05,
        'stress_hyper': 0.05
    })
    
    return (
        risk_weights['hypo'] * hypo_risk +
        risk_weights['hyper'] * hyper_risk +
        risk_weights['variability'] * variability_risk +
        risk_weights['sleep_hypo'] * sleep_hypo_risk +
        risk_weights['activity_hypo'] * activity_hypo_risk +
        risk_weights['stress_hyper'] * stress_hyper_risk
    )

def compute_clinical_risk_indicators(glucose_values: List[float], 
                                   physiological_data: Optional[Dict[str, List[float]]] = None,
                                   time_values: Optional[List[datetime]] = None) -> Dict[str, float]:
    """
    Calcula indicadores de riesgo clínico basados en valores de glucosa y señales fisiológicas.
    
    Parámetros:
    -----------
    glucose_values : List[float]
        Lista de valores de glucosa
    physiological_data : Optional[Dict[str, List[float]]], opcional
        Diccionario con señales fisiológicas (default: None)
    time_values : Optional[List[datetime]], opcional
        Lista de timestamps correspondientes a los valores (default: None)
        
    Retorna:
    --------
    Dict[str, float]
        Diccionario con indicadores de riesgo clínico
    """
    if not glucose_values:
        return _get_default_risk_indicators()
    
    # Calcular riesgos básicos
    hypo_risk, hyper_risk, variability_risk = _calculate_basic_glucose_risks(glucose_values)
    
    # Calcular riesgos fisiológicos
    sleep_hypo_risk, activity_hypo_risk, stress_hyper_risk = _calculate_physiological_risks(
        physiological_data, glucose_values
    )
    
    # Calcular riesgo general
    overall_risk = _calculate_overall_risk(
        hypo_risk, hyper_risk, variability_risk,
        sleep_hypo_risk, activity_hypo_risk, stress_hyper_risk
    )
    
    return {
        'hypo_risk': hypo_risk,
        'hyper_risk': hyper_risk,
        'variability_risk': variability_risk,
        'sleep_hypo_risk': sleep_hypo_risk,
        'activity_hypo_risk': activity_hypo_risk,
        'stress_hyper_risk': stress_hyper_risk,
        'overall_risk': overall_risk
    }

def _extract_basic_glucose_stats(cgm_window: list) -> dict:
    """Extrae estadísticas básicas de la ventana CGM."""
    if not cgm_window:
        return {
            "glucose_last": 120.0,
            "glucose_mean": 120.0,
            "glucose_std": 0.0,
            "glucose_min": 120.0,
            "glucose_max": 120.0,
        }
    
    return {
        "glucose_last": cgm_window[-1],
        "glucose_mean": float(np.mean(cgm_window)),
        "glucose_std": float(np.std(cgm_window)),
        "glucose_min": float(np.min(cgm_window)),
        "glucose_max": float(np.max(cgm_window)),
    }

def _extract_exercise_features(row: dict) -> dict:
    """Extrae características relacionadas con ejercicio."""
    return {
        "exercise_intensity": row.get("exercise_intensity", 0.0),
        "exercise_duration": row.get("exercise_duration", 0.0),
        "exercise_in_window": row.get("exercise_in_window", 0.0),
        "steps_in_window": row.get("steps_in_window", 0.0),
    }

def _extract_hypo_features(row: dict) -> dict:
    """Extrae características de eventos hipoglucémicos."""
    return {
        "hypo_event_in_window": row.get("hypo_event_in_window", 0.0)
    }

def _extract_basal_features(row: dict) -> dict:
    """Extrae características de insulina basal."""
    return {
        "effective_basal_rate": row.get("effective_basal_rate", 0.0),
        "temp_basal_active": row.get("temp_basal_active", 0.0)
    }

def _extract_glucose_patterns_24h(row: dict, extended_cgm_df: pl.DataFrame) -> dict:
    """Extrae patrones de glucosa de 24 horas."""
    if extended_cgm_df is None:
        return {}
    
    subject_id = row[SUBJECT_ID_COL]
    ts = row[TIMESTAMP_COL]
    window_start = ts - timedelta(hours=24)
    
    extended_window = extended_cgm_df.filter(
        (pl.col(SUBJECT_ID_COL) == subject_id) &
        (pl.col(TIMESTAMP_COL) >= window_start) &
        (pl.col(TIMESTAMP_COL) <= ts)
    ).sort(TIMESTAMP_COL)
    
    if extended_window.height >= 288:  # 24 horas a 5 min
        return compute_glucose_patterns_24h(extended_window[GLUCOSE_COL].to_list())
    else:
        return compute_glucose_patterns_24h([])

def _apply_transformations(enhanced_df: pl.DataFrame) -> pl.DataFrame:
    """Aplica transformaciones logarítmicas y normalización."""
    # Normalización y log1p
    log_cols = [BOLUS_COL, MEAL_COL, "effective_basal_rate", "exercise_intensity", "exercise_duration", "steps_in_window"]
    for col in log_cols:
        if col in enhanced_df.columns:
            enhanced_df = enhanced_df.with_columns(
                pl.col(col).log1p().alias(f"{col}_log1p")
            )
    
    # Normalizar features binarios
    binary_cols = ["exercise_in_window", "hypo_event_in_window", "temp_basal_active", "has_meal", "significant_meal"]
    for col in binary_cols:
        if col in enhanced_df.columns:
            enhanced_df = enhanced_df.with_columns(
                pl.col(col).cast(pl.Float64)
            )
    
    return enhanced_df

def extract_features(df: pl.DataFrame, meal_df: pl.DataFrame, extended_cgm_df: pl.DataFrame = None) -> pl.DataFrame:
    """
    Extrae características mejoradas para DRL, incluyendo contexto de basal, ejercicio, pasos, hipo_event.
    """
    print_info("Extrayendo características mejoradas para DRL...")
    enhanced_rows = []
    
    for row in df.iter_rows(named=True):
        # Features base
        cgm_window = row.get("cgm_window", [])
        basic_stats = _extract_basic_glucose_stats(cgm_window)
        
        # Contexto de tiempo
        time_features = encode_time_cyclical(row[TIMESTAMP_COL])
        
        # Contexto de comidas
        meal_context = compute_enhanced_meal_context(row[TIMESTAMP_COL], meal_df)
        
        # Riesgo clínico
        iob = row.get("insulin_on_board", 0.0)
        risk_indicators = compute_clinical_risk_indicators(cgm_window, iob)
        
        # Características específicas
        exercise_features = _extract_exercise_features(row)
        hypo_features = _extract_hypo_features(row)
        basal_features = _extract_basal_features(row)
        
        # Patrones de glucosa de 24h
        glucose_patterns = _extract_glucose_patterns_24h(row, extended_cgm_df)
        
        enhanced_row = {
            **row,
            **basic_stats,
            **time_features,
            **meal_context,
            **risk_indicators,
            **exercise_features,
            **hypo_features,
            **basal_features,
            **glucose_patterns
        }
        enhanced_rows.append(enhanced_row)
    
    enhanced_df = pl.DataFrame(enhanced_rows)
    enhanced_df = _apply_transformations(enhanced_df)
    
    print_info(f"Características DRL extraídas. Forma: {enhanced_df.shape}")
    return enhanced_df

def transform_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Aplica transformaciones mejoradas incluyendo transformaciones logarítmicas, normalización y expansión de características.
    """
    print_info("Aplicando transformaciones mejoradas...")
    
    # Log1p transformations
    log_transform_cols = [
        BOLUS_COL, "carb_input", MEAL_COL, "insulin_on_board",
        "total_carbs_window", "largest_meal_carbs"
    ]
    
    for col in log_transform_cols:
        if col in df.columns:
            df = df.with_columns(
                pl.col(col).log1p().alias(f"{col}_log1p")
            )

    # Normalize percentage features
    percentage_cols = [
        "hypo_percentage_24h", "hyper_percentage_24h", "time_in_range_24h", "cv_24h"
    ]
    
    for col in percentage_cols:
        if col in df.columns:
            df = df.with_columns(
                (pl.col(col) / 100.0).alias(f"{col}_normalized")
            )
    
    # Normalize time features
    if "meal_time_diff_hours" in df.columns:
        df = df.with_columns(
            (pl.col("meal_time_diff_hours") / 24.0).alias("meal_time_diff_normalized")
        )
    
    # Normalize glucose-related features
    glucose_norm_cols = [
        ("cgm_mean_24h", 200.0),
        ("cgm_std_24h", 100.0),
        ("cgm_median_24h", 200.0),
        ("cgm_range_24h", 300.0),
        ("mage_24h", 50.0),
        ("glucose_trend_24h", 10.0)
    ]
    
    for col, norm_factor in glucose_norm_cols:
        if col in df.columns:
            df = df.with_columns(
                (pl.col(col) / norm_factor).alias(f"{col}_normalized")
            )
    
    # Expand CGM window to individual columns
    if "cgm_window" in df.columns:
        window_size = CONFIG_PROCESSING["window_steps"]
        for i in range(window_size):
            df = df.with_columns(
                pl.col("cgm_window").list.get(i, null_on_oob=True)
                .fill_null(120.0)
                .alias(f"cgm_{i}")
            )
        df = df.drop("cgm_window")

    # Create risk composite scores
    if all(col in df.columns for col in ["current_hypo_risk", "stability_score", "iob_risk_factor"]):
        df = df.with_columns(
            (pl.col("current_hypo_risk") + pl.col("iob_risk_factor") * 0.5).alias("composite_hypo_risk")
        )
    
    if all(col in df.columns for col in ["current_hyper_risk", "glucose_rate_of_change"]):
        df = df.with_columns(
            (pl.col("current_hyper_risk") + (pl.col("glucose_rate_of_change") / 10.0).clip(0, 1)).alias("composite_hyper_risk")
        )
    
    print_info(f"Transformaciones mejoradas completadas. Forma final: {df.shape}")
    return df

def extract_features_excel(row: dict, cgm_window: np.ndarray, carb_median: float, iob_median: float, basal_df: pl.DataFrame, idx: int) -> Optional[dict]:
    """
    Extrae características para una instancia de bolo individual desde datos Excel.

    Parámetros:
    -----------
    row : dict
        Fila con datos del bolo.
    cgm_window : np.ndarray
        Ventana de datos CGM.
    carb_median : float
        Valor mediano de carbohidratos para imputación.
    iob_median : float
        Valor mediano de IOB para imputación.
    basal_df : pl.DataFrame
        DataFrame con datos de insulina basal.
    idx : int
        Índice del sujeto.

    Retorna:
    --------
    dict
        Diccionario con características extraídas o None si no hay datos suficientes.
    """
    bolus_time: datetime = row[TIMESTAMP_COL]
    if cgm_window is None:
        return None
    
    iob: float = calculate_iob(bolus_time, basal_df)
    iob = iob_median if iob == 0 else iob
    iob = np.clip(iob, 0, CONFIG_PROCESSING["cap_iob"])
    
    hour_of_day: float = bolus_time.hour / 23.0
    
    bg_input: float = row["bgInput"] if row["bgInput"] is not None else cgm_window[-1]
    bg_input = max(bg_input, 50.0)
    bg_input = np.clip(bg_input, 0, CONFIG_PROCESSING["cap_bg"])
    
    normal: float = row["normal"] if row["normal"] is not None else 0.0
    normal = np.clip(normal, 0, CONFIG_PROCESSING["cap_normal"])
    
    isf_custom: float = 50.0 if normal <= 0 else (bg_input - 100) / normal
    isf_custom = np.clip(isf_custom, 10, 100)
    
    carb_input: float = row["carbInput"] if row["carbInput"] is not None else 0.0
    carb_input = carb_median if carb_input == 0 else carb_input
    carb_input = np.clip(carb_input, 0, CONFIG_PROCESSING["cap_carb"])
    
    insulin_carb_ratio: float = row["insulinCarbRatio"] if row["insulinCarbRatio"] is not None else 10.0
    insulin_carb_ratio = np.clip(insulin_carb_ratio, 5, 20)
    
    return {
        'subject_id': idx,  # Mantener como int
        TIMESTAMP_COL: bolus_time,  # Añadir Timestamp
        'cgm_window': cgm_window.tolist(),  # Convertir a lista para evitar problemas con NumPy array
        'carb_input': float(carb_input),
        'bg_input': float(bg_input),
        'insulin_carb_ratio': float(insulin_carb_ratio),
        'insulin_sensitivity_factor': float(isf_custom),
        'insulin_on_board': float(iob),
        'hour_of_day': float(hour_of_day),
        BOLUS_COL: float(normal)
    }

def _add_single_context_data(df: pl.DataFrame, context_df: pl.DataFrame, 
                            data_type: str, prepare_func, column_name: str) -> pl.DataFrame:
    """
    Agrega un tipo específico de datos de contexto al DataFrame principal.
    
    Parámetros:
    -----------
    df : pl.DataFrame
        DataFrame principal.
    context_df : pl.DataFrame
        DataFrame con datos de contexto.
    data_type : str
        Tipo de datos de contexto (para logging).
    prepare_func : callable
        Función para preparar los datos de contexto.
    column_name : str
        Nombre de la columna a agregar.
    
    Retorna:
    --------
    pl.DataFrame
        DataFrame con datos de contexto agregados.
    """
    if context_df.is_empty():
        print_warning(f"No se encontraron datos de {data_type} en el XML procesado.")
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(column_name))
    
    try:
        prepared_data = prepare_func(context_df)
        
        if not prepared_data.is_empty():
            return _join_contextual_data_by_date(
                df, prepared_data, column_name, f'datos de {data_type}'
            )
        else:
            print_warning(f"No se pudieron preparar datos de {data_type} válidos para join")
            return df.with_columns(pl.lit(None).cast(pl.Float64).alias(column_name))
            
    except Exception as e:
        print_error(f"Error procesando datos de {data_type}: {e}")
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(column_name))

def add_context_data(df: pl.DataFrame, processed_data: dict) -> pl.DataFrame:
    """
    Agrega datos de contexto adicionales al DataFrame procesado usando joins por SubjectID y fecha.
    
    Parámetros:
    -----------
    df : pl.DataFrame
        DataFrame con datos procesados.
    processed_data : dict
        Diccionario con datos procesados del XML.
    
    Retorna:
    --------
    pl.DataFrame
        DataFrame con datos de contexto agregados.
    """
    print_debug(f"{processed_data.keys()=}")
    
    # Asegurar que df tiene columna Timestamp como datetime
    if 'Timestamp' not in df.columns:
        print_warning("Columna 'Timestamp' no encontrada en DataFrame principal")
        return df
    
    df = ensure_timestamp_datetime(df, 'Timestamp')
    
    # Configuración de tipos de datos de contexto
    context_configs = [
        {
            'key': 'sleep',
            'data_type': 'sueño',
            'prepare_func': _prepare_sleep_data_for_join,
            'column_name': 'sleep_quality'
        },
        {
            'key': 'work',
            'data_type': 'trabajo',
            'prepare_func': _prepare_work_data_for_join,
            'column_name': 'work_intensity'
        },
        {
            'key': 'exercise',
            'data_type': 'ejercicio',
            'prepare_func': _prepare_exercise_data_for_join,
            'column_name': 'exercise_intensity'
        }
    ]
    
    # Procesar cada tipo de datos de contexto
    for config in context_configs:
        context_df = processed_data.get(config['key'], pl.DataFrame())
        df = _add_single_context_data(
            df, context_df, config['data_type'], 
            config['prepare_func'], config['column_name']
        )
    
    return df

def _prepare_sleep_data_for_join(sleep_df: pl.DataFrame) -> pl.DataFrame:
    """
    Prepara los datos de sueño para join por fecha usando ts_end.
    
    Parámetros:
    -----------
    sleep_df : pl.DataFrame
        DataFrame con datos de sueño.
    
    Retorna:
    --------
    pl.DataFrame
        DataFrame preparado para join.
    """
    if 'quality' not in sleep_df.columns:
        print_warning("Columna 'quality' no encontrada en datos de sueño")
        return pl.DataFrame()
    
    # Determinar qué columna de timestamp usar
    timestamp_col = None
    if 'Timestamp_end' in sleep_df.columns:
        timestamp_col = 'Timestamp_end'
    elif 'ts_end' in sleep_df.columns:
        timestamp_col = 'ts_end'
    elif 'Timestamp' in sleep_df.columns:
        timestamp_col = 'Timestamp'
    else:
        print_warning("No se encontró columna de timestamp válida en datos de sueño")
        return pl.DataFrame()
    
    # Preparar DataFrame para join
    sleep_prepared = sleep_df.clone()
    
    # Asegurar que timestamp es datetime
    sleep_prepared = ensure_timestamp_datetime(sleep_prepared, timestamp_col)
    
    # Crear columna de fecha para matching
    sleep_prepared = sleep_prepared.with_columns([
        pl.col(timestamp_col).dt.date().alias('join_date'),
        pl.col('quality').cast(pl.Float64).alias('sleep_quality')
    ])
    
    # Filtrar valores válidos y seleccionar columnas necesarias
    sleep_prepared = sleep_prepared.filter(
        pl.col('sleep_quality').is_not_null() & 
        pl.col('join_date').is_not_null() &
        pl.col('SubjectID').is_not_null()
    ).select(['SubjectID', 'join_date', 'sleep_quality'])
    
    # Agrupar por SubjectID y fecha para obtener valor promedio si hay múltiples registros
    sleep_prepared = sleep_prepared.group_by(['SubjectID', 'join_date']).agg(
        pl.col('sleep_quality').mean().alias('sleep_quality')
    )
    
    return sleep_prepared

def _prepare_work_data_for_join(work_df: pl.DataFrame) -> pl.DataFrame:
    """
    Prepara los datos de trabajo para join por fecha.
    
    Parámetros:
    -----------
    work_df : pl.DataFrame
        DataFrame con datos de trabajo.
    
    Retorna:
    --------
    pl.DataFrame
        DataFrame preparado para join.
    """
    if 'intensity' not in work_df.columns:
        print_warning("Columna 'intensity' no encontrada en datos de trabajo")
        return pl.DataFrame()
    
    # Determinar qué columna de timestamp usar
    timestamp_col = None
    if 'Timestamp' in work_df.columns:
        timestamp_col = 'Timestamp'
    elif 'ts' in work_df.columns:
        timestamp_col = 'ts'
    elif 'Timestamp_begin' in work_df.columns:
        timestamp_col = 'Timestamp_begin'
    else:
        print_warning("No se encontró columna de timestamp válida en datos de trabajo")
        return pl.DataFrame()
    
    # Preparar DataFrame para join
    work_prepared = work_df.clone()
    
    # Asegurar que timestamp es datetime
    work_prepared = ensure_timestamp_datetime(work_prepared, timestamp_col)
    
    # Crear columna de fecha para matching
    work_prepared = work_prepared.with_columns([
        pl.col(timestamp_col).dt.date().alias('join_date'),
        pl.col('intensity').cast(pl.Float64).alias('work_intensity')
    ])
    
    # Filtrar valores válidos y seleccionar columnas necesarias
    work_prepared = work_prepared.filter(
        pl.col('work_intensity').is_not_null() & 
        pl.col('join_date').is_not_null() &
        pl.col('SubjectID').is_not_null()
    ).select(['SubjectID', 'join_date', 'work_intensity'])
    
    # Agrupar por SubjectID y fecha para obtener valor promedio si hay múltiples registros
    work_prepared = work_prepared.group_by(['SubjectID', 'join_date']).agg(
        pl.col('work_intensity').mean().alias('work_intensity')
    )
    
    return work_prepared

def _prepare_exercise_data_for_join(exercise_df: pl.DataFrame) -> pl.DataFrame:
    """
    Prepara los datos de ejercicio para join por fecha.
    
    Parámetros:
    -----------
    exercise_df : pl.DataFrame
        DataFrame con datos de ejercicio.
    
    Retorna:
    --------
    pl.DataFrame
        DataFrame preparado para join.
    """
    if 'intensity' not in exercise_df.columns:
        print_warning("Columna 'intensity' no encontrada en datos de ejercicio")
        return pl.DataFrame()
    
    # Determinar qué columna de timestamp usar
    timestamp_col = None
    if 'Timestamp' in exercise_df.columns:
        timestamp_col = 'Timestamp'
    elif 'ts' in exercise_df.columns:
        timestamp_col = 'ts'
    else:
        print_warning("No se encontró columna de timestamp válida en datos de ejercicio")
        return pl.DataFrame()
    
    # Preparar DataFrame para join
    exercise_prepared = exercise_df.clone()
    
    # Asegurar que timestamp es datetime
    exercise_prepared = ensure_timestamp_datetime(exercise_prepared, timestamp_col)
    
    # Crear columna de fecha para matching
    exercise_prepared = exercise_prepared.with_columns([
        pl.col(timestamp_col).dt.date().alias('join_date'),
        pl.col('intensity').cast(pl.Float64).alias('exercise_intensity')
    ])
    
    # Filtrar valores válidos y seleccionar columnas necesarias
    exercise_prepared = exercise_prepared.filter(
        pl.col('exercise_intensity').is_not_null() & 
        pl.col('join_date').is_not_null() &
        pl.col('SubjectID').is_not_null()
    ).select(['SubjectID', 'join_date', 'exercise_intensity'])
    
    # Agrupar por SubjectID y fecha para obtener valor promedio si hay múltiples registros
    exercise_prepared = exercise_prepared.group_by(['SubjectID', 'join_date']).agg(
        pl.col('exercise_intensity').mean().alias('exercise_intensity')
    ).fill_null(0.0)
    
    return exercise_prepared

def _join_contextual_data_by_date(
    df: pl.DataFrame, 
    contextual_df: pl.DataFrame, 
    value_column: str,
    description: str
) -> pl.DataFrame:
    """
    Une datos contextuales al DataFrame principal por SubjectID y fecha.
    
    Parámetros:
    -----------
    df : pl.DataFrame
        DataFrame principal.
    contextual_df : pl.DataFrame
        DataFrame contextual preparado.
    value_column : str
        Nombre de la columna de valor a unir.
    description : str
        Descripción del tipo de datos para logging.
    
    Retorna:
    --------
    pl.DataFrame
        DataFrame con datos contextuales unidos.
    """
    if contextual_df.is_empty():
        print_warning(f"DataFrame de {description} está vacío")
        df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(value_column))
        return df
    
    try:
        # Crear columna de fecha en el DataFrame principal
        df_with_date = df.with_columns(
            pl.col('Timestamp').dt.date().alias('join_date')
        )
        
        # Realizar join por SubjectID y fecha
        joined_df = df_with_date.join(
            contextual_df,
            on=['SubjectID', 'join_date'],
            how='left')
        
        # Eliminar columna temporal de fecha
        joined_df = joined_df.drop('join_date')
        
        # Verificar resultado del join
        non_null_count = joined_df[value_column].is_not_null().sum()
        total_count = len(joined_df)
        coverage_percentage = (non_null_count / total_count * 100) if total_count > 0 else 0
        
        print_info(f"Join exitoso con {description}: {coverage_percentage:.1f}% cobertura")
        
        return joined_df
        
    except Exception as e:
        print_error(f"Error en join con {description}: {e}")
        # En caso de error, añadir columna con valores nulos
        df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(value_column))
        return df

def process_xml_directory(data_dir: str) -> Optional[pl.DataFrame]:
    """
    Procesa un directorio XML completo, extrayendo y transformando características.
    
    Parámetros:
    -----------
    data_dir : str
        Directorio con archivos XML
        
    Retorna:
    --------
    Optional[pl.DataFrame]
        DataFrame procesado o None si hubo un error
    """
    try:
        print_info(f"Procesando directorio XML: {data_dir}")
        data = load_data(data_dir)
        
        # Preprocesar datos de bolus y meal
        processed_data = preprocess_bolus_meal(data)
        
        # Preprocesar datos de CGM
        cgm_data = preprocess_cgm(data.get("glucose_level"))
        
        # Extraer los DataFrames necesarios
        bolus_df = processed_data.get("bolus", pl.DataFrame())
        meal_df = processed_data.get("meal", pl.DataFrame())
        
        # Unir señales con los parámetros correctos
        df = join_signals(cgm_data, bolus_df, meal_df)
        
        # Crear indicadores de eventos contextuales
        df = _create_contextual_event_indicators(df)
        
        # Extraer características mejoradas
        df = extract_enhanced_features(df, processed_data.get("meal"))
        
        # Transformar características
        df = transform_enhanced_features(df)
        
        df = add_context_data(df, processed_data)
        
        print_info(f"Procesado exitoso para {data_dir}: {df.shape}")
        return df
    except Exception as e:
        print_error(f"Error procesando {data_dir}: {str(e)}")
        return None

def process_excel_subject(subject_path: str, idx: int) -> list[dict]:
    """
    Procesa los datos de un sujeto desde un archivo Excel.

    Parámetros:
    -----------
    subject_path : str
        Ruta al archivo del sujeto.
    idx : int
        Índice del sujeto.

    Retorna:
    --------
    list[dict]
        Lista de diccionarios con características procesadas.
    """
    start_time: float = time.time()
    print_info(f"Procesando {os.path.basename(subject_path)} (Sujeto {idx+1})...")
    
    cgm_df, bolus_df, basal_df = load_excel_data(subject_path)
    if cgm_df is None or bolus_df is None:
        return []

    carb_median, iob_median = calculate_medians(bolus_df, basal_df)
    processed_data: list[dict] = []
    for row in tqdm(bolus_df.iter_rows(named=True), total=len(bolus_df), desc=f"Procesando {os.path.basename(subject_path)}"):
        bolus_time: datetime = row[TIMESTAMP_COL]
        cgm_window: np.ndarray = get_cgm_window(bolus_time, cgm_df)
        features: dict = extract_features_excel(row, cgm_window, carb_median, iob_median, basal_df, idx)
        if features is not None:
            processed_data.append(features)

    elapsed_time: float = time.time() - start_time
    print_info(f"Procesado {os.path.basename(subject_path)} (Sujeto {idx+1}) en {elapsed_time:.2f} segundos")
    return processed_data

def calculate_stats_for_group(df_final_pd: pl.DataFrame, subjects: list, feature: str = 'bolus') -> tuple:
    """
    Calcula media y desviación estándar para un grupo de sujetos.

    Parámetros:
    -----------
    df_final_pd : pl.DataFrame
        DataFrame con datos procesados en formato pandas
    subjects : list
        Lista de IDs de sujetos
    feature : str, opcional
        Característica para calcular estadísticas (default: 'bolus')

    Retorna:
    --------
    tuple
        Tupla con (media, desviación estándar)
    """
    if not subjects:
        return 0, 0
    mask = df_final_pd['subject_id'].isin(subjects)
    values = df_final_pd.loc[mask, feature]
    return values.mean(), values.std()

def calculate_distribution_score(means: list, stds: list) -> float:
    """
    Calcula una puntuación de distribución basada en medias y desviaciones estándar.

    Parámetros:
    -----------
    means : list
        Lista de valores medios
    stds : list
        Lista de desviaciones estándar

    Retorna:
    --------
    float
        Puntuación que representa la variabilidad de la distribución
    """
    if not all(m != 0 for m in means):
        return float('inf')
    
    range_means = max(means) - min(means)
    range_stds = max(stds) - min(stds) if all(s != 0 for s in stds) else float('inf')
    return range_means + range_stds

def assign_subject_to_group(df_final_pd: pl.DataFrame, subject: int, 
                           train_subjects: list, val_subjects: list, test_subjects: list,
                           train_size: int, val_size: int, test_size: int) -> tuple:
    """
    Asigna un sujeto a un grupo de entrenamiento, validación o prueba basado en balance.

    Parámetros:
    -----------
    df_final_pd : pl.DataFrame
        DataFrame con datos procesados en formato pandas
    subject : int
        ID del sujeto a asignar
    train_subjects : list
        Lista actual de sujetos de entrenamiento
    val_subjects : list
        Lista actual de sujetos de validación
    test_subjects : list
        Lista actual de sujetos de prueba
    train_size : int
        Tamaño máximo del grupo de entrenamiento
    val_size : int
        Tamaño máximo del grupo de validación
    test_size : int
        Tamaño máximo del grupo de prueba

    Retorna:
    --------
    tuple
        Tupla con listas actualizadas (train_subjects, val_subjects, test_subjects)
    """
    # Calculate current stats
    train_mean, train_std = calculate_stats_for_group(df_final_pd, train_subjects)
    val_mean, val_std = calculate_stats_for_group(df_final_pd, val_subjects)
    test_mean, test_std = calculate_stats_for_group(df_final_pd, test_subjects)
    
    # Calculate potential stats if subject is added to each group
    train_temp = train_subjects + [subject]
    val_temp = val_subjects + [subject]
    test_temp = test_subjects + [subject]
    
    train_mean_new, train_std_new = calculate_stats_for_group(df_final_pd, train_temp)
    val_mean_new, val_std_new = calculate_stats_for_group(df_final_pd, val_temp)
    test_mean_new, test_std_new = calculate_stats_for_group(df_final_pd, test_temp)
    
    # Calculate scores for each option
    score_if_train = calculate_distribution_score(
        [train_mean_new, val_mean, test_mean], 
        [train_std_new, val_std, test_std]
    )
    score_if_val = calculate_distribution_score(
        [train_mean, val_mean_new, test_mean], 
        [train_std, val_std_new, test_std]
    )
    score_if_test = calculate_distribution_score(
        [train_mean, val_mean, test_mean_new], 
        [train_std, val_std, test_std_new]
    )
    
    # Assign to the group with best balance
    if len(train_subjects) < train_size and score_if_train <= min(score_if_val, score_if_test):
        train_subjects.append(subject)
    elif len(val_subjects) < val_size and score_if_val <= min(score_if_train, score_if_test):
        val_subjects.append(subject)
    elif len(test_subjects) < test_size:
        test_subjects.append(subject)
    else:
        train_subjects.append(subject)
    
    return train_subjects, val_subjects, test_subjects

def prepare_data_with_scaler(df_final_pd: pl.DataFrame, mask: pl.Series, 
                            columns: list, scaler: StandardScaler, reshape: tuple = None) -> np.ndarray:
    """
    Prepara datos con transformación StandardScaler.

    Parámetros:
    -----------
    df_final_pd : pl.DataFrame
        DataFrame con datos procesados en formato pandas
    mask : pd.Series
        Máscara booleana para seleccionar filas
    columns : list
        Lista de columnas para seleccionar
    scaler : StandardScaler
        Escalador ajustado previamente
    reshape : tuple, opcional
        Nueva forma para los datos transformados (default: None)

    Retorna:
    --------
    np.ndarray
        Array con datos transformados y opcionalmente reshapeados
    """
    data = scaler.transform(df_final_pd.loc[mask, columns])
    if reshape:
        data = data.reshape(*reshape)
    return data

def load_data(data_dir: str) -> Dict[str, pl.DataFrame]:
    """
    Carga los datos y muestra las columnas de cada DataFrame.
    Verifica que estén presentes los 6 sujetos esperados para cada año (2018 y 2020).
    Ahora soporta: glucose_level, bolus, meal, basal, temp_basal, exercise, basis_steps, hypo_event,
    finger_stick, sleep, work, stressors, illness, basis_heart_rate, basis_gsr, basis_skin_temperature,
    basis_air_temperature, basis_sleep, acceleration.
    
    Parámetros:
    -----------
    data_dir : str
        Directorio con archivos XML.
        
    Retorna:
    --------
    Dict[str, pl.DataFrame]
        Diccionario con DataFrames por tipo de dato.
    """
    print_info(f"Cargando datos desde {data_dir}")
    
    # Determinar año y sujetos esperados
    year, expected_subjects = _determine_year_and_subjects(data_dir)
    
    # Validar archivos XML
    xml_files = _validate_xml_files_existence(data_dir, expected_subjects, year)
    
    # Procesar archivos XML
    data_dict, subject_stats = _process_all_xml_files(xml_files, year)
    
    # Validar resultados finales
    _validate_final_results(data_dict, subject_stats, expected_subjects, year)
    
    return data_dict

def _determine_year_and_subjects(data_dir: str) -> tuple[str, list[str]]:
    """
    Determina el año y los sujetos esperados basándose en el directorio.
    
    Parámetros:
    -----------
    data_dir : str
        Directorio con archivos XML.
        
    Retorna:
    --------
    tuple[str, list[str]]
        Tupla con (año, lista_de_sujetos_esperados).
    """
    expected_subjects = {
        '2018': ['559-ws-training', '563-ws-training', '570-ws-training', '575-ws-training', '588-ws-training', '591-ws-training'],
        '2020': ['540-ws-training', '544-ws-training', '552-ws-training', '567-ws-training', '584-ws-training', '596-ws-training']
    }
    
    if '2018' in data_dir:
        year = '2018'
    elif '2020' in data_dir:
        year = '2020'
    else:
        year = None
    
    if year is None:
        raise ValueError(f"No se pudo determinar el año del directorio: {data_dir}")
    
    suffix = '-ws-training' if 'train' in data_dir else '-ws-testing'
    adjusted_subjects = [s.replace('-ws-training', suffix).replace('-ws-testing', suffix) for s in expected_subjects[year]]
    
    return year, adjusted_subjects

def _validate_xml_files_existence(data_dir: str, expected_subjects: list[str], year: str) -> list[str]:
    """
    Valida la existencia de archivos XML y reporta sujetos faltantes.
    
    Parámetros:
    -----------
    data_dir : str
        Directorio con archivos XML.
    expected_subjects : list[str]
        Lista de sujetos esperados.
    year : str
        Año de los datos.
        
    Retorna:
    --------
    list[str]
        Lista de archivos XML encontrados.
    """
    xml_files = glob.glob(os.path.join(data_dir, "*.xml"))
    found_subjects = [os.path.basename(f).split('.')[0] for f in xml_files]
    missing_subjects = [s for s in expected_subjects if s not in found_subjects]
    
    if missing_subjects:
        print_error(f"Faltan datos para sujetos del año {year}: {missing_subjects}")
        if not found_subjects:
            raise ValueError(f"No se encontraron archivos XML en {data_dir}")
    
    return xml_files

def _get_expected_data_types() -> list[str]:
    """
    Retorna la lista de tipos de datos esperados.
    
    Retorna:
    --------
    list[str]
        Lista de tipos de datos esperados.
    """
    return [
        'glucose_level', 'bolus', 'meal', 'basal', 'temp_basal', 'exercise', 'basis_steps', 'hypo_event',
        'finger_stick', 'sleep', 'work', 'stressors', 'illness', 'basis_heart_rate', 'basis_gsr',
        'basis_skin_temperature', 'basis_air_temperature', 'basis_sleep', 'acceleration'
    ]

def _process_all_xml_files(xml_files: list[str], year: str) -> tuple[dict[str, pl.DataFrame], dict[str, dict[str, int]]]:
    """
    Procesa todos los archivos XML y retorna los datos consolidados.
    
    Parámetros:
    -----------
    xml_files : list[str]
        Lista de archivos XML a procesar.
    year : str
        Año de los datos.
        
    Retorna:
    --------
    tuple[dict[str, pl.DataFrame], dict[str, dict[str, int]]]
        Tupla con (diccionario_de_datos, estadísticas_por_sujeto).
    """
    data_dict = {}
    subject_stats = defaultdict(lambda: defaultdict(int))
    expected_types = _get_expected_data_types()
    
    for xml_file in xml_files:
        file_data, file_stats = _process_single_xml_file_data(xml_file, year, expected_types)
        
        # Consolidar datos
        _consolidate_file_data(data_dict, file_data)
        subject_stats.update(file_stats)
    
    return data_dict, subject_stats

def _process_single_xml_file_data(xml_file: str, year: str, expected_types: list[str]) -> tuple[dict[str, pl.DataFrame], dict[str, dict[str, int]]]:
    """
    Procesa un único archivo XML.
    
    Parámetros:
    -----------
    xml_file : str
        Ruta al archivo XML.
    year : str
        Año de los datos.
    expected_types : list[str]
        Tipos de datos esperados.
        
    Retorna:
    --------
    tuple[dict[str, pl.DataFrame], dict[str, dict[str, int]]]
        Tupla con (datos_del_archivo, estadísticas_del_archivo).
    """
    subject_id = os.path.basename(xml_file).split('.')[0]
    numeric_id = extract_numeric_id(subject_id)
    
    print_info(f"\n{'='*50}")
    print_info(f"Procesando SubjectID: {subject_id} (ID numérico: {numeric_id}, Año {year})")
    print_info(f"{'='*50}")
    
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        return _extract_data_from_xml_root(root, subject_id, year, expected_types)
    except Exception as e:
        print_error(f"Error procesando {xml_file}: {e}")
        return {}, {}

def _extract_data_from_xml_root(root, subject_id: str, year: str, expected_types: list[str]) -> tuple[dict[str, pl.DataFrame], dict[str, dict[str, int]]]:
    """
    Extrae datos del elemento raíz del XML.
    
    Parámetros:
    -----------
    root : ET.Element
        Elemento raíz del XML.
    subject_id : str
        ID del sujeto.
    year : str
        Año de los datos.
    expected_types : list[str]
        Tipos de datos esperados.
        
    Retorna:
    --------
    tuple[dict[str, pl.DataFrame], dict[str, dict[str, int]]]
        Tupla con (datos_extraídos, estadísticas).
    """
    file_data = {}
    file_stats = defaultdict(int)
    
    for data_type_elem in root:
        data_type = data_type_elem.tag
        
        if _should_skip_data_type(data_type, expected_types):
            continue
            
        records = _extract_records_from_element(data_type_elem, subject_id, year)
        
        if records:
            df = _create_dataframe_from_records(records)
            file_data[data_type] = df
            file_stats[data_type] = len(records)
    
    return file_data, {subject_id: file_stats}

def _should_skip_data_type(data_type: str, expected_types: list[str]) -> bool:
    """
    Determina si se debe omitir un tipo de dato.
    
    Parámetros:
    -----------
    data_type : str
        Tipo de dato a evaluar.
    expected_types : list[str]
        Tipos de datos esperados.
        
    Retorna:
    --------
    bool
        True si se debe omitir, False en caso contrario.
    """
    return data_type == 'patient' or data_type not in expected_types

def _extract_records_from_element(data_type_elem, subject_id: str, year: str) -> list[dict]:
    """
    Extrae registros de un elemento XML de tipo de dato.
    
    Parámetros:
    -----------
    data_type_elem : ET.Element
        Elemento XML del tipo de dato.
    subject_id : str
        ID del sujeto.
    year : str
        Año de los datos.
        
    Retorna:
    --------
    list[dict]
        Lista de registros extraídos.
    """
    records = []
    for event in data_type_elem:
        record_dict = dict(event.attrib)
        record_dict['SubjectID'] = subject_id
        record_dict['Year'] = year
        records.append(record_dict)
    return records

def _create_dataframe_from_records(records: list[dict]) -> pl.DataFrame:
    """
    Crea un DataFrame de Polars a partir de registros.
    
    Parámetros:
    -----------
    records : list[dict]
        Lista de registros.
        
    Retorna:
    --------
    pl.DataFrame
        DataFrame creado.
    """
    df = pl.DataFrame(records)
    if 'value' in df.columns:
        df = df.with_columns(pl.col('value').cast(pl.Float64))
    return df

def _consolidate_file_data(data_dict: dict[str, pl.DataFrame], file_data: dict[str, pl.DataFrame]) -> None:
    """
    Consolida los datos de un archivo en el diccionario principal.
    
    Parámetros:
    -----------
    data_dict : dict[str, pl.DataFrame]
        Diccionario principal de datos.
    file_data : dict[str, pl.DataFrame]
        Datos del archivo a consolidar.
    """
    for data_type, df in file_data.items():
        if data_type in data_dict:
            data_dict[data_type] = pl.concat([data_dict[data_type], df])
        else:
            data_dict[data_type] = df

def _validate_final_results(data_dict: dict[str, pl.DataFrame], subject_stats: dict, expected_subjects: list[str], year: str) -> None:
    """
    Valida los resultados finales del procesamiento.
    
    Parámetros:
    -----------
    data_dict : dict[str, pl.DataFrame]
        Diccionario con datos procesados.
    subject_stats : dict
        Estadísticas por sujeto.
    expected_subjects : list[str]
        Lista de sujetos esperados.
    year : str
        Año de los datos.
    """
    expected_types = _get_expected_data_types()
    missing_types = [t for t in expected_types if t not in data_dict]
    
    if missing_types:
        print_warning(f"Faltan tipos de datos: {missing_types}")
    
    if len(subject_stats) != len(expected_subjects):
        print_error(f"Se encontraron datos para {len(subject_stats)}/{len(expected_subjects)} sujetos")

def _process_bolus_legacy(data: pl.DataFrame) -> pl.DataFrame:
    """Process bolus data specifically."""
    bolus = data.clone()
    if "dose" in bolus.columns:
        bolus = bolus.rename({"dose": "bolus"})
        bolus = bolus.with_columns(pl.col("bolus").cast(pl.Float64))
    if "ts_begin" in bolus.columns:
        bolus = bolus.with_columns(
            pl.col("ts_begin").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
        )
    return bolus.filter(pl.col("bolus").is_not_null() & (pl.col("bolus") > 0))

def _process_meal_legacy(data: pl.DataFrame) -> pl.DataFrame:
    """Process meal data specifically."""
    meal = data.clone()
    if "carbs" in meal.columns:
        meal = meal.rename({"carbs": "meal_carbs"})
        meal = meal.with_columns(pl.col("meal_carbs").cast(pl.Float64))
    if "ts" in meal.columns:
        meal = meal.with_columns(
            pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
        )
    return meal.filter(pl.col("meal_carbs").is_not_null() & (pl.col("meal_carbs") > 0))

def _process_basal_legacy(data: pl.DataFrame) -> pl.DataFrame:
    """Process basal data specifically."""
    basal = data.clone()
    basal = basal.rename({"value": "basal_rate"}).with_columns(
        pl.col("basal_rate").cast(pl.Float64),
        pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
    )
    return basal.filter(pl.col("basal_rate").is_not_null())

def _process_temp_basal_legacy(data: pl.DataFrame) -> pl.DataFrame:
    """Process temp basal data specifically."""
    temp_basal = data.clone()
    temp_basal = temp_basal.rename({"value": "temp_basal_rate"}).with_columns(
        pl.col("temp_basal_rate").cast(pl.Float64),
        pl.col("ts_begin").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
    )
    return temp_basal.filter(pl.col("temp_basal_rate").is_not_null())

def _process_exercise_legacy(data: pl.DataFrame) -> pl.DataFrame:
    """Process exercise data specifically."""
    exercise = data.clone()
    exercise = exercise.with_columns(
        pl.col("intensity").cast(pl.Float64),
        pl.col("duration").cast(pl.Float64),
        pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
    )
    return exercise.filter(pl.col("intensity").is_not_null())

def _process_steps_legacy(data: pl.DataFrame) -> pl.DataFrame:
    """Process steps data specifically."""
    steps = data.clone()
    steps = steps.rename({"value": "steps"}).with_columns(
        pl.col("steps").cast(pl.Float64),
        pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
    )
    return steps.filter(pl.col("steps").is_not_null())

def _process_hypo_event_legacy(data: pl.DataFrame) -> pl.DataFrame:
    """Process hypo event data specifically."""
    hypo = data.clone()
    if "ts" in hypo.columns:
        hypo = hypo.with_columns(
            pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
        )
    return hypo

def _process_finger_stick_legacy(data: pl.DataFrame) -> pl.DataFrame:
    """Process finger stick data specifically."""
    finger_stick = data.clone()
    finger_stick = finger_stick.rename({"value": "finger_stick_bg"}).with_columns(
        pl.col("finger_stick_bg").cast(pl.Float64),
        pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
    )
    return finger_stick.filter(pl.col("finger_stick_bg").is_not_null())

def _process_sleep_work_legacy(data: pl.DataFrame, quality_col: str) -> pl.DataFrame:
    """Process sleep or work data with begin/end timestamps."""
    df = data.clone()
    if "ts_begin" in df.columns and "ts_end" in df.columns:
        df = df.with_columns(
            pl.col(quality_col).cast(pl.Float64),
            pl.col("ts_begin").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp_begin"),
            pl.col("ts_end").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp_end")
        )
    elif "ts" in df.columns:
        df = df.with_columns(
            pl.col(quality_col).cast(pl.Float64),
            pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
        )
    return df.filter(pl.col(quality_col).is_not_null())

def _process_simple_event_legacy(data: pl.DataFrame) -> pl.DataFrame:
    """Process simple event data with only timestamp."""
    df = data.clone()
    if "ts" in df.columns:
        df = df.with_columns(
            pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
        )
    return df

def _process_sensor_data_legacy(data: pl.DataFrame, value_col: str) -> pl.DataFrame:
    """Process sensor data with value column."""
    df = data.clone()
    df = df.rename({"value": value_col}).with_columns(
        pl.col(value_col).cast(pl.Float64),
        pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
    )
    return df.filter(pl.col(value_col).is_not_null())

def preprocess_bolus_meal(data: Dict[str, pl.DataFrame]) -> Dict[str, pl.DataFrame]:
    """
    Renombra y convierte columnas clave de bolus, meal, basal, temp_basal, exercise, basis_steps, hypo_event,
    finger_stick, sleep, work, stressors, illness, basis_heart_rate, basis_gsr, basis_skin_temperature,
    basis_air_temperature, basis_sleep, acceleration.
    """
    processed = {}
    
    # Define processing mappings
    simple_processors = {
        "bolus": _process_bolus_legacy,
        "meal": _process_meal_legacy,
        "basal": _process_basal_legacy,
        "temp_basal": _process_temp_basal_legacy,
        "exercise": _process_exercise_legacy,
        "basis_steps": _process_steps_legacy,
        "hypo_event": _process_hypo_event_legacy,
        "finger_stick": _process_finger_stick_legacy,
    }
    
    sleep_work_processors = {
        "sleep": "quality",
        "work": "intensity",
        "basis_sleep": "quality",
    }
    
    simple_events = ["stressors", "illness"]
    
    sensor_processors = {
        "basis_heart_rate": "heart_rate",
        "basis_gsr": "gsr",
        "basis_skin_temperature": "skin_temperature",
        "basis_air_temperature": "air_temperature",
        "acceleration": "acceleration",
    }
    
    # Process using simple processors
    for data_type, processor in simple_processors.items():
        if data_type in data:
            result = processor(data[data_type])
            processed[data_type] = result
            print_info(f"Eventos {data_type} válidos: {result.height}")
    
    # Process sleep/work data
    for data_type, quality_col in sleep_work_processors.items():
        if data_type in data:
            result = _process_sleep_work_legacy(data[data_type], quality_col)
            processed[data_type] = result
            print_info(f"Eventos {data_type} válidos: {result.height}")
    
    # Process simple events
    for data_type in simple_events:
        if data_type in data:
            result = _process_simple_event_legacy(data[data_type])
            processed[data_type] = result
            print_info(f"Eventos {data_type} válidos: {result.height}")
    
    # Process sensor data
    for data_type, value_col in sensor_processors.items():
        if data_type in data:
            result = _process_sensor_data_legacy(data[data_type], value_col)
            processed[data_type] = result
            print_info(f"Eventos {data_type} válidos: {result.height}")
    
    return processed

def extract_numeric_id(subject_id: str) -> int:
    """
    Extrae el ID numérico de una cadena de identificación de sujeto.
    
    Parámetros:
    -----------
    subject_id : str
        Cadena de identificación del sujeto (ej: '559-ws-training')
        
    Retorna:
    --------
    int
        ID numérico extraído
    """
    # Usar regex para extraer la parte numérica
    match = re.search(r'^(\d+)', subject_id)
    if match:
        return int(match.group(1))
    else:
        # Si no hay número, usar un valor predeterminado o lanzar un error
        print_warning(f"No se pudo extraer ID numérico de: {subject_id}")
        return 9999  # Un valor que no colisione con IDs reales

def _validate_critical_columns_for_extraction(df: pl.DataFrame) -> None:
    """Valida que existan las columnas críticas necesarias para extracción."""
    if 'value' not in df.columns:
        print_error("Columna 'value' no encontrada en el DataFrame de entrada")
        raise ValueError("Columna 'value' no encontrada en el DataFrame de entrada")

def _preserve_critical_columns(df: pl.DataFrame) -> Dict[str, pl.Series]:
    """Preserva columnas críticas para restaurar después."""
    critical_columns = ['value', 'bolus', 'SubjectID', 'Timestamp']
    return {col: df[col] for col in critical_columns if col in df.columns}

def _get_feature_groups() -> Dict[str, List[str]]:
    """Define y retorna los grupos de características."""
    return {
        'cgm': [
            'glucose_last', 'glucose_mean', 'glucose_std', 'glucose_min',
            'glucose_max', 'glucose_range', 'glucose_slope'
        ],
        'physiological': [
            'heart_rate', 'gsr', 'skin_temperature', 'air_temperature',
            'acceleration'
        ],
        'events': [
            'sleep_event', 'work_event', 'stressors_event',
            'illness_event', 'basis_sleep_event'
        ],
        'meal_context': [
            'meal_carbs', 'meal_time_diff_minutes', 'meal_time_diff_hours',
            'has_meal', 'meals_in_window', 'significant_meal',
            'total_carbs_window', 'largest_meal_carbs', 'meal_timing_score'
        ]
    }

def _check_available_features(df: pl.DataFrame, feature_groups: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Verifica qué características están disponibles en el DataFrame."""
    available_features = {}
    for group, features in feature_groups.items():
        available_features[group] = [f for f in features if f in df.columns]
        print_info(f"{group}: {len(available_features[group])}/{len(features)} características presentes")
    return available_features

def _extract_cgm_features(df: pl.DataFrame, extended_cgm_df: Optional[pl.DataFrame]) -> pl.DataFrame:
    """Extrae características CGM del DataFrame."""
    if 'value' not in df.columns:
        return df
    
    # Asegurarse de que 'value' es una columna numérica, no una lista
    if df['value'].dtype == pl.List:
        df = df.with_columns(pl.col('value').list.first().alias('value'))
    
    # Calcular características CGM básicas
    df = df.with_columns([
        pl.col('value').alias('glucose_last'),
        pl.col('value').rolling_mean(window_size=5).alias('glucose_mean'),
        pl.col('value').rolling_std(window_size=5).alias('glucose_std'),
        pl.col('value').rolling_min(window_size=5).alias('glucose_min'),
        pl.col('value').rolling_max(window_size=5).alias('glucose_max'),
        (pl.col('value').rolling_max(window_size=5) - 
         pl.col('value').rolling_min(window_size=5)).alias('glucose_range'),
        pl.col('value').diff().alias('glucose_slope')
    ])
    
    # Calcular patrones de 24h
    df = _add_glucose_patterns_24h(df, extended_cgm_df)
    
    return df

def _add_glucose_patterns_24h(df: pl.DataFrame, extended_cgm_df: Optional[pl.DataFrame]) -> pl.DataFrame:
    """Añade patrones de glucosa de 24 horas al DataFrame."""
    if extended_cgm_df is not None and not extended_cgm_df.is_empty():
        patterns_24h = compute_glucose_patterns_24h(
            extended_cgm_df.get_column('value').to_list()
        )
    else:
        patterns_24h = compute_glucose_patterns_24h(
            df.get_column('value').to_list()
        )
        print_info("Usando datos actuales para calcular patrones de 24h")
    
    for key, value in patterns_24h.items():
        df = df.with_columns(pl.lit(value).alias(key))
    
    return df

def _extract_physiological_features(df: pl.DataFrame, available_features: Dict[str, List[str]]) -> pl.DataFrame:
    """Extrae características fisiológicas del DataFrame."""
    for signal in available_features.get('physiological', []):
        if signal in df.columns:
            # Asegurarse de que la señal es una columna numérica, no una lista
            if df[signal].dtype == pl.List:
                df = df.with_columns(pl.col(signal).list.first().alias(signal))
            
            # Calcular estadísticas de la señal
            df = df.with_columns([
                pl.col(signal).rolling_mean(window_size=5).alias(f'{signal}_mean'),
                pl.col(signal).rolling_std(window_size=5).alias(f'{signal}_std'),
                pl.col(signal).rolling_min(window_size=5).alias(f'{signal}_min'),
                pl.col(signal).rolling_max(window_size=5).alias(f'{signal}_max')
            ])
    
    return df

def _extract_event_features(df: pl.DataFrame, available_features: Dict[str, List[str]]) -> pl.DataFrame:
    """Extrae características de eventos del DataFrame."""
    for event in available_features.get('events', []):
        if event in df.columns:
            # Asegurarse de que el evento es una columna numérica, no una lista
            if df[event].dtype == pl.List:
                df = df.with_columns(pl.col(event).list.first().alias(event))
            
            # Calcular estadísticas de eventos
            df = df.with_columns([
                pl.col(event).rolling_sum(window_size=5).alias(f'{event}_count'),
                pl.col(event).rolling_mean(window_size=5).alias(f'{event}_density')
            ])
    
    return df

def _extract_meal_context_features(df: pl.DataFrame, meal_df: Optional[pl.DataFrame]) -> pl.DataFrame:
    """Extrae características de contexto de comidas."""
    if meal_df is not None and not meal_df.is_empty() and 'Timestamp' in df.columns:
        df = _process_meal_data_with_context(df, meal_df)
    else:
        df = _add_default_meal_features(df)
    
    return df

def _process_meal_data_with_context(df: pl.DataFrame, meal_df: pl.DataFrame) -> pl.DataFrame:
    """Procesa datos de comidas con contexto temporal."""
    # Calcular tiempo desde última comida
    df = df.with_columns([
        pl.col('Timestamp').diff().dt.total_minutes().alias('time_since_last_meal')
    ])
    
    # Calcular características de comidas en ventana
    window_hours = 2.0
    for row in df.iter_rows(named=True):
        bolus_time = row['Timestamp']
        meal_context = compute_enhanced_meal_context(
            bolus_time, meal_df, window_hours=window_hours
        )
        
        # Actualizar características de comidas
        for key, value in meal_context.items():
            if key in df.columns:
                df = df.with_columns(pl.lit(value).alias(key))
    
    return df

def _add_default_meal_features(df: pl.DataFrame) -> pl.DataFrame:
    """Añade características de comidas por defecto."""
    default_meal_features = {
        'time_since_last_meal': 0.0,
        'meal_carbs': 0.0,
        'meal_time_diff_minutes': 0.0,
        'meal_time_diff_hours': 0.0,
        'has_meal': 0.0,
        'meals_in_window': 0,
        'significant_meal': 0.0,
        'total_carbs_window': 0.0,
        'largest_meal_carbs': 0.0,
        'meal_timing_score': 0.0
    }
    
    for key, value in default_meal_features.items():
        if key not in df.columns:
            df = df.with_columns(pl.lit(value).alias(key))
    
    return df

def _add_risk_indicators(df: pl.DataFrame, available_features: Dict[str, List[str]]) -> pl.DataFrame:
    """Añade indicadores de riesgo clínico."""
    if 'value' not in df.columns:
        return df
    
    glucose_values = df.get_column('value').to_list()
    physiological_data = {
        signal: df.get_column(signal).to_list()
        for signal in available_features.get('physiological', [])
        if signal in df.columns
    }
    time_values = df.get_column('Timestamp').to_list() if 'Timestamp' in df.columns else None
    
    risk_indicators = compute_clinical_risk_indicators(
        glucose_values, physiological_data, time_values
    )
    
    # Actualizar indicadores de riesgo
    for key, value in risk_indicators.items():
        df = df.with_columns(pl.lit(value).alias(key))
    
    return df

def _add_time_features(df: pl.DataFrame) -> pl.DataFrame:
    """Añade características de tiempo cíclicas."""
    if 'Timestamp' not in df.columns:
        return df
    
    time_features = []
    for ts in df.get_column('Timestamp'):
        time_features.append(encode_time_cyclical(ts))
    
    # Convertir a DataFrame y unir
    time_df = pl.DataFrame(time_features)
    df = df.hstack(time_df)
    
    return df

def _restore_critical_columns(df: pl.DataFrame, preserved_columns: Dict[str, pl.Series]) -> pl.DataFrame:
    """Restaura columnas críticas preservadas."""
    for col, values in preserved_columns.items():
        if col not in df.columns:
            df = df.with_columns(values.alias(col))
    return df

def _validate_final_features(df: pl.DataFrame, feature_groups: Dict[str, List[str]]) -> None:
    """Valida las características finales generadas."""
    # Verificar características generadas
    for group, features in feature_groups.items():
        present = [f for f in features if f in df.columns]
        print_info(f"{group} generadas: {len(present)}/{len(features)}")
        if len(present) < len(features):
            missing = set(features) - set(present)
            print_warning(f"Faltan características de {group}: {missing}")
    
    # Verificar columnas críticas al final
    critical_columns = ['value', 'time_in_range_24h', 'bolus']
    missing_critical = [col for col in critical_columns if col not in df.columns]
    if missing_critical:
        print_error(f"Faltan columnas críticas al final: {missing_critical}")
        raise ValueError(f"Faltan columnas críticas al final: {missing_critical}")

def extract_enhanced_features(df: pl.DataFrame, meal_df: Optional[pl.DataFrame] = None,
                            extended_cgm_df: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    """
    Extrae características mejoradas del DataFrame, incluyendo características CGM,
    fisiológicas y de eventos.
    
    Parámetros:
    -----------
    df : pl.DataFrame
        DataFrame con datos unidos
    meal_df : Optional[pl.DataFrame], opcional
        DataFrame con datos de comidas (default: None)
    extended_cgm_df : Optional[pl.DataFrame], opcional
        DataFrame con datos CGM extendidos (default: None)
        
    Retorna:
    --------
    pl.DataFrame
        DataFrame con características extraídas
    """
    # Validaciones iniciales
    _validate_critical_columns_for_extraction(df)
    preserved_columns = _preserve_critical_columns(df)
    
    # Configuración de características
    feature_groups = _get_feature_groups()
    available_features = _check_available_features(df, feature_groups)
    
    # Extracción de características por tipo
    df = _extract_cgm_features(df, extended_cgm_df)
    df = _extract_physiological_features(df, available_features)
    df = _extract_event_features(df, available_features)
    df = _extract_meal_context_features(df, meal_df)
    df = _add_risk_indicators(df, available_features)
    df = _add_time_features(df)
    
    # Finalización
    df = df.fill_null(0)
    df = _restore_critical_columns(df, preserved_columns)
    _validate_final_features(df, feature_groups)
    
    return df

def _apply_log_transformations(df: pl.DataFrame) -> pl.DataFrame:
    """Aplica transformaciones logarítmicas a características sesgadas."""
    log_transform_cols = [
        "bolus", "carb_input", "meal_carbs", "insulin_on_board",
        "total_carbs_window", "largest_meal_carbs"
    ]
    
    for col in log_transform_cols:
        if col in df.columns:
            df = df.with_columns(
                pl.col(col).log1p().alias(f"{col}_log1p")
            )
    return df

def _apply_percentage_normalizations(df: pl.DataFrame) -> pl.DataFrame:
    """Normaliza características porcentuales de 0-100 a 0-1."""
    percentage_cols = [
        "hypo_percentage_24h", "hyper_percentage_24h", "time_in_range_24h", "cv_24h"
    ]
    
    for col in percentage_cols:
        if col in df.columns:
            df = df.with_columns(
                (pl.col(col) / 100.0).alias(f"{col}_normalized")
            )
    return df

def _apply_time_normalizations(df: pl.DataFrame) -> pl.DataFrame:
    """Normaliza características de tiempo."""
    if "meal_time_diff_hours" in df.columns:
        df = df.with_columns(
            (pl.col("meal_time_diff_hours") / 24.0).alias("meal_time_diff_normalized")
        )
    return df

def _apply_glucose_normalizations(df: pl.DataFrame) -> pl.DataFrame:
    """Normaliza características relacionadas con glucosa para estabilidad."""
    glucose_norm_cols = [
        ("cgm_mean_24h", 200.0),
        ("cgm_std_24h", 100.0),
        ("cgm_median_24h", 200.0),
        ("cgm_range_24h", 300.0),
        ("mage_24h", 50.0),
        ("glucose_trend_24h", 10.0)
    ]
    
    for col, norm_factor in glucose_norm_cols:
        if col in df.columns:
            df = df.with_columns(
                (pl.col(col) / norm_factor).alias(f"{col}_normalized")
            )
    return df

def _expand_cgm_window(df: pl.DataFrame) -> pl.DataFrame:
    """Expande la ventana CGM a columnas individuales."""
    if "cgm_window" not in df.columns:
        return df
        
    window_size = CONFIG_PROCESSING["window_steps"]
    
    for i in range(window_size):
        df = df.with_columns(
            pl.col("cgm_window").list.get(i, null_on_oob=True)
            .fill_null(120.0)
            .alias(f"cgm_{i}")
        )

    return df.drop("cgm_window")

def _create_risk_composite_scores(df: pl.DataFrame) -> pl.DataFrame:
    """Crea puntuaciones compuestas de riesgo."""
    hypo_risk_cols = ["current_hypo_risk", "stability_score", "iob_risk_factor"]
    if all(col in df.columns for col in hypo_risk_cols):
        df = df.with_columns(
            (pl.col("current_hypo_risk") + pl.col("iob_risk_factor") * 0.5).alias("composite_hypo_risk")
        )
    
    hyper_risk_cols = ["current_hyper_risk", "glucose_rate_of_change"]
    if all(col in df.columns for col in hyper_risk_cols):
        df = df.with_columns(
            (pl.col("current_hyper_risk") + (pl.col("glucose_rate_of_change") / 10.0).clip(0, 1)).alias("composite_hyper_risk")
        )
    
    return df

def _add_compatibility_features(df: pl.DataFrame) -> pl.DataFrame:
    """Añade características derivadas para compatibilidad del modelo."""
    compatibility_features = {
        "hour_of_day": "hour_of_day_normalized",
        "has_meal_binary": "has_meal",
        "significant_meal_binary": "significant_meal"
    }
    
    for new_col, source_col in compatibility_features.items():
        if source_col in df.columns and new_col not in df.columns:
            df = df.with_columns(pl.col(source_col).alias(new_col))
    
    return df

def transform_enhanced_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Aplica transformaciones mejoradas incluyendo transformaciones logarítmicas, 
    normalización y expansión de características para un espacio de observación 
    de 52 dimensiones.
    
    Parámetros:
    -----------
    df : pl.DataFrame
        DataFrame con características sin transformar
        
    Retorna:
    --------
    pl.DataFrame
        DataFrame con características transformadas
    """
    print_info("Aplicando transformaciones mejoradas...")
    
    # Aplicar todas las transformaciones usando funciones auxiliares
    df = _apply_log_transformations(df)
    df = _apply_percentage_normalizations(df)
    df = _apply_time_normalizations(df)
    df = _apply_glucose_normalizations(df)
    df = _expand_cgm_window(df)
    df = _create_risk_composite_scores(df)
    df = _add_compatibility_features(df)
    
    print_info(f"Transformaciones mejoradas completadas. Forma final: {df.shape}")
    
    return df

def _extract_cgm_statistics(cgm_window: list) -> dict:
    """Extrae estadísticas básicas de la ventana CGM."""
    if not cgm_window:
        return {
            "glucose_last": 120.0,
            "glucose_mean": 120.0,
            "glucose_std": 0.0,
            "glucose_min": 120.0,
            "glucose_max": 120.0,
            "glucose_range": 0.0,
            "glucose_slope": 0.0,
        }
    
    return {
        "glucose_last": cgm_window[-1],
        "glucose_mean": float(np.mean(cgm_window)),
        "glucose_std": float(np.std(cgm_window)),
        "glucose_min": float(np.min(cgm_window)),
        "glucose_max": float(np.max(cgm_window)),
        "glucose_range": float(np.max(cgm_window) - np.min(cgm_window)),
        "glucose_slope": float(cgm_window[-1] - cgm_window[0]) / len(cgm_window) if len(cgm_window) > 1 else 0.0,
    }

def _create_xml_compatibility_features(cgm_stats: dict, row: dict) -> dict:
    """Crea características para compatibilidad con formato XML."""
    return {
        "value": cgm_stats["glucose_last"],
        "bwz_carb_input": row.get("carb_input", 0.0),
        "SubjectID": row.get("subject_id", 0),
    }

def _process_cgm_window_excel(df: pl.DataFrame) -> pl.DataFrame:
    """Procesa la ventana CGM y extrae características mejoradas."""
    if "cgm_window" not in df.columns:
        return df
    
    enhanced_rows = []
    
    for row in df.iter_rows(named=True):
        cgm_window = row.get("cgm_window", [])
        if not cgm_window:
            continue
        
        # Extraer estadísticas CGM
        cgm_stats = _extract_cgm_statistics(cgm_window)
        
        # Características de tiempo
        timestamp = row.get("Timestamp")
        time_features = encode_time_cyclical(timestamp) if timestamp else {}
        
        # Características de compatibilidad XML
        xml_compat = _create_xml_compatibility_features(cgm_stats, row)
        
        # Características de riesgo
        risk_indicators = compute_clinical_risk_indicators(
            cgm_window, 
            physiological_data=None,
            time_values=None
        )
        
        # Patrones de glucosa de 24h
        glucose_patterns = compute_glucose_patterns_24h(cgm_window)
        
        # Combinar todas las características
        enhanced_row = {
            **row,
            **cgm_stats,
            **time_features,
            **xml_compat,
            **risk_indicators,
            **glucose_patterns
        }
        
        enhanced_rows.append(enhanced_row)
    
    return pl.DataFrame(enhanced_rows) if enhanced_rows else df

def _transform_numeric_columns_excel(df: pl.DataFrame) -> pl.DataFrame:
    """Transforma columnas numéricas asegurando tipos correctos."""
    numeric_cols = ["bolus", "carb_input", "insulin_on_board", "insulin_carb_ratio", "insulin_sensitivity_factor"]
    
    for col in numeric_cols:
        if col not in df.columns:
            continue
        
        # Asegurar que es numérica y reemplazar valores extremos
        df = df.with_columns(
            pl.when(pl.col(col).cast(pl.Float64) < 0)
            .then(0.0)
            .otherwise(pl.col(col).cast(pl.Float64))
            .alias(col)
        )
    
    return df

def _add_meal_derived_features_excel(df: pl.DataFrame) -> pl.DataFrame:
    """Agrega características derivadas para comidas."""
    if "carb_input" not in df.columns:
        return df
    
    return df.with_columns([
        (pl.col("carb_input") > 0).cast(pl.Float64).alias("has_meal"),
        (pl.col("carb_input") > 15).cast(pl.Float64).alias("significant_meal")
    ])

def _add_missing_critical_column(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Agrega una columna crítica faltante con valor por defecto."""
    if col == "value" and "glucose_last" in df.columns:
        return df.with_columns(pl.col("glucose_last").alias("value"))
    elif col == "SubjectID" and "subject_id" in df.columns:
        return df.with_columns(pl.col("subject_id").alias("SubjectID"))
    elif col == "Timestamp" and "ts" in df.columns:
        return df.with_columns(pl.col("ts").alias(col))
    elif col == "bolus":
        return df.with_columns(pl.lit(0.0).alias(col))
    elif col == "value":
        return df.with_columns(pl.lit(120.0).alias(col))
    else:
        return df.with_columns(pl.lit(None).alias(col))

def _ensure_critical_columns_excel(df: pl.DataFrame) -> pl.DataFrame:
    """Verifica y agrega columnas críticas para compatibilidad con XML."""
    critical_columns = ["value", "bolus", "SubjectID", "Timestamp"]
    missing_critical = [col for col in critical_columns if col not in df.columns]
    
    if not missing_critical:
        return df
    
    print_warning(f"Faltan columnas críticas: {missing_critical}")
    
    for col in missing_critical:
        df = _add_missing_critical_column(df, col)
    
    return df

def extract_enhanced_features_excel(df: pl.DataFrame) -> pl.DataFrame:
    """
    Extrae características mejoradas específicamente para datos Excel.
    
    Parámetros:
    -----------
    df : pl.DataFrame
        DataFrame con datos Excel
        
    Retorna:
    --------
    pl.DataFrame
        DataFrame con características extraídas
    """
    print_info("Extrayendo características mejoradas para datos Excel...")
    
    # Procesar ventana CGM y extraer características
    df = _process_cgm_window_excel(df)
    
    # Transformar columnas numéricas
    df = _transform_numeric_columns_excel(df)
    
    # Agregar características derivadas para comidas
    df = _add_meal_derived_features_excel(df)
    
    # Verificar y agregar columnas críticas
    df = _ensure_critical_columns_excel(df)
    
    print_info(f"Extracción de características Excel completada. Forma: {df.shape}")
    return df

def _apply_log_transforms_excel(df: pl.DataFrame) -> pl.DataFrame:
    """Aplica transformaciones logarítmicas a características sesgadas."""
    log_transform_cols = [
        "bolus", "carb_input", "insulin_on_board", "insulin_carb_ratio"
    ]
    
    for col in log_transform_cols:
        if col in df.columns:
            df = df.with_columns(
                pl.col(col).log1p().alias(f"{col}_log1p")
            )
    return df

def _apply_percentage_normalizations_excel(df: pl.DataFrame) -> pl.DataFrame:
    """Normaliza características porcentuales."""
    if "cgm_window" not in df.columns:
        return df
        
    percentage_cols = [
        "hypo_percentage_24h", "hyper_percentage_24h", "time_in_range_24h", "cv_24h"
    ]
    
    for col in percentage_cols:
        if col in df.columns:
            df = df.with_columns(
                (pl.col(col) / 100.0).alias(f"{col}_normalized")
            )
    return df

def _expand_cgm_window_excel(df: pl.DataFrame) -> pl.DataFrame:
    """Expande la ventana CGM a columnas individuales."""
    if "cgm_window" not in df.columns:
        return df
        
    window_size = min(CONFIG_PROCESSING["window_steps"], 24)
    
    for i in range(window_size):
        df = df.with_columns(
            pl.col("cgm_window").list.get(i, null_on_oob=True)
            .fill_null(120.0)
            .alias(f"cgm_{i}")
        )
    
    return df.drop("cgm_window")

def _create_cyclical_features_excel(df: pl.DataFrame) -> pl.DataFrame:
    """Crea características cíclicas para hora del día."""
    if "hour_of_day" not in df.columns or "hour_sin" in df.columns:
        return df
        
    hour_radians = 2 * np.pi * (df["hour_of_day"] / 24.0)
    df = df.with_columns([
        pl.Series(name="hour_sin", values=np.sin(hour_radians.to_numpy())),
        pl.Series(name="hour_cos", values=np.cos(hour_radians.to_numpy()))
    ])
    return df

def _apply_compatibility_mappings_excel(df: pl.DataFrame) -> pl.DataFrame:
    """Aplica mapeos de compatibilidad con formato XML."""
    compatibility_mappings = {
        "carb_input": "bwz_carb_input",
        "bg_input": "glucose_last",
        "hour_of_day": "hour_of_day_normalized",
        "has_meal": "has_meal_binary",
        "significant_meal": "significant_meal_binary"
    }
    
    for excel_col, xml_col in compatibility_mappings.items():
        if excel_col in df.columns and xml_col not in df.columns:
            df = df.with_columns(pl.col(excel_col).alias(xml_col))
    return df

def transform_enhanced_features_excel(df: pl.DataFrame) -> pl.DataFrame:
    """
    Aplica transformaciones a características de datos Excel para hacerlas compatibles
    con el formato esperado por los modelos.
    
    Parámetros:
    -----------
    df : pl.DataFrame
        DataFrame con características Excel extraídas
        
    Retorna:
    --------
    pl.DataFrame
        DataFrame con características transformadas
    """
    print_info("Transformando características de datos Excel...")
    
    # Aplicar transformaciones usando funciones auxiliares
    df = _apply_log_transforms_excel(df)
    df = _apply_percentage_normalizations_excel(df)
    df = _expand_cgm_window_excel(df)
    df = _create_cyclical_features_excel(df)
    df = _apply_compatibility_mappings_excel(df)
    
    print_info(f"Transformación de características Excel completada. Forma: {df.shape}")
    return df

def unify_datetime_precision(df: pl.DataFrame) -> pl.DataFrame:
    """
    Unifica la precisión de todas las columnas datetime a microsegundos.
    
    Parámetros:
    -----------
    df : pl.DataFrame
        DataFrame con columnas datetime que pueden tener diferentes precisiones
        
    Retorna:
    --------
    pl.DataFrame
        DataFrame con todas las columnas datetime convertidas a microsegundos
    """
    datetime_cols = [col for col in df.columns if df[col].dtype.base_type() == pl.Datetime]
    
    if not datetime_cols:
        return df
    
    exprs = []
    for col in datetime_cols:
        # Convertir explícitamente a microsegundos
        exprs.append(pl.col(col).cast(pl.Datetime(time_unit="us")).alias(col))
    
    if exprs:
        df = df.with_columns(exprs)
    
    return df

def _determine_preferred_type(current_type: pl.DataType, new_type: pl.DataType) -> pl.DataType:
    """
    Determina el tipo preferido entre dos tipos de datos.
    
    Parámetros:
    -----------
    current_type : pl.DataType
        Tipo actual
    new_type : pl.DataType
        Nuevo tipo a considerar
        
    Retorna:
    --------
    pl.DataType
        Tipo preferido
    """
    current_str = str(current_type)
    new_str = str(new_type)
    
    # Preferir tipos numéricos sobre string
    if current_str == "Utf8" and (new_str == "Float64" or new_str == "Int32"):
        return new_type
    
    # Preferir Float64 sobre Int32 para columnas numéricas
    if current_str == "Int32" and new_str == "Float64":
        return new_type
    
    # Para columnas de fecha, usar siempre microsegundos
    if "Datetime" in current_str and "Datetime" in new_str:
        return pl.Datetime(time_unit="us")
    
    return current_type

def _collect_column_types(data_frames: list[pl.DataFrame]) -> dict[str, pl.DataType]:
    """
    Recopila los tipos de columnas de todos los DataFrames.
    
    Parámetros:
    -----------
    data_frames : list[pl.DataFrame]
        Lista de DataFrames
        
    Retorna:
    --------
    dict[str, pl.DataType]
        Diccionario con tipos preferidos por columna
    """
    column_types = {}
    
    for df in data_frames:
        for col in df.columns:
            col_type = df[col].dtype
            
            # Saltar si la columna es de tipo Null
            if str(col_type) == "Null":
                continue
            
            if col not in column_types:
                column_types[col] = col_type
            else:
                column_types[col] = _determine_preferred_type(column_types[col], col_type)
    
    return column_types

def _create_fallback_expression(col: str, target_type: pl.DataType) -> pl.Expr:
    """
    Crea una expresión de respaldo cuando falla la conversión normal.
    
    Parámetros:
    -----------
    col : str
        Nombre de la columna
    target_type : pl.DataType
        Tipo objetivo
        
    Retorna:
    --------
    pl.Expr
        Expresión de polars para la conversión de respaldo
    """
    target_str = str(target_type)
    
    if target_str == "Utf8":
        return pl.col(col).cast(pl.Utf8).alias(col)
    elif target_str == "Float64":
        return pl.lit(None).cast(pl.Float64).alias(col)
    elif target_str == "Int32":
        return pl.lit(None).cast(pl.Int32).alias(col)
    elif "Datetime" in target_str:
        return pl.lit(None).cast(pl.Datetime(time_unit="us")).alias(col)
    else:
        return pl.col(col).alias(col)

def _convert_dataframe_columns(df: pl.DataFrame, column_types: dict[str, pl.DataType]) -> pl.DataFrame:
    """
    Convierte las columnas de un DataFrame a los tipos especificados.
    
    Parámetros:
    -----------
    df : pl.DataFrame
        DataFrame a convertir
    column_types : dict[str, pl.DataType]
        Diccionario con tipos objetivo por columna
        
    Retorna:
    --------
    pl.DataFrame
        DataFrame con columnas convertidas
    """
    exprs = []
    
    for col in df.columns:
        if col not in column_types:
            continue
            
        target_type = column_types[col]
        
        # Solo convertir si el tipo actual es diferente
        if str(df[col].dtype) != str(target_type):
            try:
                exprs.append(pl.col(col).cast(target_type).alias(col))
            except Exception as e:
                print_warning(f"Error al convertir columna {col}: {e}")
                exprs.append(_create_fallback_expression(col, target_type))
    
    # Aplicar conversiones si es necesario
    if exprs:
        df = df.with_columns(exprs)
    
    return df

def unify_column_types(data_frames: list[pl.DataFrame]) -> list[pl.DataFrame]:
    """
    Unifica los tipos de datos entre DataFrames para evitar conflictos en la concatenación.
    
    Parámetros:
    -----------
    data_frames : list[pl.DataFrame]
        Lista de DataFrames a unificar
        
    Retorna:
    --------
    list[pl.DataFrame]
        Lista de DataFrames con tipos de datos consistentes
    """
    if not data_frames:
        return data_frames
    
    print_info("Unificando tipos de columnas para concatenación...")
    
    # Determinar el tipo más apropiado para cada columna
    column_types = _collect_column_types(data_frames)
    
    # Convertir todos los DataFrames para usar tipos consistentes
    converted_frames = []
    for df in data_frames:
        converted_df = _convert_dataframe_columns(df, column_types)
        converted_frames.append(converted_df)
    
    return converted_frames

def _process_xml_data() -> tuple[list[pl.DataFrame], set]:
    """Procesa datos XML y retorna DataFrames con columnas encontradas."""
    print_info(f"Iniciando procesamiento paralelo de {len(OHIO_DATA_DIRS)} directorios XML...")
    xml_data_frames_results = Parallel(n_jobs=-1)(
        delayed(process_xml_directory)(data_dir)
        for data_dir in OHIO_DATA_DIRS
    )

    xml_data_frames = [df for df in xml_data_frames_results if df is not None]
    all_columns = set()
    
    for df in xml_data_frames:
        for col in df.columns:
            all_columns.add(col)

    print_info(f"Procesamiento XML completado. Obtenidos {len(xml_data_frames)} DataFrames válidos.")
    return xml_data_frames, all_columns

def _normalize_subject_ids(xml_data_frames: list[pl.DataFrame]) -> list[pl.DataFrame]:
    """Normaliza SubjectIDs a valores numéricos."""
    for i, df in enumerate(xml_data_frames):
        if "SubjectID" in df.columns and df["SubjectID"].dtype != pl.Int64:
            xml_data_frames[i] = df.with_columns(
                pl.col("SubjectID").map_elements(
                    lambda x: extract_numeric_id(str(x)) if x is not None else None
                ).cast(pl.Int64)
            )
    return xml_data_frames

def _process_excel_data() -> tuple[list[pl.DataFrame], set]:
    """Procesa datos Excel y retorna DataFrames con columnas encontradas."""
    excel_data_frames = []
    all_columns = set()
    
    if not USE_EXCEL_DATA:
        return excel_data_frames, all_columns
        
    subject_files = [f for f in os.listdir(DATA_PATH_SUBJECTS) 
                    if f.startswith("Subject") and f.endswith(".xlsx")]
    
    print_info(f"\nArchivos de sujetos encontrados ({len(subject_files)}):")
    for f in subject_files:
        print_info(f)

    excel_data = Parallel(n_jobs=-1)(
        delayed(process_excel_subject)(os.path.join(DATA_PATH_SUBJECTS, f), idx)
        for idx, f in enumerate(subject_files)
    )
    
    excel_data = [item for sublist in excel_data for item in sublist if item is not None]

    if excel_data:
        df_excel = pl.DataFrame(excel_data)
        df_excel = _apply_column_mappings(df_excel)
        df_excel = extract_enhanced_features_excel(df_excel)
        df_excel = transform_enhanced_features_excel(df_excel)
        
        for col in df_excel.columns:
            all_columns.add(col)
        
        excel_data_frames.append(df_excel)
        print_info(f"DataFrame Excel procesado: {df_excel.shape}")
    else:
        print_warning("No se pudieron procesar datos Excel")
    
    return excel_data_frames, all_columns

def _apply_column_mappings(df_excel: pl.DataFrame) -> pl.DataFrame:
    """Aplica mapeos de columnas entre Excel y XML."""
    column_mappings = {
        "subject_id": "SubjectID",
        "cgm_window": "cgm_window",
        "carb_input": "bwz_carb_input",
        "bg_input": "glucose_last",
        "bolus": "bolus"
    }
    
    for excel_col, xml_col in column_mappings.items():
        if excel_col in df_excel.columns and xml_col not in df_excel.columns:
            df_excel = df_excel.rename({excel_col: xml_col})
    
    return df_excel

def _get_column_type_for_missing(col: str) -> pl.DataType:
    """Determina el tipo de dato apropiado para una columna faltante."""
    if col in ['meals_in_window', 'hypo_episodes_24h', 'hyper_episodes_24h']:
        return pl.Int32
    elif any(prefix in col for prefix in ['glucose_', 'time_', 'meal_', 'risk_', 'percentage']):
        return pl.Float64
    else:
        return pl.Utf8

def _harmonize_dataframes(all_data_frames: list[pl.DataFrame], all_columns: set) -> list[pl.DataFrame]:
    """Armoniza DataFrames para que tengan las mismas columnas y tipos."""
    print_info(f"Armonizando {len(all_data_frames)} DataFrames con {len(all_columns)} columnas...")

    column_types = {}
    for df in all_data_frames:
        for col in df.columns:
            column_types[col] = df[col].dtype

    for i, df in enumerate(all_data_frames):
        missing_cols = all_columns - set(df.columns)
        
        if missing_cols:
            print_info(f"Añadiendo {len(missing_cols)} columnas faltantes al DataFrame {i+1}")
            for col in missing_cols:
                if col in column_types:
                    df = df.with_columns(pl.lit(None).cast(column_types[col]).alias(col))
                else:
                    col_type = _get_column_type_for_missing(col)
                    df = df.with_columns(pl.lit(None).cast(col_type).alias(col))
        
        all_data_frames[i] = df
    
    return all_data_frames

def _finalize_dataframes(all_data_frames: list[pl.DataFrame], all_columns: set) -> list[pl.DataFrame]:
    """Finaliza DataFrames con orden de columnas y tipos unificados."""
    ordered_columns = sorted(all_columns)
    
    for i, df in enumerate(all_data_frames):
        df = df.select(ordered_columns)
        df = unify_datetime_precision(df)
        all_data_frames[i] = df

    return unify_column_types(all_data_frames)

def preprocess_data() -> pl.DataFrame:
    """
    Preprocesa los datos utilizando preferentemente las funciones de pl_ohio_only.py
    para datos XML, y uniendo con Excel si es necesario.
    
    Retorna:
    --------
    pl.DataFrame
        DataFrame con datos preprocesados.
    """
    print_info("Procesando datos priorizando XML sobre Excel...")
    
    # 1. Procesar datos XML
    xml_data_frames, xml_columns = _process_xml_data()
    xml_data_frames = _normalize_subject_ids(xml_data_frames)
    
    # 2. Procesar datos Excel
    excel_data_frames, excel_columns = _process_excel_data()
    
    # 3. Combinar y unificar
    all_data_frames = xml_data_frames + excel_data_frames
    all_columns = xml_columns | excel_columns
    
    if not all_data_frames:
        raise ValueError("No se pudieron procesar datos de ninguna fuente")
    
    print_info(f"Total de DataFrames procesados: {len(all_data_frames)}")
    
    # 4. Armonizar y finalizar DataFrames
    all_data_frames = _harmonize_dataframes(all_data_frames, all_columns)
    all_data_frames = _finalize_dataframes(all_data_frames, all_columns)
    
    # 5. Concatenar y retornar
    print_info(f"Concatenando {len(all_data_frames)} DataFrames...")
    final_df = pl.concat(all_data_frames)
    
    print_info(f"Procesamiento completado. Forma final: {final_df.shape}")
    return final_df

def calculate_stats_for_group(df: pl.DataFrame, subjects: list, feature: str = 'bolus') -> tuple:
    """
    Calcula media y desviación estándar para un grupo de sujetos usando Polars.

    Parámetros:
    -----------
    df : pl.DataFrame
        DataFrame con datos procesados
    subjects : list
        Lista de IDs de sujetos
    feature : str, opcional
        Característica para calcular estadísticas (default: 'bolus')

    Retorna:
    --------
    tuple
        Tupla con (media, desviación estándar)
    """
    if not subjects:
        return 0.0, 0.0
    filtered_df = df.filter(pl.col('subject_id').is_in(subjects))
    mean_val = filtered_df[feature].mean()
    std_val = filtered_df[feature].std()
    return mean_val if mean_val is not None else 0.0, std_val if std_val is not None else 0.0

def calculate_distribution_score(means: list, stds: list) -> float:
    """
    Calcula una puntuación de distribución basada en medias y desviaciones estándar.

    Parámetros:
    -----------
    means : list
        Lista de valores medios
    stds : list
        Lista de desviaciones estándar

    Retorna:
    --------
    float
        Puntuación que representa la variabilidad de la distribución
    """
    if not all(m != 0 for m in means):
        return float('inf')
    
    range_means = max(means) - min(means)
    range_stds = max(stds) - min(stds) if all(s != 0 for s in stds) else float('inf')
    return range_means + range_stds

def assign_subject_to_group(df: pl.DataFrame, subject: int, 
                           train_subjects: list, val_subjects: list, test_subjects: list,
                           train_size: int, val_size: int, test_size: int) -> tuple:
    """
    Asigna un sujeto a un grupo de entrenamiento, validación o prueba basado en balance.

    Parámetros:
    -----------
    df : pl.DataFrame
        DataFrame con datos procesados
    subject : int
        ID del sujeto a asignar
    train_subjects : list
        Lista actual de sujetos de entrenamiento
    val_subjects : list
        Lista actual de sujetos de validación
    test_subjects : list
        Lista actual de sujetos de prueba
    train_size : int
        Tamaño máximo del grupo de entrenamiento
    val_size : int
        Tamaño máximo del grupo de validación
    test_size : int
        Tamaño máximo del grupo de prueba

    Retorna:
    --------
    tuple
        Tupla con listas actualizadas (train_subjects, val_subjects, test_subjects)
    """
    # Calculate current stats
    train_mean, train_std = calculate_stats_for_group(df, train_subjects)
    val_mean, val_std = calculate_stats_for_group(df, val_subjects)
    test_mean, test_std = calculate_stats_for_group(df, test_subjects)
    
    # Calculate potential stats if subject is added to each group
    train_temp = train_subjects + [subject]
    val_temp = val_subjects + [subject]
    test_temp = test_subjects + [subject]
    
    train_mean_new, train_std_new = calculate_stats_for_group(df, train_temp)
    val_mean_new, val_std_new = calculate_stats_for_group(df, val_temp)
    test_mean_new, test_std_new = calculate_stats_for_group(df, test_temp)
    
    # Calculate scores for each option
    score_if_train = calculate_distribution_score(
        [train_mean_new, val_mean, test_mean], 
        [train_std_new, val_std, test_std]
    )
    score_if_val = calculate_distribution_score(
        [train_mean, val_mean_new, test_mean], 
        [train_std, val_std_new, test_std]
    )
    score_if_test = calculate_distribution_score(
        [train_mean, val_mean, test_mean_new], 
        [train_std, val_std, test_std_new]
    )
    
    # Assign to the group with best balance
    if len(train_subjects) < train_size and score_if_train <= min(score_if_val, score_if_test):
        train_subjects.append(subject)
    elif len(val_subjects) < val_size and score_if_val <= min(score_if_train, score_if_test):
        val_subjects.append(subject)
    elif len(test_subjects) < test_size:
        test_subjects.append(subject)
    else:
        train_subjects.append(subject)
    
    return train_subjects, val_subjects, test_subjects

def standardize_columns(df: pl.DataFrame, columns: list, mean_std_dict: dict = None) -> tuple:
    """
    Estandariza columnas específicas de un DataFrame usando Polars.

    Parámetros:
    -----------
    df : pl.DataFrame
        DataFrame con datos a estandarizar
    columns : list
        Lista de columnas a estandarizar
    mean_std_dict : dict, opcional
        Diccionario con medias y desviaciones estándar precalculadas (default: None)

    Retorna:
    --------
    tuple
        (DataFrame estandarizado, diccionario con medias y desviaciones estándar)
    """
    if mean_std_dict is None:
        mean_std_dict = {}
        # Calcular media y desviación estándar para cada columna
        for col in columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            mean_std_dict[col] = (mean_val if mean_val is not None else 0.0, 
                                 std_val if std_val is not None else 1.0)
    
    # Estandarizar columnas
    exprs = []
    for col in columns:
        mean_val, std_val = mean_std_dict[col]
        # Evitar división por cero
        std_val = 1.0 if std_val == 0 else std_val
        expr = ((pl.col(col) - mean_val) / std_val).alias(col)
        exprs.append(expr)
    
    df_standardized = df.with_columns(exprs)
    return df_standardized, mean_std_dict

def split_data(df_final: pl.DataFrame) -> tuple:
    """
    Divide los datos siguiendo una estrategia para asegurar distribuciones 
    equilibradas entre los conjuntos de entrenamiento, validación y prueba, usando Polars.

    Parámetros:
    -----------
    df_final : pl.DataFrame
        DataFrame con todos los datos preprocesados

    Retorna:
    --------
    tuple
        Tupla con múltiples elementos:
        - x_cgm_train, x_cgm_val, x_cgm_test: datos CGM para cada conjunto
        - x_other_train, x_other_val, x_other_test: otras características para cada conjunto
        - x_subject_train, x_subject_val, x_subject_test: IDs de sujetos para cada conjunto
        - y_train, y_val, y_test: etiquetas para cada conjunto
        - subject_test: IDs de sujetos de prueba
        - mean_std_cgm, mean_std_other, mean_std_y: diccionarios con medias y desviaciones estándar
    """
    start_time = time.time()
    print_info("Iniciando división de datos...")
    
    # Estadísticas por sujeto
    subject_stats = df_final.group_by("subject_id").agg([
        pl.col("bolus").mean().alias("mean_dose"),
        pl.col("bolus").std().alias("std_dose")
    ])
    
    # Obtener lista de sujetos ordenados por dosis media
    sorted_subjects = subject_stats.sort("mean_dose").get_column("subject_id").to_list()
    n_subjects = len(sorted_subjects)
    train_size = int(0.8 * n_subjects)
    val_size = int(0.1 * n_subjects)
    test_size = n_subjects - train_size - val_size
    print_info(f"Total de sujetos: {n_subjects}, Train: {train_size}, Val: {val_size}, Test: {test_size}")

    # Iniciar con sujeto específico para pruebas si está disponible
    test_subjects = [49] if 49 in sorted_subjects else []
    remaining_subjects = [s for s in sorted_subjects if s != 49]
    train_subjects = []
    val_subjects = []

    # Aleatorizar la lista restante
    rng = np.random.default_rng(seed=CONST_DEFAULT_SEED)
    rng.shuffle(remaining_subjects)
    print_info("Sujetos aleatorizados para asignación.")

    # Distribuir sujetos entre los grupos
    for subject in tqdm(remaining_subjects, desc="Asignando sujetos a grupos"):
        train_subjects, val_subjects, test_subjects = assign_subject_to_group(
            df_final, subject, train_subjects, val_subjects, test_subjects,
            train_size, val_size, test_size
        )

    # Dividir el DataFrame en conjuntos
    df_train = df_final.filter(pl.col('subject_id').is_in(train_subjects))
    df_val = df_final.filter(pl.col('subject_id').is_in(val_subjects))
    df_test = df_final.filter(pl.col('subject_id').is_in(test_subjects))

    # Mostrar estadísticas post-división
    for set_name, df_set in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        y_mean = df_set['bolus'].mean()
        y_std = df_set['bolus'].std()
        print_info(f"Post-split {set_name} y: mean = {y_mean}, std = {y_std}")

    # Definir columnas para diferentes grupos de características
    cgm_columns = [f'cgm_{i}' for i in range(24)]
    other_features = ['carb_input', 'bg_input', 'insulin_on_board', 'insulin_carb_ratio', 
                      'insulin_sensitivity_factor', 'hour_of_day']

    # Estandarizar datos CGM
    df_train_cgm, mean_std_cgm = standardize_columns(df_train, cgm_columns)
    df_val_cgm, _ = standardize_columns(df_val, cgm_columns, mean_std_cgm)
    df_test_cgm, _ = standardize_columns(df_test, cgm_columns, mean_std_cgm)

    # Convertir a NumPy y reshape
    x_cgm_train = df_train_cgm.select(cgm_columns).to_numpy().reshape(-1, 24, 1)
    x_cgm_val = df_val_cgm.select(cgm_columns).to_numpy().reshape(-1, 24, 1)
    x_cgm_test = df_test_cgm.select(cgm_columns).to_numpy().reshape(-1, 24, 1)

    # Estandarizar otras características
    df_train_other, mean_std_other = standardize_columns(df_train, other_features)
    df_val_other, _ = standardize_columns(df_val, other_features, mean_std_other)
    df_test_other, _ = standardize_columns(df_test, other_features, mean_std_other)

    # Convertir a NumPy
    x_other_train = df_train_other.select(other_features).to_numpy()
    x_other_val = df_val_other.select(other_features).to_numpy()
    x_other_test = df_test_other.select(other_features).to_numpy()

    # Estandarizar etiquetas (bolus)
    df_train_y, mean_std_y = standardize_columns(df_train, ['bolus'])
    df_val_y, _ = standardize_columns(df_val, ['bolus'], mean_std_y)
    df_test_y, _ = standardize_columns(df_test, ['bolus'], mean_std_y)

    # Convertir etiquetas a NumPy y flatten
    y_train = df_train_y['bolus'].to_numpy()
    y_val = df_val_y['bolus'].to_numpy()
    y_test = df_test_y['bolus'].to_numpy()

    # Obtener IDs de sujeto
    x_subject_train = df_train['subject_id'].to_numpy()
    x_subject_val = df_val['subject_id'].to_numpy()
    x_subject_test = df_test['subject_id'].to_numpy()
    
    # Imprimir resumen
    print_info(f"Entrenamiento CGM: {x_cgm_train.shape}, Validación CGM: {x_cgm_val.shape}, Prueba CGM: {x_cgm_test.shape}")
    print_info(f"Entrenamiento Otros: {x_other_train.shape}, Validación Otros: {x_other_val.shape}, Prueba Otros: {x_other_test.shape}")
    print_info(f"Entrenamiento Subject: {x_subject_train.shape}, Validación Subject: {x_subject_val.shape}, Prueba Subject: {x_subject_test.shape}")
    print_info(f"Sujetos de prueba: {test_subjects}")

    elapsed_time = time.time() - start_time
    print_info(f"División de datos completa en {elapsed_time:.2f} segundos")
    
    return (x_cgm_train, x_cgm_val, x_cgm_test,
            x_other_train, x_other_val, x_other_test,
            x_subject_train, x_subject_val, x_subject_test,
            y_train, y_val, y_test, test_subjects,
            mean_std_cgm, mean_std_other, mean_std_y)