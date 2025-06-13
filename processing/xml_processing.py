import xml.etree.ElementTree as ET
import polars as pl
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import os
import glob
# import d3rlpy
import matplotlib.pyplot as plt

# Constantes
BASE_DATA_DIR = "data/OhioT1DM"
TRAIN_2018_DIR = os.path.join(BASE_DATA_DIR, "2018/train")
TEST_2018_DIR = os.path.join(BASE_DATA_DIR, "2018/test")
TRAIN_2020_DIR = os.path.join(BASE_DATA_DIR, "2020/train")
TEST_2020_DIR = os.path.join(BASE_DATA_DIR, "2020/test")
FIGURE_DIR = "output/figures"
TIMESTAMP_COL = "timestamp"
BG_COL = "bg"
BASAL_COL = "basal"
BOLUS_COL = "bolus"
CARBS_COL = "carbs"
TIR_MIN = 70  # mg/dL
TIR_MAX = 180  # mg/dL
TBR_THRESHOLD = 70  # mg/dL
TBR_LEVEL2 = 54  # mg/dL
TAR_LEVEL1_MAX = 250  # mg/dL
EMERGENCY_MIN = 40  # mg/dL
EMERGENCY_MAX = 450  # mg/dL
WINDOW_HOURS = 1  # Horas para la ventana de CGM
TRAIN_WEEKS = 4.8  # Semanas para entrenamiento
VAL_WEEKS = 1.2  # Semanas para validación

# Inicialización del generador de números aleatorios
rng = np.random.default_rng(seed=42)

def parse_xml_file(file_path: str) -> pl.DataFrame:
    """
    Parsea un archivo XML de OhioT1DM y retorna un DataFrame.

    Parámetros:
    -----------
    file_path : str
        Ruta al archivo XML.

    Retorna:
    --------
    pl.DataFrame
        DataFrame con columnas: timestamp, bg, basal, bolus, carbs.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    data = []
    
    # Mapeo de nombres de columnas de timestamp posibles
    timestamp_candidates = ['ts', 'ts_begin']
    
    for entry in root.findall('.//entry'):
        # Buscar el timestamp disponible (ts o ts_begin)
        timestamp = None
        for ts_col in timestamp_candidates:
            ts_element = entry.find(ts_col)
            if ts_element is not None:
                timestamp = ts_element.text
                break
        
        if timestamp is None:
            continue  # Saltar entradas sin timestamp válido
        
        bg = float(entry.find('glucose').text) if entry.find('glucose') is not None else None
        basal = float(entry.find('basal').text) if entry.find('basal') is not None else None
        bolus = float(entry.find('bolus').text) if entry.find('bolus') is not None else None
        carbs = float(entry.find('carbs').text) if entry.find('carbs') is not None else None
        data.append({TIMESTAMP_COL: timestamp, BG_COL: bg, BASAL_COL: basal, BOLUS_COL: bolus, CARBS_COL: carbs})
    
    if not data:
        return pl.DataFrame()
    
    df = pl.DataFrame(data)
    df = df.with_columns(pl.col(TIMESTAMP_COL).cast(pl.Datetime))
    return df.sort(TIMESTAMP_COL)

def find_subject_files(subject_id: str) -> Tuple[List[str], List[str]]:
    """
    Encuentra los archivos de entrenamiento y prueba para un sujeto en los directorios 2018 y 2020.

    Parámetros:
    -----------
    subject_id : str
        Identificador del sujeto.

    Retorna:
    --------
    Tuple[List[str], List[str]]
        Lista de archivos de entrenamiento y prueba.
    """
    train_pattern_2018 = os.path.join(TRAIN_2018_DIR, f"*{subject_id}*.xml")
    train_pattern_2020 = os.path.join(TRAIN_2020_DIR, f"*{subject_id}*.xml")
    test_pattern_2018 = os.path.join(TEST_2018_DIR, f"*{subject_id}*.xml")
    test_pattern_2020 = os.path.join(TEST_2020_DIR, f"*{subject_id}*.xml")
    
    train_files = glob.glob(train_pattern_2018) + glob.glob(train_pattern_2020)
    test_files = glob.glob(test_pattern_2018) + glob.glob(test_pattern_2020)
    
    return train_files, test_files

def merge_subject_data(file_paths: List[str]) -> pl.DataFrame:
    """
    Combina los datos de múltiples archivos XML en un solo DataFrame.

    Parámetros:
    -----------
    file_paths : List[str]
        Lista de rutas a archivos XML.

    Retorna:
    --------
    pl.DataFrame
        DataFrame combinado y ordenado por timestamp.
    """
    if not file_paths:
        return pl.DataFrame()
    
    dfs = [parse_xml_file(file_path) for file_path in file_paths]
    merged_df = pl.concat(dfs, how="vertical")
    return merged_df.sort(TIMESTAMP_COL).unique(subset=[TIMESTAMP_COL], keep="first")

def split_dataset(dev_df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Divide los datos de desarrollo en conjuntos de entrenamiento, validación y prueba.

    Parámetros:
    -----------
    dev_df : pl.DataFrame
        DataFrame con los datos de desarrollo.

    Retorna:
    --------
    Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
        Conjuntos de entrenamiento, validación y prueba.
    """
    start_time = dev_df[TIMESTAMP_COL].min()
    train_end = start_time + timedelta(weeks=TRAIN_WEEKS)
    val_end = start_time + timedelta(weeks=TRAIN_WEEKS + VAL_WEEKS)
    
    train_df = dev_df.filter(pl.col(TIMESTAMP_COL) < train_end)
    val_df = dev_df.filter((pl.col(TIMESTAMP_COL) >= train_end) & (pl.col(TIMESTAMP_COL) < val_end))
    test_df = dev_df.filter(pl.col(TIMESTAMP_COL) >= val_end)
    
    return train_df, val_df, test_df

def extract_cgm_features(window_data: pl.DataFrame) -> Dict[str, float]:
    """
    Extrae características de CGM para una ventana de datos.

    Parámetros:
    -----------
    window_data : pl.DataFrame
        DataFrame con datos CGM en la ventana temporal.

    Retorna:
    --------
    Dict[str, float]
        Diccionario con características: bg_current, bg_mean, bg_max, bg_min, bg_diff, hypo_pct, hyper_pct.
    """
    if window_data.is_empty():
        return {
            'bg_current': None, 'bg_mean': None, 'bg_max': None, 'bg_min': None,
            'bg_diff': None, 'hypo_pct': None, 'hyper_pct': None
        }
    
    bg_values = window_data[BG_COL].drop_nulls()
    bg_current = bg_values[-1] if len(bg_values) > 0 else None
    bg_mean = bg_values.mean()
    bg_max = bg_values.max()
    bg_min = bg_values.min()
    bg_diff = bg_values.diff().abs().max() if len(bg_values) > 1 else 0
    hypo_pct = (bg_values < TIR_MIN).mean() * 100 if len(bg_values) > 0 else 0
    hyper_pct = (bg_values > TIR_MAX).mean() * 100 if len(bg_values) > 0 else 0
    
    return {
        'bg_current': bg_current, 'bg_mean': bg_mean, 'bg_max': bg_max, 'bg_min': bg_min,
        'bg_diff': bg_diff, 'hypo_pct': hypo_pct, 'hyper_pct': hyper_pct
    }

def extract_event_features(df: pl.DataFrame, current_time: datetime) -> Dict[str, float]:
    """
    Extrae características de eventos (carbohidratos y bolo) previos al tiempo actual.

    Parámetros:
    -----------
    df : pl.DataFrame
        DataFrame con datos de eventos.
    current_time : datetime
        Tiempo actual para calcular características.

    Retorna:
    --------
    Dict[str, float]
        Diccionario con características: last_carbs_time, last_carbs_amount, last_bolus_time, last_bolus_amount.
    """
    past_data = df.filter(pl.col(TIMESTAMP_COL) <= current_time)
    last_carbs = past_data.filter(pl.col(CARBS_COL).is_not_null()).select([TIMESTAMP_COL, CARBS_COL]).tail(1)
    last_bolus = past_data.filter(pl.col(BOLUS_COL).is_not_null()).select([TIMESTAMP_COL, BOLUS_COL]).tail(1)
    
    carbs_time = (current_time - last_carbs[TIMESTAMP_COL][0]).total_seconds() / 3600 if not last_carbs.is_empty() else None
    carbs_amount = last_carbs[CARBS_COL][0] if not last_carbs.is_empty() else 0
    bolus_time = (current_time - last_bolus[TIMESTAMP_COL][0]).total_seconds() / 3600 if not last_bolus.is_empty() else None
    bolus_amount = last_bolus[BOLUS_COL][0] if not last_bolus.is_empty() else 0
    
    return {
        'last_carbs_time': carbs_time, 'last_carbs_amount': carbs_amount,
        'last_bolus_time': bolus_time, 'last_bolus_amount': bolus_amount
    }

def encode_timestamp(current_time: datetime) -> Dict[str, float]:
    """
    Codifica el tiempo en formato cíclico.

    Parámetros:
    -----------
    current_time : datetime
        Tiempo actual.

    Retorna:
    --------
    Dict[str, float]
        Diccionario con codificaciones cíclicas: sin_time, cos_time.
    """
    hour = current_time.hour
    sin_time = np.sin(2 * np.pi * hour / 24)
    cos_time = np.cos(2 * np.pi * hour / 24)
    return {'sin_time': sin_time, 'cos_time': cos_time}

def compute_reward(bg: float) -> float:
    """
    Calcula la recompensa basada en el nivel de glucosa en sangre.

    Parámetros:
    -----------
    bg : float
        Nivel de glucosa en sangre (mg/dL).

    Retorna:
    --------
    float
        Valor de la recompensa.
    """
    if bg < EMERGENCY_MIN or bg > EMERGENCY_MAX:
        return 0
    elif TIR_MIN <= bg <= TIR_MAX:
        return 1
    elif TBR_LEVEL2 <= bg < TIR_MIN:
        return -1
    elif bg < TBR_LEVEL2:
        return -2
    elif TIR_MAX < bg <= TAR_LEVEL1_MAX:
        return -1
    else:
        return -2

def compute_tbr_reward(bg: float) -> float:
    """
    Calcula la recompensa TBR para OPE.

    Parámetros:
    -----------
    bg : float
        Nivel de glucosa en sangre (mg/dL).

    Retorna:
    --------
    float
        Valor de la recompensa TBR.
    """
    return -1 if bg < TBR_THRESHOLD else 0

def extract_features_and_transitions(df: pl.DataFrame) -> pl.DataFrame:
    """
    Extrae características y transiciones para DRL, retornando un DataFrame.

    Parámetros:
    -----------
    df : pl.DataFrame
        DataFrame con datos procesados.

    Retorna:
    --------
    pl.DataFrame
        DataFrame con columnas: state, action, next_state, reward, terminal.
    """
    state_keys = [
        'bg_current', 'bg_mean', 'bg_max', 'bg_min', 'bg_diff',
        'hypo_pct', 'hyper_pct', 'last_carbs_time', 'last_carbs_amount',
        'last_bolus_time', 'last_bolus_amount', 'sin_time', 'cos_time'
    ]
    data = []
    for i in range(len(df) - 1):
        current_time = df[TIMESTAMP_COL][i]
        next_time = df[TIMESTAMP_COL][i + 1]
        
        # Extraer características para el estado actual
        window_data = df.filter((pl.col(TIMESTAMP_COL) >= current_time - timedelta(hours=WINDOW_HOURS)) &
                                (pl.col(TIMESTAMP_COL) <= current_time))
        cgm_features = extract_cgm_features(window_data)
        event_features = extract_event_features(df, current_time)
        time_features = encode_timestamp(current_time)
        state_t = [cgm_features.get(key, 0.0) for key in state_keys]
        
        # Extraer características para el estado siguiente
        next_window_data = df.filter((pl.col(TIMESTAMP_COL) >= next_time - timedelta(hours=WINDOW_HOURS)) &
                                     (pl.col(TIMESTAMP_COL) <= next_time))
        next_cgm_features = extract_cgm_features(next_window_data)
        next_event_features = extract_event_features(df, next_time)
        next_time_features = encode_timestamp(next_time)
        state_t1 = [next_cgm_features.get(key, 0.0) for key in state_keys]
        
        # Acción, recompensa y terminal
        action_t = df[BASAL_COL][i] if df[BASAL_COL][i] is not None else 0.0
        reward_t = compute_reward(df[BG_COL][i] if df[BG_COL][i] is not None else 0.0)
        terminal = 1 if reward_t == 0 else 0
        
        data.append({
            'state': state_t,
            'action': action_t,
            'next_state': state_t1,
            'reward': reward_t,
            'terminal': terminal
        })
    
    return pl.DataFrame(data)

# def create_mdp_dataset(transitions_df: pl.DataFrame) -> d3rlpy.dataset.MDPDataset:
#     """
#     Crea un MDPDataset de d3rlpy a partir de un DataFrame de transiciones.

#     Parámetros:
#     -----------
#     transitions_df : pl.DataFrame
#         DataFrame con columnas: state, action, next_state, reward, terminal.

#     Retorna:
#     --------
#     d3rlpy.dataset.MDPDataset
#         Dataset para entrenamiento en d3rlpy.
#     """
#     observations = np.array(transitions_df['state'].to_list(), dtype=np.float32)
#     actions = np.array(transitions_df['action'].to_list(), dtype=np.float32).reshape(-1, 1)
#     rewards = np.array(transitions_df['reward'].to_list(), dtype=np.float32)
#     next_observations = np.array(transitions_df['next_state'].to_list(), dtype=np.float32)
#     terminals = np.array(transitions_df['terminal'].to_list(), dtype=np.float32)
    
#     return d3rlpy.dataset.MDPDataset(
#         observations=observations,
#         actions=actions,
#         rewards=rewards,
#         next_observations=next_observations,
#         terminals=terminals
#     )

def plot_bg_trajectory(df: pl.DataFrame, subject_id: str) -> None:
    """
    Genera y guarda un gráfico de la trayectoria de glucosa.

    Parámetros:
    -----------
    df : pl.DataFrame
        DataFrame con datos de glucosa.
    subject_id : str
        Identificador del sujeto.

    Retorna:
    --------
    None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df[TIMESTAMP_COL], df[BG_COL], label='Glucosa (mg/dL)')
    plt.axhline(TIR_MIN, color='green', linestyle='--', label='TIR Min (70 mg/dL)')
    plt.axhline(TIR_MAX, color='red', linestyle='--', label='TIR Max (180 mg/dL)')
    plt.xlabel('Tiempo')
    plt.ylabel('Glucosa (mg/dL)')
    plt.title(f'Trayectoria de Glucosa - Sujeto {subject_id}')
    plt.legend()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURE_DIR, f'bg_trajectory_{subject_id}.png'))
    plt.close()

def process_subject_data(subject_id: str) -> Tuple[Optional[pl.DataFrame], Optional[pl.DataFrame], Optional[pl.DataFrame]]:
    """
    Procesa los datos de un sujeto, combinando datos de 2018 y 2020, y retorna DataFrames de transiciones.

    Parámetros:
    -----------
    subject_id : str
        Identificador del sujeto.

    Retorna:
    --------
    Tuple[Optional[pl.DataFrame], Optional[pl.DataFrame], Optional[pl.DataFrame]]
        DataFrames de transiciones para entrenamiento, validación y prueba, o None si no hay datos.
    """
    # Encontrar archivos
    train_files, test_files = find_subject_files(subject_id)
    if not train_files or not test_files:
        print(f"No se encontraron archivos para el sujeto {subject_id}")
        return None, None, None
    
    # Combinar datos
    dev_df = merge_subject_data(train_files)
    test_df = merge_subject_data(test_files)
    
    if dev_df.is_empty() or test_df.is_empty():
        print(f"Datos vacíos para el sujeto {subject_id}")
        return None, None, None
    
    # Generar gráfico de trayectoria de glucosa
    plot_bg_trajectory(dev_df, subject_id)
    
    # Dividir datos de desarrollo
    train_df, val_df, _ = split_dataset(dev_df)
    
    # Extraer transiciones
    train_transitions = extract_features_and_transitions(train_df)
    val_transitions = extract_features_and_transitions(val_df)
    test_transitions = extract_features_and_transitions(test_df)
    
    return train_transitions, val_transitions, test_transitions

def get_all_subjects() -> List[str]:
    """
    Obtiene todos los IDs de sujetos únicos desde los nombres de archivos.

    Retorna:
    --------
    List[str]
        Lista de IDs de sujetos ordenados.
    """
    subject_ids = set()
    for dir_path in [TRAIN_2018_DIR, TRAIN_2020_DIR, TEST_2018_DIR, TEST_2020_DIR]:
        for file_path in glob.glob(os.path.join(dir_path, "*.xml")):
            subject_id = os.path.basename(file_path).split("-")[0]
            subject_ids.add(subject_id)
    
    return sorted(subject_ids)