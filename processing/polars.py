import polars as pl
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import sys
import os
from joblib import Parallel, delayed
from datetime import timedelta

# Configuración de Matplotlib para evitar errores con Tkinter
import matplotlib
matplotlib.use('Agg')

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

# %% [markdown]
# ## Constantes

# %%
# Definición de la ruta del proyecto
# PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
SUBJECTS_RELATIVE_PATH = "data/Subjects"
SUBJECTS_PATH = os.path.join(PROJECT_ROOT, SUBJECTS_RELATIVE_PATH)

subject_files = [f for f in os.listdir(SUBJECTS_PATH) if f.startswith("Subject") and f.endswith(".xlsx")]
print(f"Total sujetos: {len(subject_files)}")

# %% [markdown]
# ## Preprocesamiento y Procesamiento de Datos

# %%
def get_cgm_window(bolus_time, cgm_df: pl.DataFrame, window_hours: int = 2) -> np.ndarray:
    """
    Obtiene la ventana de datos CGM para un tiempo de bolo específico.

    Parámetros:
    -----------
    bolus_time : datetime
        Tiempo del bolo de insulina
    cgm_df : pl.DataFrame
        DataFrame con datos CGM
    window_hours : int, opcional
        Horas de la ventana de datos (default: 2)

    Retorna:
    --------
    np.ndarray
        Ventana de datos CGM o None si no hay suficientes datos
    """
    window_start = bolus_time - timedelta(hours=window_hours)
    window = cgm_df.filter(
        (pl.col("date") >= window_start) & (pl.col("date") <= bolus_time)
    ).sort("date").tail(24)
    
    if window.height < 24:
        return None
    return window.get_column("mg/dl").to_numpy()

def calculate_iob(bolus_time, basal_df: pl.DataFrame, half_life_hours: float = 4.0) -> float:
    """
    Calcula la insulina activa en el cuerpo (IOB).

    Parámetros:
    -----------
    bolus_time : datetime
        Tiempo del bolo de insulina
    basal_df : pl.DataFrame
        DataFrame con datos de insulina basal
    half_life_hours : float, opcional
        Vida media de la insulina en horas (default: 4.0)

    Retorna:
    --------
    float
        Cantidad de insulina activa
    """
    if basal_df is None or basal_df.is_empty():
        return 0.0
    
    iob = 0.0
    for row in basal_df.iter_rows(named=True):
        start_time = row["date"]
        duration_hours = row["duration"] / (1000 * 3600)
        end_time = start_time + timedelta(hours=duration_hours)
        rate = row["rate"] if row["rate"] is not None else 0.9
        
        if start_time <= bolus_time <= end_time:
            time_since_start = (bolus_time - start_time).total_seconds() / 3600
            remaining = rate * (1 - (time_since_start / half_life_hours))
            iob += max(0.0, remaining)
    return iob

def load_subject_data(subject_path: str) -> tuple:
    """
    Carga los datos de un sujeto desde el archivo Excel.
    
    Parámetros:
    -----------
    subject_path : str
        Ruta al archivo del sujeto
        
    Retorna:
    --------
    tuple
        (cgm_df, bolus_df, basal_df) o (None, None, None) si hay error
    """
    try:
        cgm_df = pl.read_excel(subject_path, sheet_name="CGM")
        bolus_df = pl.read_excel(subject_path, sheet_name="Bolus")
        try:
            basal_df = pl.read_excel(subject_path, sheet_name="Basal")
        except Exception:
            basal_df = None
        
        # Conversión de fechas
        cgm_df = cgm_df.with_columns(pl.col("date").cast(pl.Datetime))
        bolus_df = bolus_df.with_columns(pl.col("date").cast(pl.Datetime))
        if basal_df is not None:
            basal_df = basal_df.with_columns(pl.col("date").cast(pl.Datetime))
        
        cgm_df = cgm_df.sort("date")
        return cgm_df, bolus_df, basal_df
    except Exception as e:
        print(f"Error al cargar {os.path.basename(subject_path)}: {e}")
        return None, None, None

def extract_features(row, bolus_time, cgm_window, basal_df, idx) -> dict:
    """
    Extrae las características de un registro de bolo.
    
    Parámetros:
    -----------
    row : dict
        Fila del DataFrame de bolus
    bolus_time : datetime
        Tiempo del bolo
    cgm_window : np.ndarray
        Ventana de datos CGM
    basal_df : pl.DataFrame
        DataFrame de datos basales
    idx : int
        Índice del sujeto
        
    Retorna:
    --------
    dict
        Diccionario con características extraídas
    """
    iob = calculate_iob(bolus_time, basal_df)
    hour_of_day = bolus_time.hour / 23.0
    bg_input = row["bgInput"] if row["bgInput"] is not None else cgm_window[-1]
    normal = row["normal"] if row["normal"] is not None else 0.0
    
    # Cálculo del factor de sensibilidad personalizado
    isf_custom = 50.0
    if normal > 0 and bg_input > 100:
        isf_custom = (bg_input - 100) / normal
    
    return {
        'subject_id': idx,
        'cgm_window': cgm_window,
        'carbInput': row["carbInput"] if row["carbInput"] is not None else 0.0,
        'bgInput': bg_input,
        'insulinCarbRatio': row["insulinCarbRatio"] if row["insulinCarbRatio"] is not None else 10.0,
        'insulinSensitivityFactor': isf_custom,
        'insulinOnBoard': iob,
        'hour_of_day': hour_of_day,
        'normal': normal
    }

def process_subject(subject_path: str, idx: int) -> list:
    """
    Procesa los datos de un sujeto.

    Parámetros:
    -----------
    subject_path : str
        Ruta al archivo del sujeto
    idx : int
        Índice del sujeto

    Retorna:
    --------
    list
        Lista de diccionarios con características procesadas
    """
    print(f"Procesando {os.path.basename(subject_path)} ({idx+1}/{len(subject_files)})...")
    
    cgm_df, bolus_df, basal_df = load_subject_data(subject_path)
    if cgm_df is None:
        return []

    processed_data = []
    for row in bolus_df.iter_rows(named=True):
        bolus_time = row["date"]
        cgm_window = get_cgm_window(bolus_time, cgm_df)
        
        if cgm_window is not None:
            features = extract_features(row, bolus_time, cgm_window, basal_df, idx)
            processed_data.append(features)
    
    return processed_data

# Ejecución en paralelo
all_processed_data = Parallel(n_jobs=-1)(
    delayed(process_subject)(
        os.path.join(SUBJECTS_PATH, f), 
        idx
    ) for idx, f in enumerate(subject_files)
)

all_processed_data = [item for sublist in all_processed_data for item in sublist]

# Conversión a DataFrame
df_processed = pl.DataFrame(all_processed_data)
print("Muestra de datos procesados combinados:")
print(df_processed.head())
print(f"Total muestras: {len(df_processed)}")

# %% [markdown]
# ### División de Ventana CGM y Valores Nulos

# %%
# Dividir ventana CGM y otras características
cgm_columns = [f'cgm_{i}' for i in range(24)]
df_cgm = pl.DataFrame({
    col: [row['cgm_window'][i] for row in all_processed_data]
    for i, col in enumerate(cgm_columns)
}, schema={col: pl.Float64 for col in cgm_columns})

# Combinar con otras características
df_processed = pl.concat([
    df_cgm,
    df_processed.drop('cgm_window')
], how="horizontal")

# Verificar valores nulos
print("Verificación de valores nulos en df_processed:")
print(df_processed.null_count())
df_processed = df_processed.drop_nulls()

# %% [markdown]
# ### Normalización de Datos

# %%
# Normalizar características
scaler_cgm = MinMaxScaler(feature_range=(0, 1))
scaler_other = StandardScaler()

# Normalizar CGM
x_cgm = scaler_cgm.fit_transform(df_processed.select(cgm_columns).to_numpy())
x_cgm = x_cgm.reshape(x_cgm.shape[0], x_cgm.shape[1], 1)

# Normalizar otras características (incluyendo hour_of_day)
other_features = ['carbInput', 'bgInput', 'insulinOnBoard', 'insulinCarbRatio', 
                  'insulinSensitivityFactor', 'subject_id', 'hour_of_day']
x_other = scaler_other.fit_transform(df_processed.select(other_features).to_numpy())

# Etiquetas
y = df_processed.get_column('normal').to_numpy()

# Verificar NaN
print("NaN en x_cgm:", np.isnan(x_cgm).sum())
print("NaN en x_other:", np.isnan(x_other).sum())
print("NaN en y:", np.isnan(y).sum())
if np.isnan(x_cgm).sum() > 0 or np.isnan(x_other).sum() > 0 or np.isnan(y).sum() > 0:
    raise ValueError("Valores NaN detectados en x_cgm, x_other o y")

# %% [markdown]
# ### División por Sujeto de los Datos

# %%
# División por sujeto
subject_ids = df_processed.get_column('subject_id').unique().to_numpy()
train_subjects, temp_subjects = train_test_split(subject_ids, test_size=0.2, random_state=42)
val_subjects, test_subjects = train_test_split(temp_subjects, test_size=0.5, random_state=42)

# %% [markdown]
# ### Creación de Máscaras

# %%
# Crear máscaras
train_mask = df_processed.get_column('subject_id').is_in(train_subjects).to_numpy()
val_mask = df_processed.get_column('subject_id').is_in(val_subjects).to_numpy()
test_mask = df_processed.get_column('subject_id').is_in(test_subjects).to_numpy()

x_cgm_train, x_cgm_val, x_cgm_test = x_cgm[train_mask], x_cgm[val_mask], x_cgm[test_mask]
x_other_train, x_other_val, x_other_test = x_other[train_mask], x_other[val_mask], x_other[test_mask]
y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]
subject_test = df_processed.filter(pl.col('subject_id').is_in(test_subjects)).get_column('subject_id').to_numpy()

print(f"Entrenamiento CGM: {x_cgm_train.shape}, Validación CGM: {x_cgm_val.shape}, Prueba CGM: {x_cgm_test.shape}")
print(f"Entrenamiento Otros: {x_other_train.shape}, Validación Otros: {x_other_val.shape}, Prueba Otros: {x_other_test.shape}")
print(f"Sujetos de prueba: {test_subjects}")