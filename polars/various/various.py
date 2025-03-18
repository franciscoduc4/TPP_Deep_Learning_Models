# %%
import polars as pl
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, LSTM, Input, Concatenate, BatchNormalization,
    Conv1D, MaxPooling1D, LayerNormalization, MultiHeadAttention,
    Add, GlobalAveragePooling1D
)
import matplotlib.pyplot as plt
import os
from joblib import Parallel, delayed
from datetime import timedelta

# Configuración de Matplotlib para evitar errores con Tkinter
import matplotlib
matplotlib.use('TkAgg')

from models.cnn import create_cnn_model
from models.transformer import create_transformer_model
from models.tcn import create_tcn_model
from models.gru import create_gru_model
from models.wavenet import create_wavenet_model
from models.tabnet import create_tabnet_model
from models.attention_only import create_attention_model
from models.rnn import create_rnn_model

# %%
# Definición de la ruta del proyecto
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SUBJECTS_RELATIVE_PATH = "data/Subjects"
SUBJECTS_PATH = os.path.join(PROJECT_ROOT, SUBJECTS_RELATIVE_PATH)

# Crear directorios para resultados
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures", "various_models")
os.makedirs(FIGURES_DIR, exist_ok=True)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

subject_files = [f for f in os.listdir(SUBJECTS_PATH) if f.startswith("Subject") and f.endswith(".xlsx")]
print(f"Total sujetos: {len(subject_files)}")

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
    
    try:
        cgm_df = pl.read_excel(subject_path, sheet_name="CGM")
        bolus_df = pl.read_excel(subject_path, sheet_name="Bolus")
        try:
            basal_df = pl.read_excel(subject_path, sheet_name="Basal")
        except Exception:
            basal_df = None
    except Exception as e:
        print(f"Error al cargar {os.path.basename(subject_path)}: {e}")
        return []

    # Conversión de fechas
    cgm_df = cgm_df.with_columns(pl.col("date").cast(pl.Datetime))
    bolus_df = bolus_df.with_columns(pl.col("date").cast(pl.Datetime))
    if basal_df is not None:
        basal_df = basal_df.with_columns(pl.col("date").cast(pl.Datetime))
    
    cgm_df = cgm_df.sort("date")

    processed_data = []
    for row in bolus_df.iter_rows(named=True):
        bolus_time = row["date"]
        cgm_window = get_cgm_window(bolus_time, cgm_df)
        
        if cgm_window is not None:
            iob = calculate_iob(bolus_time, basal_df)
            hour_of_day = bolus_time.hour / 23.0
            bg_input = row["bgInput"] if row["bgInput"] is not None else cgm_window[-1]
            normal = row["normal"] if row["normal"] is not None else 0.0
            
            # Cálculo del factor de sensibilidad personalizado
            isf_custom = 50.0
            if normal > 0 and bg_input > 100:
                isf_custom = (bg_input - 100) / normal
            
            features = {
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

# %%
# Normalizar características
scaler_cgm = MinMaxScaler(feature_range=(0, 1))
scaler_other = StandardScaler()

# Normalizar CGM
X_cgm = scaler_cgm.fit_transform(df_processed.select(cgm_columns).to_numpy())
X_cgm = X_cgm.reshape(X_cgm.shape[0], X_cgm.shape[1], 1)

# Normalizar otras características (incluyendo hour_of_day)
other_features = ['carbInput', 'bgInput', 'insulinOnBoard', 'insulinCarbRatio', 
                  'insulinSensitivityFactor', 'subject_id', 'hour_of_day']
X_other = scaler_other.fit_transform(df_processed.select(other_features).to_numpy())

# Etiquetas
y = df_processed.get_column('normal').to_numpy()

# Verificar NaN
print("NaN en X_cgm:", np.isnan(X_cgm).sum())
print("NaN en X_other:", np.isnan(X_other).sum())
print("NaN en y:", np.isnan(y).sum())
if np.isnan(X_cgm).sum() > 0 or np.isnan(X_other).sum() > 0 or np.isnan(y).sum() > 0:
    raise ValueError("Valores NaN detectados en X_cgm, X_other o y")

# %%
# División por sujeto
subject_ids = df_processed.get_column('subject_id').unique().to_numpy()
train_subjects, temp_subjects = train_test_split(subject_ids, test_size=0.2, random_state=42)
val_subjects, test_subjects = train_test_split(temp_subjects, test_size=0.5, random_state=42)

# %%
# Crear máscaras
train_mask = df_processed.get_column('subject_id').is_in(train_subjects).to_numpy()
val_mask = df_processed.get_column('subject_id').is_in(val_subjects).to_numpy()
test_mask = df_processed.get_column('subject_id').is_in(test_subjects).to_numpy()

X_cgm_train, X_cgm_val, X_cgm_test = X_cgm[train_mask], X_cgm[val_mask], X_cgm[test_mask]
X_other_train, X_other_val, X_other_test = X_other[train_mask], X_other[val_mask], X_other[test_mask]
y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]
subject_test = df_processed.filter(pl.col('subject_id').is_in(test_subjects)).get_column('subject_id').to_numpy()

print(f"Entrenamiento CGM: {X_cgm_train.shape}, Validación CGM: {X_cgm_val.shape}, Prueba CGM: {X_cgm_test.shape}")
print(f"Entrenamiento Otros: {X_other_train.shape}, Validación Otros: {X_other_val.shape}, Prueba Otros: {X_other_test.shape}")
print(f"Sujetos de prueba: {test_subjects}")

def train_and_evaluate_model(model: Model, model_name: str, 
                           X_cgm_train: np.ndarray, X_other_train: np.ndarray, 
                           y_train: np.ndarray, X_cgm_val: np.ndarray, 
                           X_other_val: np.ndarray, y_val: np.ndarray,
                           X_cgm_test: np.ndarray, X_other_test: np.ndarray, 
                           y_test: np.ndarray) -> tuple:
    """
    Entrena y evalúa un modelo específico.
    
    Parámetros:
    -----------
    model : Model
        Modelo a entrenar
    model_name : str
        Nombre del modelo para guardado/logging
    X_cgm_train, X_other_train, y_train : np.ndarray
        Datos de entrenamiento
    X_cgm_val, X_other_val, y_val : np.ndarray
        Datos de validación
    X_cgm_test, X_other_test, y_test : np.ndarray
        Datos de prueba
        
    Retorna:
    --------
    tuple
        (history, y_pred, metrics_dict)
    """
    # Compilar modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )
    
    # Entrenar modelo
    history = model.fit(
        [X_cgm_train, X_other_train],
        y_train,
        validation_data=([X_cgm_val, X_other_val], y_val),
        epochs=100,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ],
        verbose=1
    )
    
    # Predecir y evaluar
    y_pred = model.predict([X_cgm_test, X_other_test]).flatten()
    
    # Calcular métricas
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred)
    }
    
    # Guardar modelo
    model.save(os.path.join(MODELS_DIR, f'{model_name}.keras'))
    
    return history, y_pred, metrics

# %%
def plot_training_history(histories: dict, model_names: list):
    """
    Visualiza el historial de entrenamiento de múltiples modelos.
    
    Parámetros:
    -----------
    histories : dict
        Diccionario con historiales de entrenamiento por modelo
    model_names : list
        Lista de nombres de modelos
    """
    plt.figure(figsize=(12, 6))
    
    for name, history in histories.items():
        plt.plot(history.history['loss'], label=f'{name} (train)')
        plt.plot(history.history['val_loss'], label=f'{name} (val)', linestyle='--')
    
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida MSE')
    plt.title('Comparación de Historiales de Entrenamiento')
    plt.legend()
    plt.savefig(os.path.join(FIGURES_DIR, 'training_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_predictions_comparison(y_test: np.ndarray, predictions: dict):
    """
    Visualiza comparación de predicciones de múltiples modelos.
    
    Parámetros:
    -----------
    y_test : np.ndarray
        Valores reales de prueba
    predictions : dict
        Diccionario con predicciones por modelo
    """
    plt.figure(figsize=(15, 5))
    
    # Scatter plot
    plt.subplot(1, 2, 1)
    for name, y_pred in predictions.items():
        plt.scatter(y_test, y_pred, alpha=0.5, label=name)
    plt.plot([0, 15], [0, 15], 'r--')
    plt.xlabel('Dosis Real (u. de insulina)')
    plt.ylabel('Dosis Predicha (u. de insulina)')
    plt.legend()
    plt.title('Predicción vs. Real (Todos los Modelos)')
    
    # Residuals
    plt.subplot(1, 2, 2)
    for name, y_pred in predictions.items():
        plt.hist(y_test - y_pred, bins=20, alpha=0.5, label=name)
    plt.xlabel('Residuo (u. de insulina)')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.title('Distribución de Residuos')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'predictions_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

# %%
# Entrenamiento y evaluación de modelos
print("\nCreando y entrenando modelos...")

models = {
    'CNN': create_cnn_model(X_cgm_train.shape, X_other_train.shape),
    'Transformer': create_transformer_model(X_cgm_train.shape, X_other_train.shape),
    'TCN': create_tcn_model(X_cgm_train.shape, X_other_train.shape),
    'GRU': create_gru_model(X_cgm_train.shape, X_other_train.shape),
    'WaveNet': create_wavenet_model(X_cgm_train.shape, X_other_train.shape),
    'TabNet': create_tabnet_model(X_cgm_train.shape, X_other_train.shape),
    'Attention': create_attention_model(X_cgm_train.shape, X_other_train.shape),
    'RNN': create_rnn_model(X_cgm_train.shape, X_other_train.shape)
}

histories = {}
predictions = {}
metrics = {}

for name, model in models.items():
    print(f"\nEntrenando modelo {name}...")
    history, y_pred, model_metrics = train_and_evaluate_model(
        model, name,
        X_cgm_train, X_other_train, y_train,
        X_cgm_val, X_other_val, y_val,
        X_cgm_test, X_other_test, y_test
    )
    
    histories[name] = history
    predictions[name] = y_pred
    metrics[name] = model_metrics

# Evaluación por sujeto
print("\nRendimiento por sujeto:")
for subject_id in test_subjects:
    mask = subject_test == subject_id
    y_test_sub = y_test[mask]
    
    print(f"\nSujeto {subject_id}:")
    print("-" * 40)
    for name, y_pred in predictions.items():
        y_pred_sub = y_pred[mask]
        mae_sub = mean_absolute_error(y_test_sub, y_pred_sub)
        rmse_sub = np.sqrt(mean_squared_error(y_test_sub, y_pred_sub))
        r2_sub = r2_score(y_test_sub, y_pred_sub)
        print(f"{name:<15} MAE={mae_sub:.2f}, RMSE={rmse_sub:.2f}, R²={r2_sub:.2f}")

# %%
# Visualización de resultados
plot_training_history(histories, list(models.keys()))
plot_predictions_comparison(y_test, predictions)

# %%
# Imprimir métricas comparativas
print("\nComparación de métricas:")
print("-" * 50)
print(f"{'Modelo':<15} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
print("-" * 50)
for name, metric in metrics.items():
    print(f"{name:<15} {metric['mae']:8.2f} {metric['rmse']:8.2f} {metric['r2']:8.2f}")