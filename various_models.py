# %%
# Required imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
%matplotlib inline
import matplotlib.pyplot as plt
import os
from joblib import Parallel, delayed
from datetime import timedelta
import time
from tqdm import tqdm
from sklearn.model_selection import KFold

# Global configuration
CONFIG = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "batch_size": 128 if torch.cuda.is_available() else 64,
    "epochs": 300,
    "patience": 50,
    "learning_rate": 0.0008,
    "embedding_dim": 32,
    "window_hours": 3,
    "cap_normal": 30,
    "cap_bg": 300,
    "cap_iob": 5,
    "cap_carb": 150,
    "data_path": os.path.join(os.getcwd(), "subjects")
}

# %% CELL: Device Check
def check_device():
    print("Device:", CONFIG["device"])
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    if torch.cuda.is_available():
        print("Device count:", torch.cuda.device_count())
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(0))
        # Test GPU operation
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=CONFIG["device"])
        b = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=CONFIG["device"])
        c = torch.matmul(a, b)
        print("Test GPU operation successful:", c)
    else:
        print("CUDA is not available. Running on CPU.")
check_device()

# %% CELL: Data Processing Functions
# Update get_cgm_window to dynamically calculate the number of points
def get_cgm_window(bolus_time, cgm_df, window_hours=CONFIG["window_hours"]):
    """
    Obtiene una ventana de datos CGM alrededor del tiempo del bolo de insulina.
    
    Args:
        bolus_time: Tiempo del bolo de insulina
        cgm_df: DataFrame con datos CGM
        window_hours: Tamaño de la ventana en horas
    
    Returns:
        Array numpy con los valores CGM o None si no hay suficientes datos
    """
    # Calculate the number of points based on window_hours (5-minute intervals: 12 points/hour)
    num_points = int(window_hours * 12)  # e.g., 4 hours * 12 points/hour = 48 points
    
    # Calculate the start of the window
    window_start = bolus_time - timedelta(hours=window_hours)
    
    # Get CGM data within the time window
    window = cgm_df[(cgm_df['date'] >= window_start) & (cgm_df['date'] <= bolus_time)]
    
    # Sort by date and take the last num_points values
    window = window.sort_values('date').tail(num_points)
    
    # Check if we have enough data points
    if len(window) < num_points:
        # Option 1: Return None if not enough data (conservative approach)
        return None
        # Option 2: Pad with the last available value (alternative approach)
        # padding = np.full(num_points - len(window), window['mg/dl'].iloc[-1] if not window.empty else 0)
        # return np.concatenate([window['mg/dl'].values, padding]) if not window.empty else None
    
    return window['mg/dl'].values

def calculate_iob(bolus_time, basal_df, half_life_hours=4):
    """
    Calcula la insulina activa (IOB) en un momento dado.
    
    Args:
        bolus_time: Tiempo para calcular IOB
        basal_df: DataFrame con datos de insulina basal
        half_life_hours: Vida media de la insulina en horas
    
    Returns:
        Float con cantidad de insulina activa
    """
    if basal_df is None or basal_df.empty:
        return 0.0
        
    iob = 0
    # Itera sobre cada registro de insulina basal
    for _, row in basal_df.iterrows():
        start_time = row['date']
        # Convierte duración de milisegundos a horas
        duration_hours = row['duration'] / (1000 * 3600)
        end_time = start_time + timedelta(hours=duration_hours)
        # Usa tasa de 0.9 si no hay valor
        rate = row['rate'] if pd.notna(row['rate']) else 0.9
        
        # Si el tiempo del bolo está dentro del período activo
        if start_time <= bolus_time <= end_time:
            # Calcula tiempo transcurrido y restante
            time_since_start = (bolus_time - start_time).total_seconds() / 3600
            remaining = rate * (1 - (time_since_start / half_life_hours))
            iob += max(0, remaining)
            
    return iob

def process_subject(subject_path, idx):
    """
    Procesa los datos de un sujeto del estudio.
    
    Args:
        subject_path: Ruta al archivo Excel con datos del sujeto
        idx: Índice/ID del sujeto
    
    Returns:
        Lista de diccionarios con características procesadas
    """
    start_time = time.time()
    
    # Carga datos desde Excel
    try:
        excel_file = pd.ExcelFile(subject_path)
        cgm_df = pd.read_excel(excel_file, sheet_name="CGM")
        bolus_df = pd.read_excel(excel_file, sheet_name="Bolus")
        try:
            basal_df = pd.read_excel(excel_file, sheet_name="Basal")
        except ValueError:
            basal_df = None
    except Exception as e:
        print(f"Error al cargar {os.path.basename(subject_path)}: {e}")
        return []

    # Convierte fechas a datetime
    cgm_df['date'] = pd.to_datetime(cgm_df['date'])
    cgm_df = cgm_df.sort_values('date')
    bolus_df['date'] = pd.to_datetime(bolus_df['date'])
    if basal_df is not None:
        basal_df['date'] = pd.to_datetime(basal_df['date'])

    processed_data = []
    # Procesa cada registro de bolo
    for _, row in tqdm(bolus_df.iterrows(), total=len(bolus_df), desc=f"Procesando {os.path.basename(subject_path)}", leave=False):
        bolus_time = row['date']
        cgm_window = get_cgm_window(bolus_time, cgm_df)
        
        if cgm_window is not None:
            # Calcula características
            iob = calculate_iob(bolus_time, basal_df)
            hour_of_day = bolus_time.hour / 23.0  # Normaliza hora del día
            
            # Usa último valor CGM si no hay bgInput
            bg_input = row['bgInput'] if pd.notna(row['bgInput']) else cgm_window[-1]
            
            # Procesa dosis normal y límites
            normal = row['normal'] if pd.notna(row['normal']) else 0.0
            normal = np.clip(normal, 0, CONFIG["cap_normal"])
            
            # Calcula factor de sensibilidad personalizado
            isf_custom = 50.0 if normal <= 0 or bg_input <= 100 else (bg_input - 100) / normal
            
            # Aplica límites a variables
            bg_input = np.clip(bg_input, 0, CONFIG["cap_bg"])
            iob = np.clip(iob, 0, CONFIG["cap_iob"])
            carb_input = row['carbInput'] if pd.notna(row['carbInput']) else 0.0
            carb_input = np.clip(carb_input, 0, CONFIG["cap_carb"])
            
            # Crea diccionario de características
            features = {
                'subject_id': idx,
                'cgm_window': cgm_window,
                'carbInput': carb_input,
                'bgInput': bg_input,
                'insulinCarbRatio': row['insulinCarbRatio'] if pd.notna(row['insulinCarbRatio']) else 10.0,
                'insulinSensitivityFactor': isf_custom,
                'insulinOnBoard': iob,
                'hour_of_day': hour_of_day,
                'normal': normal
            }
            processed_data.append(features)

    elapsed_time = time.time() - start_time
    print(f"Procesado {os.path.basename(subject_path)} (Sujeto {idx+1}) en {elapsed_time:.2f} segundos")
    return processed_data

def preprocess_data(subject_folder):
    """
    Preprocesa los datos de todos los sujetos para el entrenamiento del modelo.
    
    Args:
        subject_folder: Ruta a la carpeta que contiene los archivos Excel de los sujetos
        
    Returns:
        X_cgm: Array de datos CGM normalizados
        X_other: Array de otras características normalizadas
        X_subject: Array de IDs de sujetos
        y: Array de dosis objetivo normalizadas
        df_final: DataFrame con todos los datos procesados
        scaler_cgm: Scaler usado para normalizar datos CGM
        scaler_other: Scaler usado para normalizar otras características
        scaler_y: Scaler usado para normalizar objetivos
    """
    start_time = time.time()
    
    subject_files = [f for f in os.listdir(subject_folder) if f.startswith("Subject") and f.endswith(".xlsx")]
    print(f"\nFound Subject files ({len(subject_files)}):")
    for f in subject_files:
        print(f)

    all_processed_data = Parallel(n_jobs=-1)(delayed(process_subject)(os.path.join(subject_folder, f), idx) 
                                            for idx, f in enumerate(subject_files))
    all_processed_data = [item for sublist in all_processed_data for item in sublist]

    df_processed = pd.DataFrame(all_processed_data)
    print("Muestra de datos procesados combinados:")
    print(df_processed.head())
    print(f"Total de muestras: {len(df_processed)}")

    # Add new features: CGM trend and variance
    num_points = int(CONFIG["window_hours"] * 12)  # Dynamically calculate based on window_hours
    cgm_columns = [f'cgm_{i}' for i in range(num_points)]  # e.g., 48 columns for 4 hours
    
    # Create DataFrame from cgm_window
    df_cgm = pd.DataFrame(df_processed['cgm_window'].tolist(), columns=cgm_columns, index=df_processed.index)
    
    # Compute CGM trend (slope) and variance
    df_processed['cgm_trend'] = df_cgm[cgm_columns].apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], axis=1)
    df_processed['cgm_variance'] = df_cgm[cgm_columns].var(axis=1)
    
    df_final = pd.concat([df_cgm, df_processed.drop(columns=['cgm_window'])], axis=1)
    df_final = df_final.dropna()

    # Outlier detection: Clip CGM values beyond 3 standard deviations
    cgm_mean = df_final[cgm_columns].mean().mean()
    cgm_std = df_final[cgm_columns].std().mean()
    df_final[cgm_columns] = df_final[cgm_columns].clip(lower=cgm_mean - 3*cgm_std, upper=cgm_mean + 3*cgm_std)

    scaler_cgm = StandardScaler()
    scaler_other = StandardScaler()
    scaler_y = StandardScaler()

    X_cgm = scaler_cgm.fit_transform(df_final[cgm_columns]).reshape(-1, num_points, 1)
    X_subject = df_final['subject_id'].values
    other_features = ['carbInput', 'bgInput', 'insulinOnBoard', 'insulinCarbRatio', 
                      'insulinSensitivityFactor', 'hour_of_day', 'cgm_trend', 'cgm_variance']
    X_other = scaler_other.fit_transform(df_final[other_features])
    y = scaler_y.fit_transform(df_final['normal'].values.reshape(-1, 1)).flatten()

    print("NaN en X_cgm:", np.isnan(X_cgm).sum())
    print("NaN en X_other:", np.isnan(X_other).sum())
    print("NaN en X_subject:", np.isnan(X_subject).sum())
    print("NaN in y:", np.isnan(y).sum())
    if np.isnan(X_cgm).sum() > 0 or np.isnan(X_other).sum() > 0 or np.isnan(X_subject).sum() > 0 or np.isnan(y).sum() > 0:
        raise ValueError("Valores NaN detectados")

    elapsed_time = time.time() - start_time
    print(f"Preprocesamiento completo en {elapsed_time:.2f} segundos")
    return X_cgm, X_other, X_subject, y, df_final, scaler_cgm, scaler_other, scaler_y


def split_data(X_cgm, X_other, X_subject, y, df_final):
    """
    Divide los datos en conjuntos de entrenamiento, validación y prueba.
    
    Args:
        X_cgm: Array de datos CGM
        X_other: Array de otras características
        X_subject: Array de IDs de sujetos
        y: Array de objetivos
        df_final: DataFrame con todos los datos
        
    Returns:
        Tupla con datos divididos para entrenamiento, validación y prueba
    """
    start_time = time.time()
    
    # Dividir sujetos en conjuntos de entrenamiento, validación y prueba
    subject_ids = df_final['subject_id'].unique()
    train_subjects, temp_subjects = train_test_split(subject_ids, test_size=0.2, random_state=42)
    val_subjects, test_subjects = train_test_split(temp_subjects, test_size=0.5, random_state=42)

    # Crear máscaras para cada conjunto
    train_mask = df_final['subject_id'].isin(train_subjects)
    val_mask = df_final['subject_id'].isin(val_subjects)
    test_mask = df_final['subject_id'].isin(test_subjects)

    # Aplicar máscaras para dividir los datos
    X_cgm_train, X_cgm_val, X_cgm_test = X_cgm[train_mask], X_cgm[val_mask], X_cgm[test_mask]
    X_other_train, X_other_val, X_other_test = X_other[train_mask], X_other[val_mask], X_other[test_mask]
    X_subject_train, X_subject_val, X_subject_test = X_subject[train_mask], X_subject[val_mask], X_subject[test_mask]
    y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]
    subject_test = df_final[test_mask]['subject_id'].values

    # Imprimir información sobre las divisiones
    print(f"Entrenamiento CGM: {X_cgm_train.shape}, Validación CGM: {X_cgm_val.shape}, Prueba CGM: {X_cgm_test.shape}")
    print(f"Entrenamiento Otros: {X_other_train.shape}, Validación Otros: {X_other_val.shape}, Prueba Otros: {X_other_test.shape}")
    print(f"Entrenamiento Subject: {X_subject_train.shape}, Validación Subject: {X_subject_val.shape}, Prueba Subject: {X_subject_test.shape}")
    print(f"Sujetos de prueba: {test_subjects}")

    elapsed_time = time.time() - start_time
    print(f"División de datos completa en {elapsed_time:.2f} segundos")
    return (X_cgm_train, X_cgm_val, X_cgm_test,
            X_other_train, X_other_val, X_other_test,
            X_subject_train, X_subject_val, X_subject_test,
            y_train, y_val, y_test, subject_test)

# %% CELL: Create DataLoaders
def create_dataloaders(X_cgm_train, X_cgm_val, X_cgm_test,
                      X_other_train, X_other_val, X_other_test,
                      X_subject_train, X_subject_val, X_subject_test,
                      y_train, y_val, y_test):
    """
    Crea los DataLoaders para entrenamiento, validación y prueba.
    
    Args:
        X_cgm_train/val/test: Datos CGM para cada conjunto
        X_other_train/val/test: Otras características para cada conjunto  
        X_subject_train/val/test: IDs de sujetos para cada conjunto
        y_train/val/test: Variables objetivo para cada conjunto

    Returns:
        train_loader: DataLoader para entrenamiento
        val_loader: DataLoader para validación  
        test_loader: DataLoader para prueba
    """
    start_time = time.time()
    
    # Convertir datos a tensores PyTorch y mover a GPU/CPU
    X_cgm_train = torch.tensor(X_cgm_train, dtype=torch.float32).to(CONFIG["device"])
    X_cgm_val = torch.tensor(X_cgm_val, dtype=torch.float32).to(CONFIG["device"])
    X_cgm_test = torch.tensor(X_cgm_test, dtype=torch.float32).to(CONFIG["device"])
    X_other_train = torch.tensor(X_other_train, dtype=torch.float32).to(CONFIG["device"])
    X_other_val = torch.tensor(X_other_val, dtype=torch.float32).to(CONFIG["device"]) 
    X_other_test = torch.tensor(X_other_test, dtype=torch.float32).to(CONFIG["device"])
    X_subject_train = torch.tensor(X_subject_train, dtype=torch.long).to(CONFIG["device"])
    X_subject_val = torch.tensor(X_subject_val, dtype=torch.long).to(CONFIG["device"])
    X_subject_test = torch.tensor(X_subject_test, dtype=torch.long).to(CONFIG["device"])
    y_train = torch.tensor(y_train, dtype=torch.float32).to(CONFIG["device"])
    y_val = torch.tensor(y_val, dtype=torch.float32).to(CONFIG["device"])
    y_test = torch.tensor(y_test, dtype=torch.float32).to(CONFIG["device"])

    # Crear datasets
    train_dataset = TensorDataset(X_cgm_train, X_other_train, X_subject_train, y_train)
    val_dataset = TensorDataset(X_cgm_val, X_other_val, X_subject_val, y_val)
    test_dataset = TensorDataset(X_cgm_test, X_other_test, X_subject_test, y_test)

    # Crear dataloaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    elapsed_time = time.time() - start_time
    print(f"Creación de DataLoaders completa en {elapsed_time:.2f} segundos")
    return train_loader, val_loader, test_loader

# %% CELL: Model Definitions 
class EnhancedLSTM(nn.Module):
    def __init__(self, num_subjects, embedding_dim=CONFIG["embedding_dim"]):
        super().__init__()
        self.subject_embedding = nn.Embedding(num_subjects, embedding_dim)
        self.lstm = nn.LSTM(1, 384, num_layers=3, batch_first=True, dropout=0.3)  # Increased hidden units, reduced dropout
        self.attention = nn.MultiheadAttention(embed_dim=384, num_heads=8, dropout=0.3)
        self.batch_norm1 = nn.BatchNorm1d(384)
        self.dropout1 = nn.Dropout(0.3)
        self.concat_dense1 = nn.Linear(384 + 8 + embedding_dim, 256)  # Updated for new features
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        self.concat_dense2 = nn.Linear(256, 128)  # Additional dense layer
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.3)
        self.output_layer = nn.Linear(128, 1)
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, cgm_input, other_input, subject_ids):
        batch_size = cgm_input.size(0)
        subject_embed = self.subject_embedding(subject_ids)
        
        lstm_out, _ = self.lstm(cgm_input)  # [batch_size, seq_len, 512]
        lstm_out = lstm_out.permute(1, 0, 2)  # [seq_len, batch_size, 512] for attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)  # Apply attention
        attn_out = attn_out.permute(1, 0, 2)  # [batch_size, seq_len, 512]
        attn_out = attn_out[:, -1, :]  # Take last timestep
        
        attn_out = self.batch_norm1(attn_out.unsqueeze(2)).squeeze(2)
        attn_out = self.dropout1(attn_out)
        
        combined = torch.cat((attn_out, other_input, subject_embed), dim=1)
        dense_out = torch.relu(self.concat_dense1(combined))
        dense_out = self.batch_norm2(dense_out.unsqueeze(2)).squeeze(2)
        dense_out = self.dropout2(dense_out)
        dense_out = torch.relu(self.concat_dense2(dense_out))
        dense_out = self.batch_norm3(dense_out.unsqueeze(2)).squeeze(2)
        dense_out = self.dropout3(dense_out)
        return self.output_layer(dense_out)

class TCNTransformer(nn.Module):
    """
    Modelo híbrido que combina TCN (Temporal Convolutional Network) y Transformer.
    """
    def __init__(self, num_subjects, embedding_dim=CONFIG["embedding_dim"]):
        super().__init__()
        # Embedding de sujetos
        self.subject_embedding = nn.Embedding(num_subjects, embedding_dim)
        
        # Capas convolucionales temporales
        self.conv1 = nn.Conv1d(1, 128, kernel_size=3, padding='same')
        self.pool1 = nn.MaxPool1d(2) 
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding='same')
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding='same')
        
        # Capas del transformer
        self.ln1 = nn.LayerNorm(128)
        self.transformer = nn.TransformerEncoderLayer(128, nhead=8, dropout=0.2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer, num_layers=1)
        self.ln2 = nn.LayerNorm(128)
        
        # Pooling y capas fully connected
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.concat_dense1 = nn.Linear(128 + 6 + embedding_dim, 256)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.4)
        self.dense2 = nn.Linear(256, 128)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.4)
        self.output_layer = nn.Linear(128, 1)

    def forward(self, cgm_input, other_input, subject_ids):
        # Proceso de embedding
        subject_embed = self.subject_embedding(subject_ids)
        
        # Proceso TCN
        cgm_input = cgm_input.permute(0, 2, 1)
        tcn_out = torch.relu(self.conv1(cgm_input))
        tcn_out = self.pool1(tcn_out)
        tcn_out = torch.relu(self.conv2(tcn_out))
        tcn_out = self.pool2(tcn_out)
        tcn_out = torch.relu(self.conv3(tcn_out))
        
        # Proceso Transformer
        tcn_out = tcn_out.permute(0, 2, 1)
        tcn_out = self.ln1(tcn_out)
        transformer_out = self.transformer_encoder(tcn_out)
        transformer_out = transformer_out + tcn_out # Conexión residual
        transformer_out = self.ln2(transformer_out)
        
        # Proceso final
        transformer_out = transformer_out.permute(0, 2, 1)
        pooled = self.global_pool(transformer_out).squeeze(-1)
        combined = torch.cat((pooled, other_input, subject_embed), dim=1)
        dense_out = torch.relu(self.concat_dense1(combined))
        dense_out = self.batch_norm1(dense_out)
        dense_out = self.dropout1(dense_out)
        dense_out = torch.relu(self.dense2(dense_out))
        dense_out = self.batch_norm2(dense_out)
        dense_out = self.dropout2(dense_out)
        return self.output_layer(dense_out)

class EnhancedGRU(nn.Module):
    def __init__(self, num_subjects, embedding_dim=CONFIG["embedding_dim"]):
        super().__init__()
        self.subject_embedding = nn.Embedding(num_subjects, embedding_dim)
        self.gru = nn.GRU(1, 384, num_layers=3, batch_first=True, dropout=0.3)  # Increased hidden units, reduced dropout
        self.attention = nn.MultiheadAttention(embed_dim=384, num_heads=8, dropout=0.3)
        self.batch_norm1 = nn.BatchNorm1d(384)
        self.dropout1 = nn.Dropout(0.3)
        self.concat_dense1 = nn.Linear(384 + 8 + embedding_dim, 256)  # Updated for new features
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        self.concat_dense2 = nn.Linear(256, 128)  # Additional dense layer
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.3)
        self.output_layer = nn.Linear(128, 1)
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, cgm_input, other_input, subject_ids):
        subject_embed = self.subject_embedding(subject_ids)
        gru_out, _ = self.gru(cgm_input)
        gru_out = gru_out.permute(1, 0, 2)  # [seq_len, batch_size, 512] for attention
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        attn_out = attn_out.permute(1, 0, 2)  # [batch_size, seq_len, 512]
        attn_out = attn_out[:, -1, :]
        
        attn_out = self.batch_norm1(attn_out.unsqueeze(2)).squeeze(2)
        attn_out = self.dropout1(attn_out)
        
        combined = torch.cat((attn_out, other_input, subject_embed), dim=1)
        dense_out = torch.relu(self.concat_dense1(combined))
        dense_out = self.batch_norm2(dense_out.unsqueeze(2)).squeeze(2)
        dense_out = self.dropout2(dense_out)
        dense_out = torch.relu(self.concat_dense2(dense_out))
        dense_out = self.batch_norm3(dense_out.unsqueeze(2)).squeeze(2)
        dense_out = self.dropout3(dense_out)
        return self.output_layer(dense_out)

# %% CELL: Training Functions
"""
Funciones de entrenamiento y evaluación de modelos de aprendizaje profundo.
"""

def custom_mse(y_true, y_pred, subject_ids, problem_subject=49, weight_reduction=0.5, l1_lambda=1e-5):    
    """
    Error cuadrático medio personalizado con penalización por sobrepredicción.
    
    Args:
        y_true: Valores objetivo reales
        y_pred: Predicciones del modelo 
        subject_ids: IDs de los sujetos
        problem_subject: ID del sujeto problemático (default: 49)
        weight_reduction: Factor de reducción de peso para sujeto problemático (default: 0.5)
    
    Returns:
        Error medio ponderado con penalización
    """
    error = y_true - y_pred
    overprediction_penalty = torch.where(error < 0, 2 * error**2, error**2)
    
    # Clinical penalty: Higher penalty for predictions leading to hypoglycemia
    glucose_pred = 100 - (y_pred * 50)  # Simplified mapping (adjust based on actual glucose prediction)
    hypo_penalty = torch.where(glucose_pred < 70, (70 - glucose_pred)**2, torch.zeros_like(glucose_pred))
    
    weights = torch.ones_like(y_true)
    weights[subject_ids == problem_subject] = weight_reduction
    mse_loss = torch.mean((overprediction_penalty + hypo_penalty) * weights)
    
    # L1 regularization
    l1_loss = 0
    for param in model.parameters():
        l1_loss += torch.norm(param, p=1)
    return mse_loss + l1_lambda * l1_loss

def train_model(model, train_loader, val_loader, model_name=None):
    """
    Entrena un modelo usando early stopping y learning rate scheduling.
    
    Args:
        model: Modelo a entrenar 
        train_loader: DataLoader con datos de entrenamiento
        val_loader: DataLoader con datos de validación
        
    Returns:
        Tuple con historiales de pérdida de entrenamiento y validación
    """
    start_time = time.time()
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"], eta_min=1e-6)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    train_losses, val_losses = [], []
    
    for epoch in tqdm(range(CONFIG["epochs"]), desc="Entrenamiento", unit="época"):
        model.train()
        train_loss = 0
        for cgm_batch, other_batch, subject_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(cgm_batch, other_batch, subject_batch).squeeze()
            loss = custom_mse(y_batch, y_pred, subject_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Increased
            optimizer.step()
            train_loss += loss.item() * len(y_batch)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for cgm_batch, other_batch, subject_batch, y_batch in val_loader:
                y_pred = model(cgm_batch, other_batch, subject_batch).squeeze()
                loss = custom_mse(y_batch, y_pred, subject_batch)
                val_loss += loss.item() * len(y_batch)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        scheduler.step()
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                print("Early stopping triggered")
                break

    model.load_state_dict(best_model_state)
    elapsed_time = time.time() - start_time
    print(f"Entrenamiento completo en {elapsed_time:.2f} segundos")
    return train_losses, val_losses

# %% CELL: Evaluation Functions
def evaluate_model(model, loader, scaler_y):
    """
    Evalúa el modelo en un conjunto de datos.
    
    Args:
        model: Modelo a evaluar
        loader: DataLoader con datos de evaluación
        scaler_y: Scaler para desnormalizar predicciones
        
    Returns:
        Tuple con predicciones y valores objetivo desnormalizados
    """
    start_time = time.time()
    model.eval()
    predictions, targets = [], []
    
    # Generar predicciones
    with torch.no_grad():
        for cgm_batch, other_batch, subject_batch, y_batch in tqdm(loader, desc="Evaluando", leave=False):
            y_pred = model(cgm_batch, other_batch, subject_batch).squeeze()
            predictions.append(y_pred.cpu().numpy())
            targets.append(y_batch.cpu().numpy())
            
    # Procesar y desnormalizar resultados
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
    targets = scaler_y.inverse_transform(targets.reshape(-1, 1)).flatten()

    elapsed_time = time.time() - start_time
    print(f"Evaluación completa en {elapsed_time:.2f} segundos")
    return predictions, targets

def rule_based_prediction(X_other, scaler_other, scaler_y, target_bg=100):
    """
    Genera predicciones basadas en reglas médicas estándar.
    
    Args:
        X_other: Características adicionales normalizadas
        scaler_other: Scaler para desnormalizar características
        scaler_y: Scaler para normalizar predicciones
        target_bg: Nivel objetivo de glucosa en sangre
        
    Returns:
        Array con predicciones de dosis
    """
    start_time = time.time()
    
    # Preparación de datos
    if hasattr(X_other, "cpu"):
        X_other_np = X_other.cpu().numpy()
    else:
        X_other_np = X_other
        
    # Desnormalizar características
    inverse_transformed = scaler_other.inverse_transform(X_other_np)
    carb_input, bg_input, icr, isf = (inverse_transformed[:, 0],
                                     inverse_transformed[:, 1],
                                     inverse_transformed[:, 3],
                                     inverse_transformed[:, 4])
    
    # Evitar división por cero
    icr = np.where(icr == 0, 1e-6, icr)
    isf = np.where(isf == 0, 1e-6, isf)
    
    # Cálculo de componentes de la dosis
    carb_component = np.divide(carb_input, icr, out=np.zeros_like(carb_input), where=icr!=0)
    bg_component = np.divide(bg_input - target_bg, isf, out=np.zeros_like(bg_input), where=isf!=0)
    
    # Predicción final
    prediction = carb_component + bg_component
    prediction = np.clip(prediction, 0, CONFIG["cap_normal"])

    elapsed_time = time.time() - start_time
    print(f"Predicción basada en reglas completa en {elapsed_time:.2f} segundos")
    return prediction

# %% CELL: Visualization Functions
def plot_training_history(lstm_losses, gru_losses):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(lstm_losses[0], label='Train LSTM')
    plt.plot(lstm_losses[1], label='Val LSTM')
    plt.xlabel('Epoch')
    plt.ylabel('Custom MSE Loss')
    plt.legend()
    plt.title('LSTM Training History')

    # plt.subplot(1, 3, 2)
    # plt.plot(tcn_losses[0], label='Train TCN')
    # plt.plot(tcn_losses[1], label='Val TCN')
    # plt.xlabel('Epoch')
    # plt.ylabel('Custom MSE Loss')
    # plt.legend()
    # plt.title('TCN Training History')

    plt.subplot(1, 2, 2)
    plt.plot(gru_losses[0], label='Train GRU')
    plt.plot(gru_losses[1], label='Val GRU')
    plt.xlabel('Epoch')
    plt.ylabel('Custom MSE Loss')
    plt.legend()
    plt.title('GRU Training History')

    plt.tight_layout()
    plt.show()
#%%
def plot_evaluation(y_test, y_pred_lstm,  y_pred_gru, y_rule, subject_test, scaler_y):
    start_time = time.time()
    y_test_denorm = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.scatter(y_test_denorm, y_pred_lstm, label='LSTM', alpha=0.5, color='blue')
    # plt.scatter(y_test_denorm, y_pred_tcn, label='TCN', alpha=0.5, color='green')
    plt.scatter(y_test_denorm, y_pred_gru, label='GRU', alpha=0.5, color='red')
    plt.scatter(y_test_denorm, y_rule, label='Rules', alpha=0.5, color='orange')
    plt.plot([0, 15], [0, 15], 'r--')
    plt.xlabel('Real Dose (units)')
    plt.ylabel('Predicted Dose (units)')
    plt.legend()
    plt.title('Predictions vs Real')

    plt.subplot(2, 2, 2)
    plt.hist(y_test_denorm - y_pred_lstm, bins=20, label='LSTM', alpha=0.5, color='blue')
    # plt.hist(y_test_denorm - y_pred_tcn, bins=20, label='TCN', alpha=0.5, color='green')
    plt.hist(y_test_denorm - y_pred_gru, bins=20, label='GRU', alpha=0.5, color='red')
    plt.hist(y_test_denorm - y_rule, bins=20, label='Rules', alpha=0.5, color='orange')
    plt.xlabel('Residual (units)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Residual Distribution')

    test_subjects = np.unique(subject_test)
    mae_lstm, mae_tcn, mae_gru, mae_rule = [], [], [], []
    for sid in test_subjects:
        mask = subject_test == sid
        if np.sum(mask) > 0:
            mae_lstm.append(mean_absolute_error(y_test_denorm[mask], y_pred_lstm[mask]))
            # mae_tcn.append(mean_absolute_error(y_test_denorm[mask], y_pred_tcn[mask]))
            mae_gru.append(mean_absolute_error(y_test_denorm[mask], y_pred_gru[mask]))
            mae_rule.append(mean_absolute_error(y_test_denorm[mask], y_rule[mask]))

    plt.subplot(2, 2, 3)
    plt.bar(np.arange(len(test_subjects)) - 0.3, mae_lstm, width=0.2, label='LSTM', color='blue')
    # plt.bar(np.arange(len(test_subjects)) - 0.1, mae_tcn, width=0.2, label='TCN', color='green')
    plt.bar(np.arange(len(test_subjects)) + 0.1, mae_gru, width=0.2, label='GRU', color='red')
    plt.bar(np.arange(len(test_subjects)) + 0.3, mae_rule, width=0.2, label='Rules', color='orange')
    plt.xlabel('Subject')
    plt.ylabel('MAE (units)')
    plt.xticks(np.arange(len(test_subjects)), test_subjects)
    plt.legend()
    plt.title('MAE by Subject')

    plt.tight_layout()
    plt.show()
    elapsed_time = time.time() - start_time
    print(f"Visualización completa en {elapsed_time:.2f} segundos")

"""
Módulo principal de ejecución del modelo de predicción de insulina.
Este módulo coordina el preprocesamiento de datos, entrenamiento y evaluación 
de múltiples modelos de deep learning para predecir dosis de insulina.

Modelos implementados:
- LSTM mejorado con embeddings de sujetos
- TCN híbrido con transformer 
- GRU mejorado con regularización
- Modelo basado en reglas médicas
"""

# %% CELL: Main Execution - Preprocess Data
# Preprocesa los datos y aplica normalización
X_cgm, X_other, X_subject, y, df_final, scaler_cgm, scaler_other, scaler_y = preprocess_data(CONFIG["data_path"])
# %% CELL: Main Execution - Split Data
# División estratificada de datos en conjuntos de train/val/test manteniendo la distribución de sujetos
(X_cgm_train, X_cgm_val, X_cgm_test,
    X_other_train, X_other_val, X_other_test,
    X_subject_train, X_subject_val, X_subject_test,
    y_train, y_val, y_test, subject_test) = split_data(X_cgm, X_other, X_subject, y, df_final)

# %% CELL: Main Execution - Create DataLoaders
# Crea los DataLoaders para entrenamiento eficiente con batches
train_loader, val_loader, test_loader = create_dataloaders(X_cgm_train, X_cgm_val, X_cgm_test,
                                                            X_other_train, X_other_val, X_other_test,
                                                            X_subject_train, X_subject_val, X_subject_test,
                                                            y_train, y_val, y_test)

# %% CELL: Main Execution - Model Initialization
# Inicializa los tres modelos de deep learning
num_subjects = len(df_final['subject_id'].unique())
models = {
    "LSTM": EnhancedLSTM(num_subjects).to(CONFIG["device"]),
    #"TCN": TCNTransformer(num_subjects).to(CONFIG["device"]),
    "GRU": EnhancedGRU(num_subjects).to(CONFIG["device"])
}

# %% CELL: Main Execution - Training
# Entrena cada modelo usando early stopping y learning rate scheduling
losses = {}
for name, model in models.items():
    print(f"\nEntrenando {name}...")
    losses[name] = train_model(model, train_loader, val_loader, model_name=name)

# %% CELL: Main Execution - Evaluation
# Evalúa los modelos en el conjunto de test
y_pred = {name: evaluate_model(model, test_loader, scaler_y)[0] for name, model in models.items()}
y_rule = rule_based_prediction(X_other_test, scaler_other, scaler_y)

# %% CELL: Main Execution - Print Metrics
# Calcula y muestra métricas globales para cada modelo
for name in models:
    mae = mean_absolute_error(scaler_y.inverse_transform(y_test.reshape(-1, 1)), y_pred[name])
    rmse = np.sqrt(mean_squared_error(scaler_y.inverse_transform(y_test.reshape(-1, 1)), y_pred[name]))
    r2 = r2_score(scaler_y.inverse_transform(y_test.reshape(-1, 1)), y_pred[name])
    print(f"{name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

# Métricas para el modelo basado en reglas
mae_rule = mean_absolute_error(scaler_y.inverse_transform(y_test.reshape(-1, 1)), y_rule)
rmse_rule = np.sqrt(mean_squared_error(scaler_y.inverse_transform(y_test.reshape(-1, 1)), y_rule))
r2_rule = r2_score(scaler_y.inverse_transform(y_test.reshape(-1, 1)), y_rule)
print(f"Rules - MAE: {mae_rule:.2f}, RMSE: {rmse_rule:.2f}, R²: {r2_rule:.2f}")

# %% CELL: Main Execution - Visualization
# Genera visualizaciones comparativas del entrenamiento y predicciones
# plot_training_history(losses["LSTM"], losses["TCN"], losses["GRU"])
plot_training_history(losses["LSTM"], losses["GRU"])

# plot_evaluation(y_test, y_pred["LSTM"], y_pred["TCN"], y_pred["GRU"], y_rule, subject_test, scaler_y)
plot_evaluation(y_test, y_pred["LSTM"],y_pred["GRU"], y_rule, subject_test, scaler_y)

# %% CELL: Main Execution - Metrics per Subject
# Analiza el rendimiento individual por sujeto
print("\nRendimiento por sujeto:")
for subject_id in np.unique(subject_test):
    mask = subject_test == subject_id
    if np.sum(mask) > 0:
        y_test_sub = scaler_y.inverse_transform(y_test[mask].reshape(-1, 1)).flatten()
        print(f"Sujeto {subject_id}: ", end="")
        for name in models:
            mae_sub = mean_absolute_error(y_test_sub, y_pred[name][mask])
            print(f"{name} MAE={mae_sub:.2f}, ", end="")
        mae_rule_sub = mean_absolute_error(y_test_sub, y_rule[mask])
        print(f"Rules MAE={mae_rule_sub:.2f}")




