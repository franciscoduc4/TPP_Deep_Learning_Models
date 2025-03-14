# %% [markdown]
# ### Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import os
from joblib import Parallel, delayed
from datetime import timedelta

# Configuración de matplotlib
import matplotlib
matplotlib.use('TkAgg')  # o 'Agg' para no interactivo
import matplotlib.pyplot as plt

# %% [markdown]
# ### Verify GPU Availability with PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0))

# Small test operation on GPU
if torch.cuda.is_available():
    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device)
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=device)
    c = torch.matmul(a, b)
    print("Test GPU operation successful:", c)

# %% [markdown]
# ### Data Preprocessing
subject_folder = os.path.join(os.getcwd(), "subjects")  # Path to subjects/ folder
subject_files = [f for f in os.listdir(subject_folder) if f.startswith("Subject") and f.endswith(".xlsx")]
print(f"\nFound Subject files ({len(subject_files)}):")
for f in subject_files:
    print(f)

# Paso 1: Preprocesar datos para todos los sujetos
def get_cgm_window(bolus_time, cgm_df, window_hours=2, interval_minutes=5):
    window_start = bolus_time - timedelta(hours=window_hours)
    window = cgm_df[(cgm_df['date'] >= window_start) & (cgm_df['date'] <= bolus_time)]
    window = window.sort_values('date').tail(24)
    if len(window) < 24:
        return None
    return window['mg/dl'].values

def calculate_iob(bolus_time, basal_df, half_life_hours=4):
    if basal_df is None or basal_df.empty:
        return 0.0
    iob = 0
    for _, row in basal_df.iterrows():
        start_time = row['date']
        duration_hours = row['duration'] / (1000 * 3600)
        end_time = start_time + timedelta(hours=duration_hours)
        rate = row['rate'] if pd.notna(row['rate']) else 0.9
        if start_time <= bolus_time <= end_time:
            time_since_start = (bolus_time - start_time).total_seconds() / 3600
            remaining = rate * (1 - (time_since_start / half_life_hours))
            iob += max(0, remaining)
    return iob

def process_subject(subject_path, idx):
    print(f"Procesando {os.path.basename(subject_path)} ({idx+1}/{len(subject_files)})...")
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

    cgm_df['date'] = pd.to_datetime(cgm_df['date'])
    cgm_df = cgm_df.sort_values('date')
    bolus_df['date'] = pd.to_datetime(bolus_df['date'])
    if basal_df is not None:
        basal_df['date'] = pd.to_datetime(basal_df['date'])

    processed_data = []
    for _, row in bolus_df.iterrows():
        bolus_time = row['date']
        cgm_window = get_cgm_window(bolus_time, cgm_df)
        if cgm_window is not None:
            iob = calculate_iob(bolus_time, basal_df)
            hour_of_day = bolus_time.hour / 23.0
            bg_input = row['bgInput'] if pd.notna(row['bgInput']) else cgm_window[-1]
            normal = row['normal'] if pd.notna(row['normal']) else 0.0
            # Cap para normal (etiqueta objetivo)
            normal = np.clip(normal, 0, 30)
            isf_custom = 50.0
            if normal > 0 and bg_input > 100:
                isf_custom = (bg_input - 100) / normal
            # Cap outliers for all subjects
            bg_input = np.clip(bg_input, 0, 300)
            iob = np.clip(iob, 0, 5)
            carb_input = row['carbInput'] if pd.notna(row['carbInput']) else 0.0
            carb_input = np.clip(carb_input, 0, 150)
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
    
    return processed_data

# Ejecución en paralelo
all_processed_data = Parallel(n_jobs=-1)(delayed(process_subject)(os.path.join(subject_folder, f), idx) 
                                        for idx, f in enumerate(subject_files))

all_processed_data = [item for sublist in all_processed_data for item in sublist]

df_processed = pd.DataFrame(all_processed_data)
print("Muestra de datos procesados combinados:")
print(df_processed.head())
print(f"Total de muestras: {len(df_processed)}")

# Inspección de outliers en 'normal' por sujeto
print("\nEstadísticas de 'normal' por sujeto (antes de normalización):")
for subject_id in df_processed['subject_id'].unique():
    subject_data = df_processed[df_processed['subject_id'] == subject_id]['normal']
    print(f"Sujeto {subject_id}: min={subject_data.min():.2f}, max={subject_data.max():.2f}, mean={subject_data.mean():.2f}, std={subject_data.std():.2f}")

# Process CGM Window and Other Features
cgm_columns = [f'cgm_{i}' for i in range(24)]
df_cgm = pd.DataFrame(df_processed['cgm_window'].tolist(), columns=cgm_columns, index=df_processed.index)
df_final = pd.concat([df_cgm, df_processed.drop(columns=['cgm_window'])], axis=1)

# Verificar valores NaN
print("Verificación de NaN en df_final:")
df_final = df_final.dropna()
print(df_final.isna().sum())

# Normalizar características
scaler_cgm = MinMaxScaler(feature_range=(0, 1))
scaler_other = StandardScaler()
scaler_y = MinMaxScaler(feature_range=(0, 1))  # Normalizar etiquetas

# Normalizar CGM
X_cgm = scaler_cgm.fit_transform(df_final[cgm_columns])
X_cgm = X_cgm.reshape(X_cgm.shape[0], X_cgm.shape[1], 1)

# Extraer subject_id como tensor separado (sin normalizar, para embeddings)
X_subject = df_final['subject_id'].values

# Normalizar otras características (excluyendo subject_id)
other_features = ['carbInput', 'bgInput', 'insulinOnBoard', 'insulinCarbRatio', 
                  'insulinSensitivityFactor', 'hour_of_day']
X_other = scaler_other.fit_transform(df_final[other_features])

# Normalizar etiquetas
y = df_final['normal'].values.reshape(-1, 1)
y = scaler_y.fit_transform(y).flatten()

# Verificar NaN
print("NaN en X_cgm:", np.isnan(X_cgm).sum())
print("NaN en X_other:", np.isnan(X_other).sum())
print("NaN en X_subject:", np.isnan(X_subject).sum())
print("NaN en y:", np.isnan(y).sum())
if np.isnan(X_cgm).sum() > 0 or np.isnan(X_other).sum() > 0 or np.isnan(X_subject).sum() > 0 or np.isnan(y).sum() > 0:
    raise ValueError("Valores NaN detectados en X_cgm, X_other, X_subject o y")

# División por sujeto
subject_ids = df_final['subject_id'].unique()
train_subjects, temp_subjects = train_test_split(subject_ids, test_size=0.2, random_state=42)
val_subjects, test_subjects = train_test_split(temp_subjects, test_size=0.5, random_state=42)

train_mask = df_final['subject_id'].isin(train_subjects)
val_mask = df_final['subject_id'].isin(val_subjects)
test_mask = df_final['subject_id'].isin(test_subjects)

X_cgm_train, X_cgm_val, X_cgm_test = X_cgm[train_mask], X_cgm[val_mask], X_cgm[test_mask]
X_other_train, X_other_val, X_other_test = X_other[train_mask], X_other[val_mask], X_other[test_mask]
X_subject_train, X_subject_val, X_subject_test = X_subject[train_mask], X_subject[val_mask], X_subject[test_mask]
y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]
subject_test = df_final[test_mask]['subject_id'].values

print(f"Entrenamiento CGM: {X_cgm_train.shape}, Validación CGM: {X_cgm_val.shape}, Prueba CGM: {X_cgm_test.shape}")
print(f"Entrenamiento Otros: {X_other_train.shape}, Validación Otros: {X_other_val.shape}, Prueba Otros: {X_other_test.shape}")
print(f"Entrenamiento Subject: {X_subject_train.shape}, Validación Subject: {X_subject_val.shape}, Prueba Subject: {X_subject_test.shape}")
print(f"Sujetos de prueba: {test_subjects}")

# Convert Data to PyTorch Tensors
X_cgm_train = torch.tensor(X_cgm_train, dtype=torch.float32).to(device)
X_cgm_val = torch.tensor(X_cgm_val, dtype=torch.float32).to(device)
X_cgm_test = torch.tensor(X_cgm_test, dtype=torch.float32).to(device)
X_other_train = torch.tensor(X_other_train, dtype=torch.float32).to(device)
X_other_val = torch.tensor(X_other_val, dtype=torch.float32).to(device)
X_other_test = torch.tensor(X_other_test, dtype=torch.float32).to(device)
X_subject_train = torch.tensor(X_subject_train, dtype=torch.long).to(device)
X_subject_val = torch.tensor(X_subject_val, dtype=torch.long).to(device)
X_subject_test = torch.tensor(X_subject_test, dtype=torch.long).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

# Create DataLoaders (incluyendo subject_id)
train_dataset = TensorDataset(X_cgm_train, X_other_train, X_subject_train, y_train)
val_dataset = TensorDataset(X_cgm_val, X_other_val, X_subject_val, y_val)
test_dataset = TensorDataset(X_cgm_test, X_other_test, X_subject_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# %% [markdown]
# ### Define Models in PyTorch

# Enhanced LSTM Model
class EnhancedLSTM(nn.Module):
    def __init__(self, num_subjects, embedding_dim=8):
        super(EnhancedLSTM, self).__init__()
        # Embedding para subject_id
        self.subject_embedding = nn.Embedding(num_subjects, embedding_dim)
        # Simplificado a 2 capas
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=128, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.4)  # Aumentado dropout
        # other_input tiene 6 características + embedding_dim
        self.concat_dense = nn.Linear(64 + 6 + embedding_dim, 128)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.4)  # Aumentado dropout
        self.output_layer = nn.Linear(128, 1)

    def forward(self, cgm_input, other_input, subject_ids):
        # Obtener embeddings para subject_id
        subject_embed = self.subject_embedding(subject_ids)
        lstm_out, _ = self.lstm1(cgm_input)
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.batch_norm1(lstm_out)
        lstm_out = self.dropout1(lstm_out)
        # Concatenar lstm_out, other_input y subject_embed
        combined = torch.cat((lstm_out, other_input, subject_embed), dim=1)
        dense_out = torch.relu(self.concat_dense(combined))
        dense_out = self.batch_norm2(dense_out)
        dense_out = self.dropout2(dense_out)
        output = self.output_layer(dense_out)
        return output

# Transformer with TCN Model
class TCNTransformer(nn.Module):
    def __init__(self, num_subjects, embedding_dim=8):
        super(TCNTransformer, self).__init__()
        # Embedding para subject_id
        self.subject_embedding = nn.Embedding(num_subjects, embedding_dim)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding='same')
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding='same')
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding='same')
        self.ln1 = nn.LayerNorm(128)
        # Añadido batch_first=True
        self.transformer = nn.TransformerEncoderLayer(d_model=128, nhead=8, dropout=0.2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer, num_layers=2)
        self.ln2 = nn.LayerNorm(128)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        # other_input tiene 6 características + embedding_dim
        self.concat_dense1 = nn.Linear(128 + 6 + embedding_dim, 256)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.4)  # Aumentado dropout
        self.dense2 = nn.Linear(256, 128)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.4)  # Aumentado dropout
        self.output_layer = nn.Linear(128, 1)

    def forward(self, cgm_input, other_input, subject_ids):
        # Obtener embeddings para subject_id
        subject_embed = self.subject_embedding(subject_ids)
        cgm_input = cgm_input.permute(0, 2, 1)
        tcn_out = torch.relu(self.conv1(cgm_input))
        tcn_out = self.pool1(tcn_out)
        tcn_out = torch.relu(self.conv2(tcn_out))
        tcn_out = self.pool2(tcn_out)
        tcn_out = torch.relu(self.conv3(tcn_out))
        tcn_out = tcn_out.permute(0, 2, 1)
        tcn_out = self.ln1(tcn_out)
        transformer_out = self.transformer_encoder(tcn_out)
        transformer_out = transformer_out + tcn_out
        transformer_out = self.ln2(transformer_out)
        transformer_out = transformer_out.permute(0, 2, 1)
        pooled = self.global_pool(transformer_out).squeeze(-1)
        # Concatenar pooled, other_input y subject_embed
        combined = torch.cat((pooled, other_input, subject_embed), dim=1)
        dense_out = torch.relu(self.concat_dense1(combined))
        dense_out = self.batch_norm1(dense_out)
        dense_out = self.dropout1(dense_out)
        dense_out = torch.relu(self.dense2(dense_out))
        dense_out = self.batch_norm2(dense_out)
        dense_out = self.dropout2(dense_out)
        output = self.output_layer(dense_out)
        return output

# Custom Loss Function
def custom_mse(y_true, y_pred):
    error = y_true - y_pred
    overprediction_penalty = torch.where(error < 0, 2 * error**2, error**2)
    return torch.mean(overprediction_penalty)

# Training Loop
def train_model(model, train_loader, val_loader, epochs=200, patience=30):
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)  # Reducido lr
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for cgm_batch, other_batch, subject_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(cgm_batch, other_batch, subject_batch).squeeze()
            loss = custom_mse(y_batch, y_pred)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            train_loss += loss.item() * len(y_batch)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for cgm_batch, other_batch, subject_batch, y_batch in val_loader:
                y_pred = model(cgm_batch, other_batch, subject_batch).squeeze()
                loss = custom_mse(y_batch, y_pred)
                val_loss += loss.item() * len(y_batch)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    model.load_state_dict(best_model_state)
    return train_losses, val_losses

# Número total de sujetos para embeddings
num_subjects = len(subject_ids)

# Initialize models
model_lstm = EnhancedLSTM(num_subjects=num_subjects, embedding_dim=8).to(device)
model_tcn = TCNTransformer(num_subjects=num_subjects, embedding_dim=8).to(device)

# Train models
print("\nEntrenando LSTM Mejorado...")
lstm_train_losses, lstm_val_losses = train_model(model_lstm, train_loader, val_loader)

print("\nEntrenando Transformer con TCN...")
tcn_train_losses, tcn_val_losses = train_model(model_tcn, train_loader, val_loader)

# %% [markdown]
# ### Plot Training History
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(lstm_train_losses, label='Pérdida de Entrenamiento (LSTM)')
plt.plot(lstm_val_losses, label='Pérdida de Validación (LSTM)')
plt.xlabel('Época')
plt.ylabel('Pérdida MSE Personalizada')
plt.legend()
plt.title('Historial de Entrenamiento - LSTM Mejorado')

plt.subplot(1, 2, 2)
plt.plot(tcn_train_losses, label='Pérdida de Entrenamiento (TCN)')
plt.plot(tcn_val_losses, label='Pérdida de Validación (TCN)')
plt.xlabel('Época')
plt.ylabel('Pérdida MSE Personalizada')
plt.legend()
plt.title('Historial de Entrenamiento - Transformer con TCN')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Evaluate Models
print("\n=== Paso 3: Evaluar los modelos ===")

def evaluate_model(model, loader):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for cgm_batch, other_batch, subject_batch, y_batch in loader:
            y_pred = model(cgm_batch, other_batch, subject_batch).squeeze()
            predictions.append(y_pred.cpu().numpy())
            targets.append(y_batch.cpu().numpy())
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    # Desnormalizar predicciones y etiquetas
    predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
    targets = scaler_y.inverse_transform(targets.reshape(-1, 1)).flatten()
    return predictions, targets

# Evaluate LSTM
y_pred_lstm, y_test_lstm = evaluate_model(model_lstm, test_loader)
mae_lstm = mean_absolute_error(y_test_lstm, y_pred_lstm)
rmse_lstm = np.sqrt(mean_squared_error(y_test_lstm, y_pred_lstm))
r2_lstm = r2_score(y_test_lstm, y_pred_lstm)
print(f"LSTM Mejorado - MAE general: {mae_lstm:.2f} unidades")
print(f"LSTM Mejorado - RMSE general: {rmse_lstm:.2f} unidades")
print(f"LSTM Mejorado - R² general: {r2_lstm:.2f}")

# Evaluate TCN
y_pred_tcn, y_test_tcn = evaluate_model(model_tcn, test_loader)
mae_tcn = mean_absolute_error(y_test_tcn, y_pred_tcn)
rmse_tcn = np.sqrt(mean_squared_error(y_test_tcn, y_pred_tcn))
r2_tcn = r2_score(y_test_tcn, y_pred_tcn)
print(f"Transformer con TCN - MAE general: {mae_tcn:.2f} unidades")
print(f"Transformer con TCN - RMSE general: {rmse_tcn:.2f} unidades")
print(f"Transformer con TCN - R² general: {r2_tcn:.2f}")

# Rule-based prediction
def rule_based_prediction(X_other, target_bg=100):
    inverse_transformed = scaler_other.inverse_transform(X_other.cpu().numpy())
    carb_input = inverse_transformed[:, 0]
    bg_input = inverse_transformed[:, 1]
    icr = inverse_transformed[:, 3]
    isf = inverse_transformed[:, 4]
    
    icr = np.where(icr == 0, 1e-6, icr)
    isf = np.where(isf == 0, 1e-6, isf)
    
    carb_component = np.divide(carb_input, icr, out=np.zeros_like(carb_input), where=icr!=0)
    bg_component = np.divide(bg_input - target_bg, isf, out=np.zeros_like(bg_input), where=isf!=0)
    prediction = carb_component + bg_component
    
    prediction = np.clip(prediction, 0, 30)
    
    return prediction

y_rule = rule_based_prediction(X_other_test)
print("Infinities in predictions:", np.isinf(y_rule).sum())
print("NaNs in predictions:", np.isnan(y_rule).sum())

# Desnormalizar y_test para la evaluación del modelo basado en reglas
y_test_denorm = scaler_y.inverse_transform(y_test.cpu().numpy().reshape(-1, 1)).flatten()

mae_rule = mean_absolute_error(y_test_denorm, y_rule)
rmse_rule = np.sqrt(mean_squared_error(y_test_denorm, y_rule))
r2_rule = r2_score(y_test_denorm, y_rule)
print(f"MAE basado en reglas: {mae_rule:.2f} unidades")
print(f"RMSE basado en reglas: {rmse_rule:.2f} unidades")
print(f"R² basado en reglas: {r2_rule:.2f}")

# %% [markdown]
# ### Metrics per Subject
print("\nRendimiento por sujeto:")
for subject_id in test_subjects:
    mask = subject_test == subject_id
    y_test_sub = y_test_denorm[mask]
    y_pred_lstm_sub = y_pred_lstm[mask]
    y_pred_tcn_sub = y_pred_tcn[mask]
    y_rule_sub = y_rule[mask]
    if len(y_test_sub) > 0:
        mae_lstm_sub = mean_absolute_error(y_test_sub, y_pred_lstm_sub)
        rmse_lstm_sub = np.sqrt(mean_squared_error(y_test_sub, y_pred_lstm_sub))
        r2_lstm_sub = r2_score(y_test_sub, y_pred_lstm_sub)
        mae_tcn_sub = mean_absolute_error(y_test_sub, y_pred_tcn_sub)
        rmse_tcn_sub = np.sqrt(mean_squared_error(y_test_sub, y_pred_tcn_sub))
        r2_tcn_sub = r2_score(y_test_sub, y_pred_tcn_sub)
        mae_rule_sub = mean_absolute_error(y_test_sub, y_rule_sub)
        print(f"Sujeto {subject_id}: LSTM MAE={mae_lstm_sub:.2f}, RMSE={rmse_lstm_sub:.2f}, R²={r2_lstm_sub:.2f}, "
              f"TCN MAE={mae_tcn_sub:.2f}, RMSE={rmse_tcn_sub:.2f}, R²={r2_tcn_sub:.2f}, "
              f"MAE basado en reglas={mae_rule_sub:.2f}")

# %% [markdown]
# ### Visualization
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.scatter(y_test_denorm, y_pred_lstm, label='Predicciones LSTM', alpha=0.5, color='blue')
plt.scatter(y_test_denorm, y_pred_tcn, label='Predicciones TCN', alpha=0.5, color='green')
plt.scatter(y_test_denorm, y_rule, label='Basado en Reglas', alpha=0.5, color='orange')
plt.plot([0, 15], [0, 15], 'r--')
plt.xlabel('Dosis Real (unidades)')
plt.ylabel('Dosis Predicha (unidades)')
plt.legend()
plt.title('Predicciones vs Real (Todos los Sujetos)')

plt.subplot(2, 2, 2)
plt.hist(y_test_denorm - y_pred_lstm, bins=20, label='Residuos LSTM', alpha=0.5, color='blue')
plt.hist(y_test_denorm - y_pred_tcn, bins=20, label='Residuos TCN', alpha=0.5, color='green')
plt.hist(y_test_denorm - y_rule, bins=20, label='Residuos Basados en Reglas', alpha=0.5, color='orange')
plt.xlabel('Residuo (unidades)')
plt.ylabel('Frecuencia')
plt.legend()
plt.title('Distribución de Residuos (Todos los Sujetos)')

plt.subplot(2, 2, 3)
mae_lstm_subjects = [mean_absolute_error(y_test_denorm[subject_test == sid], y_pred_lstm[subject_test == sid]) 
                    for sid in test_subjects if len(y_test_denorm[subject_test == sid]) > 0]
mae_tcn_subjects = [mean_absolute_error(y_test_denorm[subject_test == sid], y_pred_tcn[subject_test == sid]) 
                   for sid in test_subjects if len(y_test_denorm[subject_test == sid]) > 0]
mae_rule_subjects = [mean_absolute_error(y_test_denorm[subject_test == sid], y_rule[subject_test == sid]) 
                    for sid in test_subjects if len(y_test_denorm[subject_test == sid]) > 0]
plt.bar(np.arange(len(test_subjects)) - 0.2, mae_lstm_subjects, width=0.2, label='LSTM', color='blue')
plt.bar(np.arange(len(test_subjects)), mae_tcn_subjects, width=0.2, label='TCN', color='green')
plt.bar(np.arange(len(test_subjects)) + 0.2, mae_rule_subjects, width=0.2, label='Reglas', color='orange')
plt.xlabel('Sujeto')
plt.ylabel('MAE (unidades)')
plt.xticks(np.arange(len(test_subjects)), test_subjects)
plt.legend()
plt.title('Comparación de MAE por Sujeto')

plt.subplot(2, 2, 4)
r2_lstm_subjects = [r2_score(y_test_denorm[subject_test == sid], y_pred_lstm[subject_test == sid]) 
                   for sid in test_subjects if len(y_test_denorm[subject_test == sid]) > 0]
r2_tcn_subjects = [r2_score(y_test_denorm[subject_test == sid], y_pred_tcn[subject_test == sid]) 
                  for sid in test_subjects if len(y_test_denorm[subject_test == sid]) > 0]
r2_rule_subjects = [r2_score(y_test_denorm[subject_test == sid], y_rule[subject_test == sid]) 
                   for sid in test_subjects if len(y_test_denorm[subject_test == sid]) > 0]
plt.bar(np.arange(len(test_subjects)) - 0.2, r2_lstm_subjects, width=0.2, label='LSTM', color='blue')
plt.bar(np.arange(len(test_subjects)), r2_tcn_subjects, width=0.2, label='TCN', color='green')
plt.bar(np.arange(len(test_subjects)) + 0.2, r2_rule_subjects, width=0.2, label='Reglas', color='orange')
plt.xlabel('Sujeto')
plt.ylabel('R²')
plt.xticks(np.arange(len(test_subjects)), test_subjects)
plt.legend()
plt.title('Comparación de R² por Sujeto')

plt.tight_layout()
plt.show()
# %%
