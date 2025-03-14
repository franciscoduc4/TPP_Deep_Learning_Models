# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, LSTM, Input, Concatenate, BatchNormalization,
    Conv1D, MaxPooling1D, LayerNormalization, MultiHeadAttention,
    Add, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import os
from joblib import Parallel, delayed
from datetime import timedelta

# Configuración de matplotlib
import matplotlib
matplotlib.use('TkAgg')  # o 'Agg' para no interactivo
import matplotlib.pyplot as plt

# %%
# Los sujetos están en la raíz y cada archivo comienza con "Subject"
subject_files = [f for f in os.listdir() if os.path.isfile(f) and f.startswith("Subject")]
print(f"Total de sujetos: {len(subject_files)}")

# %%
# Paso 1: Preprocesar datos para todos los sujetos
def get_cgm_window(bolus_time, cgm_df, window_hours=2, interval_minutes=5):
    window_start = bolus_time - timedelta(hours=window_hours)
    window = cgm_df[(cgm_df['date'] >= window_start) & (cgm_df['date'] <= bolus_time)]
    window = window.sort_values('date').tail(24)  # Últimas 24 lecturas (~2 horas)
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
            hour_of_day = bolus_time.hour / 23.0  # Normalizar entre 0 y 1
            bg_input = row['bgInput'] if pd.notna(row['bgInput']) else cgm_window[-1]
            normal = row['normal'] if pd.notna(row['normal']) else 0.0
            isf_custom = 50.0
            if normal > 0 and bg_input > 100:
                isf_custom = (bg_input - 100) / normal
            features = {
                'subject_id': idx,
                'cgm_window': cgm_window,
                'carbInput': row['carbInput'] if pd.notna(row['carbInput']) else 0.0,
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
subject_folder = os.getcwd()
all_processed_data = Parallel(n_jobs=-1)(delayed(process_subject)(os.path.join(subject_folder, f), idx) 
                                        for idx, f in enumerate(subject_files))

all_processed_data = [item for sublist in all_processed_data for item in sublist]

df_processed = pd.DataFrame(all_processed_data)
print("Muestra de datos procesados combinados:")
print(df_processed.head())
print(f"Total de muestras: {len(df_processed)}")

# %%
# Dividir ventana CGM y otras características
cgm_columns = [f'cgm_{i}' for i in range(24)]
df_cgm = pd.DataFrame(df_processed['cgm_window'].tolist(), columns=cgm_columns, index=df_processed.index)
df_final = pd.concat([df_cgm, df_processed.drop(columns=['cgm_window'])], axis=1)

# Verificar valores NaN
print("Verificación de NaN en df_final:")
df_final = df_final.dropna()
print(df_final.isna().sum())

# %%
# Normalizar características
scaler_cgm = MinMaxScaler(feature_range=(0, 1))
scaler_other = StandardScaler()

# Normalizar CGM
X_cgm = scaler_cgm.fit_transform(df_final[cgm_columns])
X_cgm = X_cgm.reshape(X_cgm.shape[0], X_cgm.shape[1], 1)

# Normalizar otras características (incluyendo hour_of_day)
other_features = ['carbInput', 'bgInput', 'insulinOnBoard', 'insulinCarbRatio', 
                  'insulinSensitivityFactor', 'subject_id', 'hour_of_day']
X_other = scaler_other.fit_transform(df_final[other_features])

# Etiquetas
y = df_final['normal'].values

# Verificar NaN
print("NaN en X_cgm:", np.isnan(X_cgm).sum())
print("NaN en X_other:", np.isnan(X_other).sum())
print("NaN en y:", np.isnan(y).sum())
if np.isnan(X_cgm).sum() > 0 or np.isnan(X_other).sum() > 0 or np.isnan(y).sum() > 0:
    raise ValueError("Valores NaN detectados en X_cgm, X_other o y")

# %%
# División por sujeto
subject_ids = df_final['subject_id'].unique()
train_subjects, temp_subjects = train_test_split(subject_ids, test_size=0.2, random_state=42)
val_subjects, test_subjects = train_test_split(temp_subjects, test_size=0.5, random_state=42)

train_mask = df_final['subject_id'].isin(train_subjects)
val_mask = df_final['subject_id'].isin(val_subjects)
test_mask = df_final['subject_id'].isin(test_subjects)

X_cgm_train, X_cgm_val, X_cgm_test = X_cgm[train_mask], X_cgm[val_mask], X_cgm[test_mask]
X_other_train, X_other_val, X_other_test = X_other[train_mask], X_other[val_mask], X_other[test_mask]
y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]
subject_test = df_final[test_mask]['subject_id'].values

print(f"Entrenamiento CGM: {X_cgm_train.shape}, Validación CGM: {X_cgm_val.shape}, Prueba CGM: {X_cgm_test.shape}")
print(f"Entrenamiento Otros: {X_other_train.shape}, Validación Otros: {X_other_val.shape}, Prueba Otros: {X_other_test.shape}")
print(f"Sujetos de prueba: {test_subjects}")

# %%
# Paso 2: Establecer modelos (LSTM Mejorado y Transformer con TCN)

# Modelo LSTM Mejorado
print("\n=== Paso 2: Establecer un modelo LSTM mejorado ===")
cgm_input_lstm = Input(shape=(24, 1), name='cgm_input_lstm')
other_input_lstm = Input(shape=(7,), name='other_input_lstm')

lstm_out = LSTM(128, return_sequences=True)(cgm_input_lstm)
lstm_out = LSTM(64, return_sequences=False)(lstm_out)
lstm_out = BatchNormalization()(lstm_out)
lstm_out = Dropout(0.2)(lstm_out)

combined_lstm = Concatenate()([lstm_out, other_input_lstm])
dense_lstm = Dense(64, activation='relu')(combined_lstm)
dense_lstm = BatchNormalization()(dense_lstm)
dense_lstm = Dropout(0.2)(dense_lstm)
output_lstm = Dense(1, activation='linear')(dense_lstm)

model_lstm = Model(inputs=[cgm_input_lstm, other_input_lstm], outputs=output_lstm)

# Función de pérdida personalizada
def custom_mse(y_true, y_pred):
    error = y_true - y_pred
    overprediction_penalty = tf.where(error < 0, 2 * tf.square(error), tf.square(error))
    return tf.reduce_mean(overprediction_penalty)

model_lstm.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=custom_mse)
model_lstm.summary()

# Modelo Transformer con TCN
print("\n=== Paso 2: Establecer un modelo Transformer con TCN ===")
cgm_input_tcn = Input(shape=(24, 1), name='cgm_input_tcn')
other_input_tcn = Input(shape=(7,), name='other_input_tcn')

# TCN para preprocesar las lecturas de CGM
tcn_out = Conv1D(filters=64, kernel_size=3, padding='causal', activation='relu')(cgm_input_tcn)
tcn_out = MaxPooling1D(pool_size=2)(tcn_out)
tcn_out = Conv1D(filters=64, kernel_size=3, padding='causal', activation='relu')(tcn_out)
tcn_out = MaxPooling1D(pool_size=2)(tcn_out)
tcn_out = LayerNormalization()(tcn_out)

# Transformer con atención multi-cabeza
attention_output = MultiHeadAttention(key_dim=64, num_heads=4)(tcn_out, tcn_out)
attention_output = Dropout(0.2)(attention_output)
tcn_out = Add()([tcn_out, attention_output])  # Conexión residual
tcn_out = LayerNormalization()(tcn_out)

# Flatten para reducir dimensiones
tcn_out = GlobalAveragePooling1D()(tcn_out)  # Reemplaza reduce_mean

# Combinar con otras características
combined_tcn = Concatenate()([tcn_out, other_input_tcn])

# Capas densas finales
dense_tcn = Dense(128, activation='relu')(combined_tcn)
dense_tcn = BatchNormalization()(dense_tcn)
dense_tcn = Dropout(0.2)(dense_tcn)
dense_tcn = Dense(64, activation='relu')(dense_tcn)
dense_tcn = BatchNormalization()(dense_tcn)
dense_tcn = Dropout(0.2)(dense_tcn)
output_tcn = Dense(1, activation='linear')(dense_tcn)

model_tcn = Model(inputs=[cgm_input_tcn, other_input_tcn], outputs=output_tcn)
model_tcn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=custom_mse)
model_tcn.summary()

# %%
# Convertir datos a float32
X_cgm_train = np.array(X_cgm_train, dtype=np.float32)
X_cgm_val = np.array(X_cgm_val, dtype=np.float32)
X_cgm_test = np.array(X_cgm_test, dtype=np.float32)
X_other_train = np.array(X_other_train, dtype=np.float32)
X_other_val = np.array(X_other_val, dtype=np.float32)
X_other_test = np.array(X_other_test, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)
y_val = np.array(y_val, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)

# Verificar formas y tipos de datos
print("Forma de X_cgm_train:", X_cgm_train.shape, "dtype:", X_cgm_train.dtype)
print("Forma de X_cgm_val:", X_cgm_val.shape, "dtype:", X_cgm_val.dtype)
print("Forma de X_other_train:", X_other_train.shape, "dtype:", X_other_train.dtype)
print("Forma de X_other_val:", X_other_val.shape, "dtype:", X_other_val.dtype)
print("Forma de y_train:", y_train.shape, "dtype:", y_train.dtype)
print("Forma de y_val:", y_val.shape, "dtype:", y_val.dtype)

# Entrenar modelos
print("\nEntrenando LSTM Mejorado...")
history_lstm = model_lstm.fit(
    [X_cgm_train, X_other_train], y_train,
    validation_data=([X_cgm_val, X_other_val], y_val),
    epochs=100,
    batch_size=32,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
    verbose=1
)

print("\nEntrenando Transformer con TCN...")
history_tcn = model_tcn.fit(
    [X_cgm_train, X_other_train], y_train,
    validation_data=([X_cgm_val, X_other_val], y_val),
    epochs=100,
    batch_size=32,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
    verbose=1
)

# %%
# Graficar historial de entrenamiento para ambos modelos
plt.figure(figsize=(12, 5))

# LSTM Mejorado
plt.subplot(1, 2, 1)
plt.plot(history_lstm.history['loss'], label='Pérdida de Entrenamiento (LSTM)')
plt.plot(history_lstm.history['val_loss'], label='Pérdida de Validación (LSTM)')
plt.xlabel('Época')
plt.ylabel('Pérdida MSE Personalizada')
plt.legend()
plt.title('Historial de Entrenamiento - LSTM Mejorado')

# Transformer con TCN
plt.subplot(1, 2, 2)
plt.plot(history_tcn.history['loss'], label='Pérdida de Entrenamiento (TCN)')
plt.plot(history_tcn.history['val_loss'], label='Pérdida de Validación (TCN)')
plt.xlabel('Época')
plt.ylabel('Pérdida MSE Personalizada')
plt.legend()
plt.title('Historial de Entrenamiento - Transformer con TCN')

plt.tight_layout()
plt.show()

# %%
# Paso 3: Evaluar ambos modelos
print("\n=== Paso 3: Evaluar los modelos ===")

# Predicciones LSTM
y_pred_lstm = model_lstm.predict([X_cgm_test, X_other_test]).flatten()
mae_lstm = mean_absolute_error(y_test, y_pred_lstm)
rmse_lstm = np.sqrt(mean_squared_error(y_test, y_pred_lstm))
r2_lstm = r2_score(y_test, y_pred_lstm)
print(f"LSTM Mejorado - MAE general: {mae_lstm:.2f} unidades")
print(f"LSTM Mejorado - RMSE general: {rmse_lstm:.2f} unidades")
print(f"LSTM Mejorado - R² general: {r2_lstm:.2f}")

# Predicciones Transformer con TCN
y_pred_tcn = model_tcn.predict([X_cgm_test, X_other_test]).flatten()
mae_tcn = mean_absolute_error(y_test, y_pred_tcn)
rmse_tcn = np.sqrt(mean_squared_error(y_test, y_pred_tcn))
r2_tcn = r2_score(y_test, y_pred_tcn)
print(f"Transformer con TCN - MAE general: {mae_tcn:.2f} unidades")
print(f"Transformer con TCN - RMSE general: {rmse_tcn:.2f} unidades")
print(f"Transformer con TCN - R² general: {r2_tcn:.2f}")

# %%
# Línea base basada en reglas
def rule_based_prediction(X_other, target_bg=100):
    inverse_transformed = scaler_other.inverse_transform(X_other)
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

mae_rule = mean_absolute_error(y_test, y_rule)
rmse_rule = np.sqrt(mean_squared_error(y_test, y_rule))
r2_rule = r2_score(y_test, y_rule)
print(f"MAE basado en reglas: {mae_rule:.2f} unidades")
print(f"RMSE basado en reglas: {rmse_rule:.2f} unidades")
print(f"R² basado en reglas: {r2_rule:.2f}")

# %%
# Métricas por sujeto para ambos modelos
print("\nRendimiento por sujeto:")
for subject_id in test_subjects:
    mask = subject_test == subject_id
    y_test_sub = y_test[mask]
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

# %%
# Visualización
plt.figure(figsize=(15, 10))

# Predicciones vs Real para ambos modelos
plt.subplot(2, 2, 1)
plt.scatter(y_test, y_pred_lstm, label='Predicciones LSTM', alpha=0.5, color='blue')
plt.scatter(y_test, y_pred_tcn, label='Predicciones TCN', alpha=0.5, color='green')
plt.scatter(y_test, y_rule, label='Basado en Reglas', alpha=0.5, color='orange')
plt.plot([0, 15], [0, 15], 'r--')
plt.xlabel('Dosis Real (unidades)')
plt.ylabel('Dosis Predicha (unidades)')
plt.legend()
plt.title('Predicciones vs Real (Todos los Sujetos)')

# Distribución de Residuos para ambos modelos
plt.subplot(2, 2, 2)
plt.hist(y_test - y_pred_lstm, bins=20, label='Residuos LSTM', alpha=0.5, color='blue')
plt.hist(y_test - y_pred_tcn, bins=20, label='Residuos TCN', alpha=0.5, color='green')
plt.hist(y_test - y_rule, bins=20, label='Residuos Basados en Reglas', alpha=0.5, color='orange')
plt.xlabel('Residuo (unidades)')
plt.ylabel('Frecuencia')
plt.legend()
plt.title('Distribución de Residuos (Todos los Sujetos)')

# Comparación de MAE por sujeto
plt.subplot(2, 2, 3)
mae_lstm_subjects = [mean_absolute_error(y_test[subject_test == sid], y_pred_lstm[subject_test == sid]) 
                    for sid in test_subjects if len(y_test[subject_test == sid]) > 0]
mae_tcn_subjects = [mean_absolute_error(y_test[subject_test == sid], y_pred_tcn[subject_test == sid]) 
                   for sid in test_subjects if len(y_test[subject_test == sid]) > 0]
mae_rule_subjects = [mean_absolute_error(y_test[subject_test == sid], y_rule[subject_test == sid]) 
                    for sid in test_subjects if len(y_test[subject_test == sid]) > 0]
plt.bar(np.arange(len(test_subjects)) - 0.2, mae_lstm_subjects, width=0.2, label='LSTM', color='blue')
plt.bar(np.arange(len(test_subjects)), mae_tcn_subjects, width=0.2, label='TCN', color='green')
plt.bar(np.arange(len(test_subjects)) + 0.2, mae_rule_subjects, width=0.2, label='Reglas', color='orange')
plt.xlabel('Sujeto')
plt.ylabel('MAE (unidades)')
plt.xticks(np.arange(len(test_subjects)), test_subjects)
plt.legend()
plt.title('Comparación de MAE por Sujeto')

# Comparación de R² por sujeto
plt.subplot(2, 2, 4)
r2_lstm_subjects = [r2_score(y_test[subject_test == sid], y_pred_lstm[subject_test == sid]) 
                   for sid in test_subjects if len(y_test[subject_test == sid]) > 0]
r2_tcn_subjects = [r2_score(y_test[subject_test == sid], y_pred_tcn[subject_test == sid]) 
                  for sid in test_subjects if len(y_test[subject_test == sid]) > 0]
r2_rule_subjects = [r2_score(y_test[subject_test == sid], y_rule[subject_test == sid]) 
                   for sid in test_subjects if len(y_test[subject_test == sid]) > 0]
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