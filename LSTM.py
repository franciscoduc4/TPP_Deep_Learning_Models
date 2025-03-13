# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, Concatenate, BatchNormalization
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
            # Extraer hora del día como nueva característica
            hour_of_day = bolus_time.hour / 23.0  # Normalizar entre 0 y 1
            # Calcular insulinSensitivityFactor personalizado
            bg_input = row['bgInput'] if pd.notna(row['bgInput']) else cgm_window[-1]
            normal = row['normal'] if pd.notna(row['normal']) else 0.0
            isf_custom = 50.0  # Valor por defecto
            if normal > 0 and bg_input > 100:  # Evitar divisiones por cero
                isf_custom = (bg_input - 100) / normal  # Aproximación simple
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
# Paso 2: Establecer un modelo LSTM mejorado
print("\n=== Paso 2: Establecer un modelo LSTM mejorado ===")

# Definir las entradas
cgm_input = Input(shape=(24, 1), name='cgm_input')
other_input = Input(shape=(7,), name='other_input')  # Actualizado a 7 por hour_of_day

# Capas LSTM apiladas para procesar CGM
lstm_out = LSTM(128, return_sequences=True)(cgm_input)  # Primera capa con 128 unidades
lstm_out = LSTM(64, return_sequences=False)(lstm_out)   # Segunda capa con 64 unidades
lstm_out = BatchNormalization()(lstm_out)              # Normalización por lotes
lstm_out = Dropout(0.2)(lstm_out)

# Combinar con otras características
combined = Concatenate()([lstm_out, other_input])

# Capas densas
dense = Dense(64, activation='relu')(combined)        # Aumentar a 64 unidades
dense = BatchNormalization()(dense)
dense = Dropout(0.2)(dense)
output = Dense(1, activation='linear')(dense)

# Definir el modelo
model = Model(inputs=[cgm_input, other_input], outputs=output)

# Función de pérdida personalizada para penalizar sobrepredicciones
def custom_mse(y_true, y_pred):
    error = y_true - y_pred
    overprediction_penalty = tf.where(error < 0, 2 * tf.square(error), tf.square(error))
    return tf.reduce_mean(overprediction_penalty)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=custom_mse)
model.summary()

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

# Entrenar modelo
history = model.fit(
    [X_cgm_train, X_other_train], y_train,
    validation_data=([X_cgm_val, X_other_val], y_val),
    epochs=100,
    batch_size=32,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
    verbose=1
)

# %%
# Graficar historial de entrenamiento
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de Validación')
plt.xlabel('Época')
plt.ylabel('Pérdida MSE Personalizada')
plt.legend()
plt.title('Historial de Entrenamiento (Todos los Sujetos) - LSTM Mejorado')
plt.show()

# %%
# Paso 3: Evaluar el modelo LSTM mejorado
print("\n=== Paso 3: Evaluar el modelo LSTM mejorado ===")

# Predicciones
y_pred = model.predict([X_cgm_test, X_other_test]).flatten()

# Métricas generales
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"MAE general: {mae:.2f} unidades")
print(f"RMSE general: {rmse:.2f} unidades")
print(f"R² general: {r2:.2f}")

# %%
# Línea base basada en reglas
def rule_based_prediction(X_other, target_bg=100):
    # Desnormalizar todas las características
    inverse_transformed = scaler_other.inverse_transform(X_other)
    carb_input = inverse_transformed[:, 0]
    bg_input = inverse_transformed[:, 1]
    icr = inverse_transformed[:, 3]
    isf = inverse_transformed[:, 4]
    
    # Prevenir divisiones por cero
    icr = np.where(icr == 0, 1e-6, icr)
    isf = np.where(isf == 0, 1e-6, isf)
    
    # Calcular predicción
    carb_component = np.divide(carb_input, icr, out=np.zeros_like(carb_input), where=icr!=0)
    bg_component = np.divide(bg_input - target_bg, isf, out=np.zeros_like(bg_input), where=isf!=0)
    prediction = carb_component + bg_component
    
    # Limitar a un rango razonable
    prediction = np.clip(prediction, 0, 30)
    
    return prediction

# Predicciones y verificación
y_rule = rule_based_prediction(X_other_test)
print("Infinities in predictions:", np.isinf(y_rule).sum())
print("NaNs in predictions:", np.isnan(y_rule).sum())

# Métricas
mae_rule = mean_absolute_error(y_test, y_rule)
rmse_rule = np.sqrt(mean_squared_error(y_test, y_rule))
r2_rule = r2_score(y_test, y_rule)
print(f"MAE basado en reglas: {mae_rule:.2f} unidades")
print(f"RMSE basado en reglas: {rmse_rule:.2f} unidades")
print(f"R² basado en reglas: {r2_rule:.2f}")

# %%
# Métricas por sujeto
print("\nRendimiento por sujeto:")
for subject_id in test_subjects:
    mask = subject_test == subject_id
    y_test_sub = y_test[mask]
    y_pred_sub = y_pred[mask]
    y_rule_sub = y_rule[mask]
    if len(y_test_sub) > 0:
        mae_sub = mean_absolute_error(y_test_sub, y_pred_sub)
        rmse_sub = np.sqrt(mean_squared_error(y_test_sub, y_pred_sub))
        r2_sub = r2_score(y_test_sub, y_pred_sub)
        mae_rule_sub = mean_absolute_error(y_test_sub, y_rule_sub)
        print(f"Sujeto {subject_id}: LSTM MAE={mae_sub:.2f}, RMSE={rmse_sub:.2f}, R²={r2_sub:.2f}, MAE basado en reglas={mae_rule_sub:.2f}")

# %%
# Visualización
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, label='Predicciones LSTM', alpha=0.5)
plt.scatter(y_test, y_rule, label='Basado en Reglas', alpha=0.5)
plt.plot([0, 15], [0, 15], 'r--')
plt.xlabel('Dosis Real (unidades)')
plt.ylabel('Dosis Predicha (unidades)')
plt.legend()
plt.title('Predicciones vs Real (Todos los Sujetos) - LSTM Mejorado')

plt.subplot(1, 2, 2)
plt.hist(y_test - y_pred, bins=20, label='Residuos LSTM', alpha=0.5)
plt.hist(y_test - y_rule, bins=20, label='Residuos Basados en Reglas', alpha=0.5)
plt.xlabel('Residuo (unidades)')
plt.ylabel('Frecuencia')
plt.legend()
plt.title('Distribución de Residuos (Todos los Sujetos) - LSTM Mejorado')
plt.tight_layout()
plt.show()
# %%