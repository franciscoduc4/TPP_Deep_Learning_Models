# %% CELL: Required Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from joblib import Parallel, delayed
from datetime import timedelta
import time
from tqdm import tqdm
%matplotlib inline

# DRL-specific imports
import gym
import gymnasium as gym  # Use gymnasium instead of gym
from gymnasium import spaces  # Update to gymnasium.spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Global configuration
CONFIG = {
    "batch_size": 128,
    "window_hours": 2,
    "cap_normal": 30,
    "cap_bg": 300,
    "cap_iob": 5,
    "cap_carb": 150,
    "data_path": os.path.join(os.getcwd(), "subjects")
}

# %% CELL: Data Processing Functions
def get_cgm_window(bolus_time, cgm_df, window_hours=CONFIG["window_hours"]):
    """
    Obtiene una ventana de datos CGM alrededor del tiempo del bolo de insulina.
    
    Args:
        bolus_time: Tiempo del bolo de insulina
        cgm_df: DataFrame con datos CGM
        window_hours: Tamaño de la ventana en horas
    
    Returns:
        Array numpy con los últimos 24 valores CGM o None si no hay suficientes datos
    """
    window_start = bolus_time - timedelta(hours=window_hours)
    window = cgm_df[(cgm_df['date'] >= window_start) & (cgm_df['date'] <= bolus_time)]
    window = window.sort_values('date').tail(24)
    return window['mg/dl'].values if len(window) >= 24 else None

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
    for _, row in basal_df.iterrows():
        start_time = row['date']
        duration_hours = row['duration'] / (1000 * 3600)
        end_time = start_time + timedelta(hours=duration_hours)
        rate = row['rate'] if pd.notna(row['rate']) else 0.9
        rate = min(rate, 2.0)
        
        if start_time <= bolus_time <= end_time:
            time_since_start = (bolus_time - start_time).total_seconds() / 3600
            remaining = rate * (1 - (time_since_start / half_life_hours))
            iob += max(0, remaining)
            
    return min(iob, 5.0)

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

    non_zero_carbs = bolus_df[bolus_df['carbInput'] > 0]['carbInput']
    carb_median = non_zero_carbs.median() if not non_zero_carbs.empty else 10.0

    iob_values = [calculate_iob(row['date'], basal_df) for _, row in bolus_df.iterrows()]
    non_zero_iob = [iob for iob in iob_values if iob > 0]
    iob_median = np.median(non_zero_iob) if non_zero_iob else 0.5

    processed_data = []
    for _, row in tqdm(bolus_df.iterrows(), total=len(bolus_df), desc=f"Procesando {os.path.basename(subject_path)}", leave=False):
        bolus_time = row['date']
        cgm_window = get_cgm_window(bolus_time, cgm_df)
        
        if cgm_window is not None:
            iob = calculate_iob(bolus_time, basal_df)
            iob = iob_median if iob == 0 else iob
            hour_of_day = bolus_time.hour / 23.0
            bg_input = row['bgInput'] if pd.notna(row['bgInput']) else cgm_window[-1]
            
            normal = row['normal'] if pd.notna(row['normal']) else 0.0
            normal = np.clip(normal, 0, CONFIG["cap_normal"])
            
            bg_input = max(bg_input, 50.0)
            isf_custom = 50.0 if normal <= 0 else (bg_input - 100) / normal
            isf_custom = np.clip(isf_custom, 10, 100)
            
            bg_input = np.clip(bg_input, 0, CONFIG["cap_bg"])
            iob = np.clip(iob, 0, CONFIG["cap_iob"])
            carb_input = row['carbInput'] if pd.notna(row['carbInput']) else 0.0
            carb_input = carb_median if carb_input == 0 else carb_input
            carb_input = np.clip(carb_input, 0, CONFIG["cap_carb"])
            
            features = {
                'subject_id': idx,
                'cgm_window': cgm_window,
                'carbInput': carb_input,
                'bgInput': bg_input,
                'insulinCarbRatio': np.clip(row['insulinCarbRatio'] if pd.notna(row['insulinCarbRatio']) else 10.0, 5, 20),
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

    print("\nEstadísticas de 'normal' por sujeto (antes de normalización):")
    for subject_id in df_processed['subject_id'].unique():
        subject_data = df_processed[df_processed['subject_id'] == subject_id]['normal']
        print(f"Sujeto {subject_id}: min={subject_data.min():.2f}, max={subject_data.max():.2f}, mean={subject_data.mean():.2f}, std={subject_data.std():.2f}")

    df_processed['normal'] = np.log1p(df_processed['normal'])
    df_processed['carbInput'] = np.log1p(df_processed['carbInput'])
    df_processed['insulinOnBoard'] = np.log1p(df_processed['insulinOnBoard'])
    df_processed['bgInput'] = np.log1p(df_processed['bgInput'])
    df_processed['cgm_window'] = df_processed['cgm_window'].apply(lambda x: np.log1p(x))

    cgm_columns = [f'cgm_{i}' for i in range(24)]
    df_cgm = pd.DataFrame(df_processed['cgm_window'].tolist(), columns=cgm_columns, index=df_processed.index)
    df_final = pd.concat([df_cgm, df_processed.drop(columns=['cgm_window'])], axis=1)
    
    df_final = df_final.dropna()
    print("Verificación de NaN en df_final:")
    print(df_final.isna().sum())

    scaler_cgm = StandardScaler()
    scaler_other = StandardScaler()
    scaler_y = StandardScaler()

    X_cgm = scaler_cgm.fit_transform(df_final[cgm_columns]).reshape(-1, 24, 1)
    X_subject = df_final['subject_id'].values
    other_features = ['carbInput', 'bgInput', 'insulinOnBoard', 'insulinCarbRatio', 
                      'insulinSensitivityFactor', 'hour_of_day']
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

    elapsed_time = time.time() - start_time
    print(f"División de datos completa en {elapsed_time:.2f} segundos")
    return (X_cgm_train, X_cgm_val, X_cgm_test,
            X_other_train, X_other_val, X_other_test,
            X_subject_train, X_subject_val, X_subject_test,
            y_train, y_val, y_test, subject_test)

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
    
    X_other_np = X_other
    inverse_transformed = scaler_other.inverse_transform(X_other_np)
    carb_input, bg_input, icr, isf = (inverse_transformed[:, 0],
                                     inverse_transformed[:, 1],
                                     inverse_transformed[:, 3],
                                     inverse_transformed[:, 4])
    
    icr = np.where(icr == 0, 1e-6, icr)
    isf = np.where(isf == 0, 1e-6, isf)
    
    carb_component = np.divide(carb_input, icr, out=np.zeros_like(carb_input), where=icr!=0)
    bg_component = np.divide(bg_input - target_bg, isf, out=np.zeros_like(bg_input), where=isf!=0)
    
    prediction = carb_component + bg_component
    prediction = np.clip(prediction, 0, CONFIG["cap_normal"])

    elapsed_time = time.time() - start_time
    print(f"Predicción basada en reglas completa en {elapsed_time:.2f} segundos")
    return prediction

# %% CELL: DRL Environment Definition
class InsulinDoseEnv(gym.Env):
    def __init__(self, X_cgm, X_other, y, scaler_y):
        super(InsulinDoseEnv, self).__init__()
        self.X_cgm = X_cgm.astype(np.float32)  # Ensure input data is float32
        self.X_other = X_other.astype(np.float32)  # Ensure input data is float32
        self.y = y.astype(np.float32)  # Ensure target data is float32
        self.scaler_y = scaler_y  # Scaler for denormalizing doses
        self.current_step = 0
        self.n_samples = len(X_cgm)
        
        # Define the state space (CGM + other features)
        state_dim = X_cgm.shape[2] * X_cgm.shape[1] + X_other.shape[1]  # Flatten CGM (24*1) + other features (6)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        
        # Define the action space (insulin dose, continuous)
        self.action_space = spaces.Box(low=-3, high=3, shape=(1,), dtype=np.float32)  # Adjust based on your scaler
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = np.random.randint(0, self.n_samples)
        cgm_state = self.X_cgm[self.current_step].flatten()  # Shape: (24,)
        other_state = self.X_other[self.current_step]  # Shape: (6,)
        state = np.concatenate([cgm_state, other_state]).astype(np.float32)  # Ensure state is float32
        self.current_state = state  # Store the current state for use in step()
        return state, {}  # Return state and info dict
    
    def step(self, action):
        true_dose = self.scaler_y.inverse_transform(self.y[self.current_step].reshape(-1, 1)).flatten()[0]
        predicted_dose = self.scaler_y.inverse_transform(action.reshape(-1, 1)).flatten()[0]
        reward = -abs(predicted_dose - true_dose)
        reward = float(reward)  # Explicitly convert reward to a Python float
        done = True
        truncated = False
        info = {"true_dose": true_dose, "predicted_dose": predicted_dose}
        next_state = self.current_state
        return next_state, reward, done, truncated, info
    
    def render(self, mode='human'):
        pass

# %% CELL: Visualization Functions
def compute_metrics(y_true, y_pred, scaler_y):
    y_true_denorm = scaler_y.inverse_transform(y_true.reshape(-1, 1)).flatten()
    y_pred_denorm = y_pred  # Already denormalized
    mae = mean_absolute_error(y_true_denorm, y_pred_denorm)
    rmse = np.sqrt(mean_squared_error(y_true_denorm, y_pred_denorm))
    r2 = r2_score(y_true_denorm, y_pred_denorm)
    return mae, rmse, r2

def plot_evaluation(y_test, y_pred_ppo, y_rule, subject_test, scaler_y):
    start_time = time.time()
    y_test_denorm = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    colors = {'PPO': 'green', 'Rules': 'orange'}

    plt.figure(figsize=(15, 12))

    # 1. Predictions vs Real (Separate Hexbin Plots)
    offset = 1e-2
    plt.subplot(3, 2, (1, 2))
    plt.subplot(3, 2, 1)
    plt.hexbin(y_test_denorm + offset, y_pred_ppo + offset, gridsize=50, cmap='viridis', 
               mincnt=1, bins='log')
    plt.plot([offset, 15], [offset, 15], 'k--', label='Perfect Prediction')
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks([0.01, 0.1, 1, 10], ['0.01', '0.1', '1', '10'])
    plt.yticks([0.01, 0.1, 1, 10], ['0.01', '0.1', '1', '10'])
    plt.xlabel('Real Dose (units)', fontsize=10)
    plt.ylabel('Predicted Dose (units)', fontsize=10)
    plt.title('PPO: Predictions vs Real', fontsize=12)
    plt.colorbar(label='Number of Predictions (log scale)')
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.hexbin(y_test_denorm + offset, y_rule + offset, gridsize=50, cmap='viridis', 
               mincnt=1, bins='log')
    plt.plot([offset, 15], [offset, 15], 'k--', label='Perfect Prediction')
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks([0.01, 0.1, 1, 10], ['0.01', '0.1', '1', '10'])
    plt.yticks([0.01, 0.1, 1, 10], ['0.01', '0.1', '1', '10'])
    plt.xlabel('Real Dose (units)', fontsize=10)
    plt.ylabel('Predicted Dose (units)', fontsize=10)
    plt.title('Rules: Predictions vs Real', fontsize=12)
    plt.colorbar(label='Number of Predictions (log scale)')
    plt.legend()

    # 2. Residual Distribution (KDE Plots)
    plt.subplot(3, 1, 2)
    residuals_ppo = y_test_denorm - y_pred_ppo
    residuals_rule = y_test_denorm - y_rule

    sns.kdeplot(residuals_ppo, label='PPO', color=colors['PPO'], fill=True, alpha=0.3)
    sns.kdeplot(residuals_rule, label='Rules', color=colors['Rules'], fill=True, alpha=0.3)
    plt.xlabel('Residual (units)', fontsize=10)
    plt.ylabel('Density', fontsize=10)
    plt.title('Residual Distribution (KDE)', fontsize=12)
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

    # 3. MAE by Subject
    test_subjects = np.unique(subject_test)
    mae_ppo, mae_rule = [], []
    for sid in test_subjects:
        mask = subject_test == sid
        if np.sum(mask) > 0:
            mae_ppo.append(mean_absolute_error(y_test_denorm[mask], y_pred_ppo[mask]))
            mae_rule.append(mean_absolute_error(y_test_denorm[mask], y_rule[mask]))

    plt.subplot(3, 1, 3)
    bar_width = 0.35
    x = np.arange(len(test_subjects))
    plt.bar(x - bar_width/2, mae_ppo, width=bar_width, label='PPO', color=colors['PPO'], alpha=0.8)
    plt.bar(x + bar_width/2, mae_rule, width=bar_width, label='Rules', color=colors['Rules'], alpha=0.8)
    plt.xlabel('Subject', fontsize=10)
    plt.ylabel('MAE (units)', fontsize=10)
    plt.xticks(x, test_subjects, rotation=45, ha='right', fontsize=8)
    plt.ylim(0, 2.3)
    plt.title('MAE by Subject', fontsize=12)
    plt.legend(fontsize=8)
    plt.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

    elapsed_time = time.time() - start_time
    print(f"Visualización completa en {elapsed_time:.2f} segundos")

# %% CELL: Main Execution - Preprocess Data
X_cgm, X_other, X_subject, y, df_final, scaler_cgm, scaler_other, scaler_y = preprocess_data(CONFIG["data_path"])

# %% CELL: Main Execution - Split Data
(X_cgm_train, X_cgm_val, X_cgm_test,
 X_other_train, X_other_val, X_other_test,
 X_subject_train, X_subject_val, X_subject_test,
 y_train, y_val, y_test, subject_test) = split_data(X_cgm, X_other, X_subject, y, df_final)

# %% CELL: Main Execution - DRL (PPO) Training
# Create the training environment
train_env = InsulinDoseEnv(X_cgm_train, X_other_train, y_train, scaler_y)

# Verify that the environment is correctly implemented
check_env(train_env)

# Initialize the PPO model
model_ppo = PPO("MlpPolicy", train_env, verbose=1, learning_rate=3e-4, n_steps=2048, batch_size=64)

# Train the model
total_timesteps = 100000  # Adjust based on convergence
model_ppo.learn(total_timesteps=total_timesteps)

# Save the model (optional)
model_ppo.save("ppo_insulin_dose")

# Function to generate predictions with PPO
def predict_with_ppo(model, X_cgm, X_other):
    predictions = []
    env = InsulinDoseEnv(X_cgm, X_other, np.zeros(len(X_cgm)), scaler_y)  # Dummy y since we only need X for prediction
    
    for i in range(len(X_cgm)):
        cgm_state = X_cgm[i].flatten()  # Shape: (24,)
        other_state = X_other[i]  # Shape: (6,)
        state = np.concatenate([cgm_state, other_state])  # Shape: (30,)
        
        action, _ = model.predict(state, deterministic=True)  # Use deterministic mode for evaluation
        predicted_dose = scaler_y.inverse_transform(action.reshape(-1, 1)).flatten()[0]
        predictions.append(predicted_dose)
    
    return np.array(predictions)

# Generate predictions on the test set
y_pred_ppo = predict_with_ppo(model_ppo, X_cgm_test, X_other_test)

# Generate rule-based predictions for comparison
y_rule = rule_based_prediction(X_other_test, scaler_other, scaler_y)

# %% CELL: Main Execution - Print Metrics
# Metrics for PPO
mae_ppo = mean_absolute_error(scaler_y.inverse_transform(y_test.reshape(-1, 1)), y_pred_ppo)
rmse_ppo = np.sqrt(mean_squared_error(scaler_y.inverse_transform(y_test.reshape(-1, 1)), y_pred_ppo))
r2_ppo = r2_score(scaler_y.inverse_transform(y_test.reshape(-1, 1)), y_pred_ppo)
print(f"PPO - MAE: {mae_ppo:.2f}, RMSE: {rmse_ppo:.2f}, R²: {r2_ppo:.2f}")

# Metrics for the Rules-based model
mae_rule = mean_absolute_error(scaler_y.inverse_transform(y_test.reshape(-1, 1)), y_rule)
rmse_rule = np.sqrt(mean_squared_error(scaler_y.inverse_transform(y_test.reshape(-1, 1)), y_rule))
r2_rule = r2_score(scaler_y.inverse_transform(y_test.reshape(-1, 1)), y_rule)
print(f"Rules - MAE: {mae_rule:.2f}, RMSE: {rmse_rule:.2f}, R²: {r2_rule:.2f}")

# %% CELL: Main Execution - Visualization
# Visualize the results
plot_evaluation(y_test, y_pred_ppo, y_rule, subject_test, scaler_y)

# %% CELL: Main Execution - Metrics per Subject
# Analyze performance per subject
print("\nRendimiento por sujeto:")
for subject_id in np.unique(subject_test):
    mask = subject_test == subject_id
    if np.sum(mask) > 0:
        y_test_sub = scaler_y.inverse_transform(y_test[mask].reshape(-1, 1)).flatten()
        print(f"Sujeto {subject_id}: ", end="")
        mae_ppo_sub = mean_absolute_error(y_test_sub, y_pred_ppo[mask])
        print(f"PPO MAE={mae_ppo_sub:.2f}, ", end="")
        mae_rule_sub = mean_absolute_error(y_test_sub, y_rule[mask])
        print(f"Rules MAE={mae_rule_sub:.2f}")
# %%
