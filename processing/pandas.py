# %% CELL: Required Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from joblib import Parallel, delayed
import time
from tqdm import tqdm
from datetime import timedelta

# Global configuration
CONFIG = {
    "batch_size": 128,
    "window_hours": 2,
    "cap_normal": 30,
    "cap_bg": 300,
    "cap_iob": 5,
    "cap_carb": 150,
    "data_path": os.path.join(os.getcwd(), "subjects"),
    "low_dose_threshold": 7.0  # Clinical threshold for low-dose insulin
}

# %% CELL: Data Processing Functions
def get_cgm_window(bolus_time, cgm_df, window_hours=CONFIG["window_hours"]):
    window_start = bolus_time - timedelta(hours=window_hours)
    window = cgm_df[(cgm_df['date'] >= window_start) & (cgm_df['date'] <= bolus_time)]
    window = window.sort_values('date').tail(24)
    return window['mg/dl'].values if len(window) >= 24 else None

def calculate_iob(bolus_time, basal_df, half_life_hours=4):
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

    elapsed_time = time.time() - start_time
    print(f"Preprocesamiento completo en {elapsed_time:.2f} segundos")
    return df_final

def split_data(df_final):
    start_time = time.time()
    subject_stats = df_final.groupby('subject_id')['normal'].agg(['mean', 'std']).reset_index()
    subject_stats.columns = ['subject_id', 'mean_dose', 'std_dose']
    subject_ids = df_final['subject_id'].unique()

    sorted_subjects = subject_stats.sort_values('mean_dose')['subject_id'].values
    n_subjects = len(sorted_subjects)
    train_size = int(0.8 * n_subjects)
    val_size = int(0.1 * n_subjects)
    test_size = n_subjects - train_size - val_size

    test_subjects = [49] if 49 in sorted_subjects else []
    remaining_subjects = [s for s in sorted_subjects if s != 49]
    train_subjects = []
    val_subjects = []

    remaining_subjects_list = list(remaining_subjects)
    np.random.shuffle(remaining_subjects_list)

    for i, subject in enumerate(remaining_subjects_list):
        train_mean = df_final[df_final['subject_id'].isin(train_subjects)]['normal'].mean() if train_subjects else 0
        val_mean = df_final[df_final['subject_id'].isin(val_subjects)]['normal'].mean() if val_subjects else 0
        test_mean = df_final[df_final['subject_id'].isin(test_subjects)]['normal'].mean() if test_subjects else 0
        train_std = df_final[df_final['subject_id'].isin(train_subjects)]['normal'].std() if train_subjects else 0
        val_std = df_final[df_final['subject_id'].isin(val_subjects)]['normal'].std() if val_subjects else 0
        test_std = df_final[df_final['subject_id'].isin(test_subjects)]['normal'].std() if test_subjects else 0

        train_temp = train_subjects + [subject]
        val_temp = val_subjects + [subject]
        test_temp = test_subjects + [subject]

        train_mean_new = df_final[df_final['subject_id'].isin(train_temp)]['normal'].mean()
        val_mean_new = df_final[df_final['subject_id'].isin(val_temp)]['normal'].mean()
        test_mean_new = df_final[df_final['subject_id'].isin(test_temp)]['normal'].mean()
        train_std_new = df_final[df_final['subject_id'].isin(train_temp)]['normal'].std()
        val_std_new = df_final[df_final['subject_id'].isin(val_temp)]['normal'].std()
        test_std_new = df_final[df_final['subject_id'].isin(test_temp)]['normal'].std()

        means_if_train = [train_mean_new, val_mean, test_mean]
        means_if_val = [train_mean, val_mean_new, test_mean]
        means_if_test = [train_mean, val_mean, test_mean_new]
        stds_if_train = [train_std_new, val_std, test_std]
        stds_if_val = [train_std, val_std_new, test_std]
        stds_if_test = [train_std, val_std, test_std_new]

        range_means_if_train = max(means_if_train) - min(means_if_train) if all(m != 0 for m in means_if_train) else float('inf')
        range_means_if_val = max(means_if_val) - min(means_if_val) if all(m != 0 for m in means_if_val) else float('inf')
        range_means_if_test = max(means_if_test) - min(means_if_test) if all(m != 0 for m in means_if_test) else float('inf')
        range_stds_if_train = max(stds_if_train) - min(stds_if_train) if all(s != 0 for s in stds_if_train) else float('inf')
        range_stds_if_val = max(stds_if_val) - min(stds_if_val) if all(s != 0 for s in stds_if_val) else float('inf')
        range_stds_if_test = max(stds_if_test) - min(stds_if_test) if all(s != 0 for s in stds_if_test) else float('inf')

        score_if_train = range_means_if_train + range_stds_if_train
        score_if_val = range_means_if_val + range_stds_if_val
        score_if_test = range_means_if_test + range_stds_if_test

        if len(train_subjects) < train_size and score_if_train <= min(score_if_val, score_if_test):
            train_subjects.append(subject)
        elif len(val_subjects) < val_size and score_if_val <= min(score_if_train, score_if_test):
            val_subjects.append(subject)
        elif len(test_subjects) < test_size:
            test_subjects.append(subject)
        else:
            train_subjects.append(subject)

    train_mask = df_final['subject_id'].isin(train_subjects)
    val_mask = df_final['subject_id'].isin(val_subjects)
    test_mask = df_final['subject_id'].isin(test_subjects)

    y_train_temp = df_final.loc[train_mask, 'normal']
    y_val_temp = df_final.loc[val_mask, 'normal']
    y_test_temp = df_final.loc[test_mask, 'normal']
    print("Post-split Train y: mean =", y_train_temp.mean(), "std =", y_train_temp.std())
    print("Post-split Val y: mean =", y_val_temp.mean(), "std =", y_val_temp.std())
    print("Post-split Test y: mean =", y_test_temp.mean(), "std =", y_test_temp.std())

    scaler_cgm = StandardScaler()
    scaler_other = StandardScaler()
    scaler_y = StandardScaler()
    cgm_columns = [f'cgm_{i}' for i in range(24)]
    other_features = ['carbInput', 'bgInput', 'insulinOnBoard', 'insulinCarbRatio', 
                      'insulinSensitivityFactor', 'hour_of_day']

    x_cgm_train = scaler_cgm.fit_transform(df_final.loc[train_mask, cgm_columns]).reshape(-1, 24, 1)
    x_cgm_val = scaler_cgm.transform(df_final.loc[val_mask, cgm_columns]).reshape(-1, 24, 1)
    x_cgm_test = scaler_cgm.transform(df_final.loc[test_mask, cgm_columns]).reshape(-1, 24, 1)
    x_other_train = scaler_other.fit_transform(df_final.loc[train_mask, other_features])
    x_other_val = scaler_other.transform(df_final.loc[val_mask, other_features])
    x_other_test = scaler_other.transform(df_final.loc[test_mask, other_features])
    y_train = scaler_y.fit_transform(df_final.loc[train_mask, 'normal'].values.reshape(-1, 1)).flatten()
    y_val = scaler_y.transform(df_final.loc[val_mask, 'normal'].values.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(df_final.loc[test_mask, 'normal'].values.reshape(-1, 1)).flatten()

    x_subject_train = df_final.loc[train_mask, 'subject_id'].values
    x_subject_val = df_final.loc[val_mask, 'subject_id'].values
    x_subject_test = df_final.loc[test_mask, 'subject_id'].values
    subject_test = x_subject_test

    print(f"Entrenamiento CGM: {x_cgm_train.shape}, Validación CGM: {x_cgm_val.shape}, Prueba CGM: {x_cgm_test.shape}")
    print(f"Entrenamiento Otros: {x_other_train.shape}, Validación Otros: {x_other_val.shape}, Prueba Otros: {x_other_test.shape}")
    print(f"Entrenamiento Subject: {x_subject_train.shape}, Validación Subject: {x_subject_val.shape}, Prueba Subject: {x_subject_test.shape}")
    print(f"Sujetos de prueba: {test_subjects}")

    elapsed_time = time.time() - start_time
    print(f"División de datos completa en {elapsed_time:.2f} segundos")
    return (x_cgm_train, x_cgm_val, x_cgm_test,
            x_other_train, x_other_val, x_other_test,
            x_subject_train, x_subject_val, x_subject_test,
            y_train, y_val, y_test, subject_test,
            scaler_cgm, scaler_other, scaler_y)

df_final = pd.read_csv('df_final.csv')

(x_cgm_train, x_cgm_val, x_cgm_test,
 x_other_train, x_other_val, x_other_test,
 x_subject_train, x_subject_val, x_subject_test,
 y_train, y_val, y_test, subject_test,
 scaler_cgm, scaler_other, scaler_y) = split_data(df_final)