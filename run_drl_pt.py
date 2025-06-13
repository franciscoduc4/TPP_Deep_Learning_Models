# Imports
import os
import sys
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
import torch
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from validation.model_validation import validate_dosing_model
from validation.simulator import GlucoseSimulator
from constants.constants import CONST_DEFAULT_SEED, CONST_METRIC_MAE, CONST_METRIC_R2, CONST_METRIC_RMSE, SEVERE_HYPOGLYCEMIA_THRESHOLD, HYPOGLYCEMIA_THRESHOLD, HYPERGLYCEMIA_THRESHOLD, SEVERE_HYPERGLYCEMIA_THRESHOLD, IDEAL_UPPER_BOUND, IDEAL_LOWER_BOUND
from training.pytorch import (
    train_multiple_models, calculate_metrics, evaluate_clinical_metrics, 
    optimize_ensemble_weights_clinical, enhance_features
)
from config.feature_selection import prepare_data_for_drl_training, prepare_features_for_drl, get_feature_groups, validate_contextual_data_coverage


PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT)

# Printer
from custom.printer import coloured, print_critical, print_debug, print_error, print_info, print_warning, print_log, print_success

# Configuración 
from config.params import FRAMEWORK, PROCESSING, MODELS, MODELS_USAGE, EVALUATE, EVALUATE_USAGE, TRAINING_CONFIG

# Procesamiento
from processing.pandas import preprocess_data as pd_preprocess, split_data as pd_split
from processing.polars import preprocess_data as pl_preprocess, split_data as pl_split

# Visualización
from visualization.plotting import visualize_model_results, plot_model_evaluation_summary

# Reporte
from report.generate_report import create_report, render_to_pdf

# Validación
from validation.model_validation import validate_model_with_simulator
from validation.simulator import GlucoseSimulator

# Constantes para los directorios y nombres
CONST_MODELS_DIR = "models"
CONST_RESULTS_DIR = "results"
CONST_ENSEMBLE = "ensemble"

torch.autograd.set_detect_anomaly(True)

# Auxiliary Functions
def is_model_creator(fn: Any) -> bool:
    """
    Verifica si una función es un model creator que debe ser llamada para obtener
    la función de creación del modelo.
    
    Parámetros:
    -----------
    fn : Any
        Función a verificar
        
    Retorna:
    --------
    bool
        True si es un model creator, False si ya es una función de creación de modelo
    """
    if callable(fn):
        try:
            import inspect
            sig = inspect.signature(fn)
            # Si no tiene parámetros, es probable que es un model creator
            # que debe llamarse para obtener la función de creación real
            return len(sig.parameters) == 0
        except Exception:
            pass
    return False

# Importación dinámica de módulos de entrenamiento según el framework seleccionado
coloured(f"Framework seleccionado: {FRAMEWORK}", 'blue', 'bold')

if torch.cuda.is_available():
    coloured(f"GPU available: {torch.cuda.device_count()} devices", 'green')
    coloured(f"Using: {torch.cuda.get_device_name(0)}", 'green')
else:
    coloured("No GPUs detected, using CPU", 'yellow')

# Constante para mensaje repetido
CONST_MODEL_ACTIVATED = "Modelo {} activado."
CONST_MODEL_DEACTIVATED = "Modelo {} desactivado."

use_models = {}

for model_name, use in MODELS_USAGE.items():
    if use:
        model_fn = MODELS[model_name]
        if is_model_creator(model_fn):
            model_fn = model_fn()
        use_models[model_name] = model_fn
        coloured(CONST_MODEL_ACTIVATED.format(model_name), 'green', 'bold')
    else:
        coloured(CONST_MODEL_DEACTIVATED.format(model_name), 'red', 'bold')

# Validaciones Previas
if PROCESSING not in ["pandas", "polars"]:
    coloured(f"Error: El procesamiento debe ser 'pandas' o 'polars'. Se recibió '{PROCESSING}'", 'red', 'bold')
    sys.exit(1)
if not use_models:
    coloured("Error: No se ha activado ningún modelo. Por favor, activa al menos un modelo en 'MODELS_USAGE'.", 'red', 'bold')
    sys.exit(1)
if MODELS_USAGE.values() == [False] * len(use_models):
    coloured("Error: Todos los modelos están desactivados. Por favor, activa al menos un modelo en 'MODELS_USAGE'.", 'red', 'bold')
    sys.exit(1)
if len(use_models) == 0:
    coloured("Error: No se ha activado ningún modelo. Por favor, activa al menos un modelo en 'MODELS_USAGE'.", 'red', 'bold')
    sys.exit(1)

# Rutas de datos y figuras
SUBJECTS_PATH = os.path.join(PROJECT_ROOT, "data", "subjects")
coloured(f"Ruta de sujetos: {SUBJECTS_PATH}", 'yellow', 'bold')

# Crear directorio para modelos según el framework
MODELS_SAVE_DIR = os.path.join(PROJECT_ROOT, CONST_RESULTS_DIR, CONST_MODELS_DIR, FRAMEWORK)
os.makedirs(MODELS_SAVE_DIR, exist_ok=True)
coloured(f"Ruta para guardar modelos: {MODELS_SAVE_DIR}", 'yellow', 'bold')

# Crear directorio para resultados según el framework
RESULTS_SAVE_DIR = os.path.join(PROJECT_ROOT, CONST_RESULTS_DIR, FRAMEWORK)
os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)
coloured(f"Ruta para guardar resultados: {RESULTS_SAVE_DIR}", 'yellow', 'bold')

# Crear directorios para figuras
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures", "various_models", FRAMEWORK)
os.makedirs(FIGURES_DIR, exist_ok=True)
coloured(f"Ruta de figuras: {FIGURES_DIR}", 'yellow', 'bold')

subject_files = [f for f in os.listdir(SUBJECTS_PATH) if f.startswith("Subject") and f.endswith(".xlsx")]
coloured(f"Total sujetos: {len(subject_files)}", 'yellow', 'bold')

# Procesamiento de datos
coloured("Procesando datos con polars...", 'blue', 'bold')
df_pl: pl.DataFrame = pl_preprocess()

train_df: pl.DataFrame = None
val_df: pl.DataFrame = None
test_df: pl.DataFrame = None
train_df, val_df, test_df = pl_split(df_pl)

if train_df is None or val_df is None or test_df is None:
    print_critical("Fallo en el preprocesamiento de datos. Terminando ejecución.")
    sys.exit(1)
if train_df.is_empty() or val_df.is_empty() or test_df.is_empty():
    print_critical("Uno o más conjuntos de datos (train, val, test) están vacíos. Terminando ejecución.")
    sys.exit(1)

feature_groups = get_feature_groups() # Sigue siendo útil para configuración

coloured("\n==== INFORMACIÓN DE DATOS PREPARADOS (DATAFRAMES) ====", 'cyan', 'bold')
coloured(f"Entrenamiento - Shape: {train_df.shape}", 'green')
coloured(f"Validación    - Shape: {val_df.shape}", 'green')
coloured(f"Test          - Shape: {test_df.shape}", 'green')

# Mejorar características directamente en los DataFrames
coloured("\n==== GENERACIÓN DE CARACTERÍSTICAS ADICIONALES (DATAFRAMES) ====", 'cyan', 'bold')
train_df_enhanced = enhance_features(train_df.clone()) # Usar clone si enhance_features modifica inplace
val_df_enhanced = enhance_features(val_df.clone())
test_df_enhanced = enhance_features(test_df.clone())

coloured("Forma de DataFrames mejorados:", 'green')
coloured(f"  Train: {train_df_enhanced.shape}", 'green')
coloured(f"  Val:   {val_df_enhanced.shape}", 'green')
coloured(f"  Test:  {test_df_enhanced.shape}", 'green')

# input_shapes ya no se define de la misma manera, los modelos/wrappers usarán feature_config

# Estructurar all_data para pasar a train_multiple_models
# Ahora contendrá DataFrames directamente.
# El wrapper se encargará de extraer 'y' (target) y otras partes del DataFrame.
all_data_dfs = {
    'train': train_df_enhanced,
    'val': val_df_enhanced,
    'test': test_df_enhanced
    # Ya no se necesita 'subject_id' por separado si está como columna en los DFs
}

# Entrenamiento de modelos
coloured("\n==== ENTRENAMIENTO DE MODELOS ====", 'cyan', 'bold')
# train_multiple_models necesita ser adaptado para pasar DataFrames a model_wrapper.fit
# y para que model_wrapper.fit maneje DataFrames.

# Asumimos que feature_config se pasa a los wrappers durante su creación en train_multiple_models
# o que los wrappers lo obtienen de get_feature_groups()

histories, predictions_dfs, clinical_metrics_results, trained_models = train_multiple_models(
    data_dfs=all_data_dfs, # Pasar los DataFrames
    models_to_use=MODELS_USAGE,
    model_creators=MODELS, # Los creadores ahora deben configurar el wrapper con feature_config
    models_dir=MODELS_SAVE_DIR,
    results_dir=RESULTS_SAVE_DIR,
    figures_dir=FIGURES_DIR,
    training_config=TRAINING_CONFIG,
    train_per_patient=True, # Esta lógica necesitará adaptarse si se mantiene
    feature_config=feature_groups # Pasar la configuración de features
)

print_log(f"{histories}")
print(predictions_dfs.head())
print_success(f"{clinical_metrics_results}")

# # Validar cobertura de datos contextuales
# coloured("\n==== VALIDACIÓN DE DATOS CONTEXTUALES ====", 'cyan', 'bold')
# contextual_coverage = validate_contextual_data_coverage(df_pl)
# # Preparar datos con división temporal apropiada
# coloured("\n==== PREPARACIÓN DE DATOS PARA DRL ====", 'cyan', 'bold')
# # Se asume que prepare_data_for_drl_training ahora retorna también los diccionarios de contexto y subject_id
# (x_cgm_train, x_other_train, y_train, context_train, subject_id_train,
#     x_cgm_val, x_other_val, y_val, context_val, subject_id_val,
#     x_cgm_test, x_other_test, y_test, context_test, subject_id_test) = prepare_data_for_drl_training(df_pl)

# # Mostrar información sobre características seleccionadas
# feature_groups = get_feature_groups()

# # Mostrar información sobre los datos preparados
# coloured("\n==== INFORMACIÓN DE DATOS PREPARADOS ====", 'cyan', 'bold')
# coloured(f"Entrenamiento - CGM: {x_cgm_train.shape}, Otros: {x_other_train.shape}, Target: {y_train.shape}, SubjectIDs: {subject_id_train.shape}", 'green')
# coloured(f"Validación    - CGM: {x_cgm_val.shape}, Otros: {x_other_val.shape}, Target: {y_val.shape}, SubjectIDs: {subject_id_val.shape}", 'green')
# coloured(f"Test          - CGM: {x_cgm_test.shape}, Otros: {x_other_test.shape}, Target: {y_test.shape}, SubjectIDs: {subject_id_test.shape}", 'green')
# if context_train is not None:
#     coloured(f"Contexto Train - Claves: {list(context_train.keys())}, Muestra forma: {context_train[list(context_train.keys())[0]].shape if context_train else 'N/A'}", 'green')
# if context_val is not None:
#     coloured(f"Contexto Val   - Claves: {list(context_val.keys())}, Muestra forma: {context_val[list(context_val.keys())[0]].shape if context_val else 'N/A'}", 'green')
# if context_test is not None:
#     coloured(f"Contexto Test  - Claves: {list(context_test.keys())}, Muestra forma: {context_test[list(context_test.keys())[0]].shape if context_test else 'N/A'}", 'green')
# # Mejorar características utilizando la función del framework seleccionado
# coloured("\n==== GENERACIÓN DE CARACTERÍSTICAS ADICIONALES ====", 'cyan', 'bold')
# x_cgm_train_enhanced, x_other_train_enhanced = enhance_features(x_cgm_train, x_other_train)
# x_cgm_val_enhanced, x_other_val_enhanced = enhance_features(x_cgm_val, x_other_val)
# x_cgm_test_enhanced, x_other_test_enhanced = enhance_features(x_cgm_test, x_other_test)

# coloured("Forma de datos mejorados:", 'green')
# coloured(f"  Train - CGM: {x_cgm_train_enhanced.shape}, Otros: {x_other_train_enhanced.shape}", 'green')
# coloured(f"  Val   - CGM: {x_cgm_val_enhanced.shape}, Otros: {x_other_val_enhanced.shape}", 'green')
# coloured(f"  Test  - CGM: {x_cgm_test_enhanced.shape}, Otros: {x_other_test_enhanced.shape}", 'green')

# # Definir formas de entrada para los modelos
# input_shapes = (x_cgm_train_enhanced.shape[1:], x_other_train_enhanced.shape[1:])
# coloured(f"Formas de entrada para los modelos: CGM {input_shapes[0]}, Otros {input_shapes[1]}", 'green')

# # Estructurar all_data para pasar a train_multiple_models
# all_data = {
#     'train': {
#         'x_cgm': x_cgm_train_enhanced,
#         'x_other': x_other_train_enhanced,
#         'y': y_train,
#         'context': context_train,
#         'subject_id': subject_id_train
#     },
#     'val': {
#         'x_cgm': x_cgm_val_enhanced,
#         'x_other': x_other_val_enhanced,
#         'y': y_val,
#         'context': context_val,
#         'subject_id': subject_id_val
#     },
#     'test': {
#         'x_cgm': x_cgm_test_enhanced,
#         'x_other': x_other_test_enhanced,
#         'y': y_test,
#         'context': context_test,
#         'subject_id': subject_id_test
#     }
# }

# # Entrenamiento de modelos
# coloured("\n==== ENTRENAMIENTO DE MODELOS ====", 'cyan', 'bold')

# histories, predictions, clinical_metrics_results, trained_models = train_multiple_models(
#     data=all_data,
#     models_to_use=MODELS_USAGE,
#     model_creators=MODELS, # Suponiendo que MODELS es el dict de creadores
#     models_dir=MODELS_SAVE_DIR,
#     results_dir=RESULTS_SAVE_DIR, # Pasar si es necesario internamente
#     figures_dir=FIGURES_DIR,     # Pasar si es necesario internamente
#     training_config=TRAINING_CONFIG,
#     train_per_patient=True  # O False para entrenamiento global
# )

# # Creación del Ensamble
# coloured("\n==== CREACIÓN DEL ENSAMBLE ====", 'cyan', 'bold')
# ensemble_prediction = None
# ensemble_metrics = None
# clinical_results = {}

# # Inicializar simulador para métricas clínicas
# simulator = GlucoseSimulator()

# # Extraer valores iniciales de glucosa y carbohidratos del conjunto de prueba
# initial_glucose = np.array([x_cgm_test_enhanced[i, -1, 0] for i in range(len(x_cgm_test_enhanced))])

# # Manera más robusta de identificar la columna de carbohidratos
# # Ahora se extrae del diccionario de contexto del conjunto de prueba
# carb_intake_key = 'meal_carbs' # o 'carb_intake' si se renombra en prepare_drl_data
# if context_test and carb_intake_key in context_test:
#     carb_intake = context_test[carb_intake_key]
#     if carb_intake.ndim > 1 and carb_intake.shape[1] == 1: # Asegurar que sea 1D
#         carb_intake = carb_intake.flatten()
# elif x_other_test_enhanced.shape[1] > 0:
#     # Fallback si no está en el contexto, intentar buscar en x_other_test_enhanced
#     # Esto requeriría conocer el índice o nombre de la columna de carbohidratos en x_other_test_enhanced
#     print_warning(f"'{carb_intake_key}' no encontrado en context_test. Intentando fallback (puede ser incorrecto).")
#     # Asumir un índice (esto es frágil, idealmente carb_intake siempre vendrá de context_test)
#     # Por ejemplo, si 'meal_carbs_log1p' fuera la primera columna de x_other_test_enhanced:
#     # carb_intake_idx = 0 # Esto es un placeholder, debe ajustarse si se usa este fallback
#     # carb_intake = np.array([x_other_test_enhanced[i, carb_intake_idx] for i in range(len(x_other_test_enhanced))])
#     # Es mejor asegurar que context_test contenga la ingesta de carbohidratos.
#     # Por ahora, si no está en el contexto, se podría generar un error o un valor por defecto.
#     print_error("No se pudo determinar la ingesta de carbohidratos para la evaluación clínica. Usando valores por defecto (0).")
#     carb_intake = np.zeros(len(x_cgm_test_enhanced))
# else:
#     print_error("No se pudo determinar la ingesta de carbohidratos para la evaluación clínica. Usando valores por defecto (0).")
#     carb_intake = np.zeros(len(x_cgm_test_enhanced))
# # Evaluar métricas clínicas para cada modelo
# for model_name, model_pred in predictions.items():
#     clinical_metrics = evaluate_clinical_metrics(
#         simulator=simulator,
#         predictions=model_pred,
#         initial_glucose=initial_glucose,
#         carb_intake=carb_intake
#     )
#     clinical_results[model_name] = clinical_metrics
    
#     # coloured(f"\nMétricas clínicas para {model_name}:", 'green', 'bold')
#     # coloured(f"  Tiempo Severamente Bajo Rango: {clinical_metrics['time_severe_below']:.2f}%", 'red')
#     # coloured(f"  Tiempo Bajo Rango: {clinical_metrics['time_below_range']:.2f}%", 'yellow')
#     # coloured(f"  Tiempo en Rango: {clinical_metrics['time_in_range']:.2f}%", 'green')
#     # coloured(f"  Tiempo Sobre Rango: {clinical_metrics['time_above_range']:.2f}%", 'yellow')
#     # coloured(f"  Tiempo Severamente Sobre Rango: {clinical_metrics['time_severe_above']:.2f}%", 'red')
#     print_debug(f"\nMétricas clínicas para {model_name}:")
#     print_debug(f"{clinical_metrics}")
    
#     # Guardar métricas clínicas
#     with open(os.path.join(RESULTS_SAVE_DIR, f"{model_name}_clinical_metrics.json"), 'w') as f:
#         json.dump(clinical_metrics, f, indent=2)

# # Crear ensamble si hay más de un modelo
# if len(predictions) > 1:
#     coloured("Optimizando pesos del ensamble basados en métricas clínicas...", 'magenta')
    
#     initial_glucose_for_ensemble = initial_glucose # Already global
#     carb_intake_for_ensemble = carb_intake       # Already global

#     # Transformar predictions de Dict[model_name, Dict[patient_id, np.ndarray]]
#     # a Dict[model_name, np.ndarray (global_preds)]
#     global_predictions_for_ensemble = {}
#     num_test_samples = len(y_test)
#     test_subject_ids_flat = all_data['test']['subject_id']

#     for model_name, patient_preds_map in predictions.items():
#         model_global_preds = np.full(num_test_samples, np.nan, dtype=float)
#         processed_indices = np.zeros(num_test_samples, dtype=bool) # Para evitar sobreescrituras si hay IDs duplicados

#         # Ordenar patient_id para un orden de concatenación determinístico si es necesario,
#         # pero el enfoque de rellenar usando test_subject_ids_flat es más robusto.
#         # Asumimos que patient_id en patient_preds_map.keys() son str o int y compatibles con test_subject_ids_flat
        
#         sorted_patient_ids = sorted(patient_preds_map.keys(), key=lambda x: str(x))

#         for patient_id_key in sorted_patient_ids:
#             preds_array_for_patient = patient_preds_map[patient_id_key]
            
#             # Intentar convertir patient_id_key a int si es necesario, asumiendo IDs numéricos
#             try:
#                 current_patient_id = int(patient_id_key)
#             except ValueError:
#                 print_warning(f"No se pudo convertir patient_id '{patient_id_key}' a int para el modelo {model_name}. Omitiendo este paciente para el ensamble global.")
#                 continue

#             current_patient_mask_in_test = (test_subject_ids_flat == current_patient_id)
#             num_expected_preds_for_patient = np.sum(current_patient_mask_in_test)

#             if preds_array_for_patient is not None:
#                 preds_array_for_patient_flat = preds_array_for_patient.flatten()
#                 if len(preds_array_for_patient_flat) == num_expected_preds_for_patient:
#                     # Asegurarse de no escribir dos veces en los mismos índices si hay problemas con las máscaras
#                     # Esto es una salvaguarda, idealmente las máscaras no se solapan para diferentes pacientes.
#                     target_indices = np.where(current_patient_mask_in_test)[0]
#                     if np.any(processed_indices[target_indices]):
#                         print_warning(f"Índices superpuestos detectados para paciente {current_patient_id} en modelo {model_name}. Esto no debería ocurrir.")
#                     model_global_preds[target_indices] = preds_array_for_patient_flat
#                     processed_indices[target_indices] = True
#                 else:
#                     print_warning(f"Discrepancia en la longitud de predicciones para el modelo {model_name}, paciente {current_patient_id}. "
#                                   f"Esperado: {num_expected_preds_for_patient}, Obtenido: {len(preds_array_for_patient_flat)}. Se rellenará con NaNs.")
#                     # Los NaNs ya están allí por defecto, o se puede rellenar explícitamente una porción si se conoce.
#             else:
#                 print_warning(f"Predicciones son None para el modelo {model_name}, paciente {current_patient_id}.")

#         global_predictions_for_ensemble[model_name] = model_global_preds

#     weights, ensemble_prediction = optimize_ensemble_weights_clinical(
#         predictions=global_predictions_for_ensemble, # Usar el diccionario transformado
#         initial_glucose=initial_glucose_for_ensemble,
#         carb_intake=carb_intake_for_ensemble,
#         simulator=simulator,
#         y_true=y_test 
#     )
    
#     # Calcular métricas para el ensamble
#     # ensemble_metrics = calculate_metrics(y_test, ensemble_prediction)
#     ensemble_clinical = evaluate_clinical_metrics(
#         simulator=simulator,
#         predictions=ensemble_prediction,
#         initial_glucose=initial_glucose,
#         carb_intake=carb_intake
#     )
    
#     # Mostrar pesos y métricas del ensamble
#     coloured("\nPesos del ensamble:", 'green', 'bold')
#     for i, (model_name, weight) in enumerate(zip(predictions.keys(), weights)):
#         coloured(f"  {model_name}: {weight:.4f}", 'green')
    
#     # coloured("\nMétricas clínicas del ensamble:", 'green', 'bold')
#     # coloured(f"  Tiempo en Rango: {ensemble_clinical['time_in_range']:.2f}%", 'green')
#     # coloured(f"  Tiempo Bajo Rango: {ensemble_clinical['time_below_range']:.2f}%", 'yellow')
#     # coloured(f"  Tiempo Sobre Rango: {ensemble_clinical['time_above_range']:.2f}%", 'yellow')
#     print_debug("\nMétricas clínicas del ensamble:")
#     print_debug(f"{ensemble_clinical}")
    
#     # Guardar predicciones y métricas del ensamble
#     clinical_results[CONST_ENSEMBLE] = ensemble_clinical
#     np.save(os.path.join(RESULTS_SAVE_DIR, f"{CONST_ENSEMBLE}_predictions.npy"), ensemble_prediction)
#     with open(os.path.join(RESULTS_SAVE_DIR, f"{CONST_ENSEMBLE}_metrics.json"), 'w') as f:
#         json.dump({**ensemble_clinical}, f, indent=2)
# else:
#     coloured("No se puede crear ensamble con menos de 2 modelos", 'yellow', 'bold')
    
# # Evaluación con FQE y Doubly Robust
# coloured("\n==== EVALUACIÓN OFFLINE RL ====", 'cyan', 'bold')

# # Seleccionar evaluadores activados
# active_evaluators = {}
# for eval_name, use in EVALUATE_USAGE.items():
#     if use:
#         coloured(f"Evaluador {eval_name} activado.", 'green')
#         active_evaluators[eval_name] = EVALUATE[eval_name]
#     else:
#         coloured(f"Evaluador {eval_name} desactivado.", 'yellow')

# # Realizar evaluación si hay evaluadores activos
# offline_results: Dict[str, Dict[str, float]] = {}
# if active_evaluators:
#     # Asumimos que input_shapes[0] es para CGM y input_shapes[1] para Other
#     # y que estos son tuplas (timesteps, features_cgm) y (features_other,)
#     cgm_dim_eval = input_shapes[0] if input_shapes and len(input_shapes) > 0 else (all_data['test']['x_cgm'].shape[1], all_data['test']['x_cgm'].shape[2] if all_data['test']['x_cgm'].ndim > 2 else 1)
#     other_dim_eval = input_shapes[1] if input_shapes and len(input_shapes) > 1 and input_shapes[1] else (all_data['test']['x_other'].shape[1] if all_data['test']['x_other'].ndim > 1 and all_data['test']['x_other'].shape[1] > 0 else 0,)
    
#     if isinstance(other_dim_eval, int): # Asegurar que other_dim_eval sea una tupla
#         other_dim_eval = (other_dim_eval,)

#     for eval_name, eval_creator in active_evaluators.items():
#         print_info(f"Creando evaluador: {eval_name} con cgm_dim={cgm_dim_eval}, other_dim={other_dim_eval}")
#         eval_instance = eval_creator(cgm_input_dim=cgm_dim_eval, other_input_dim=other_dim_eval)
        
#         # Entrenar el evaluador (FQE y DRE necesitan ser entrenados en los datos offline)
#         # Usamos los datos de entrenamiento completos para ajustar Q-network y behavior policy
#         print_info(f"Entrenando evaluador offline: {eval_name}...")
#         eval_instance.fit(
#             x_cgm=all_data['train']['x_cgm'], 
#             x_other=all_data['train']['x_other'],
#             y_actions=all_data['train']['y'], # Acciones de comportamiento
#             validation_data=(
#                 (all_data['val']['x_cgm'], all_data['val']['x_other']), 
#                 all_data['val']['y']
#             )
#         )
        
#         # Evaluar cada política entrenada (de DRL) con este evaluador offline
#         for model_name, trained_model_wrapper in trained_models.items():
#             print_info(f"Evaluando política del modelo '{model_name}' con evaluador offline '{eval_name}'...")
#             # y_test son las acciones de comportamiento del conjunto de prueba
#             eval_metrics = eval_instance.evaluate_policy(
#                 policy=trained_model_wrapper, # El DRLModelWrapperPyTorch
#                 x_cgm_test=all_data['test']['x_cgm'],
#                 x_other_test=all_data['test']['x_other'],
#                 y_actions_test=all_data['test']['y'],
#                 context_test_data=all_data['test']['context'], # Pasar datos de contexto
#                 simulator=simulator 
#             )
#             offline_results[f"{model_name}_{eval_name}"] = eval_metrics
#             print_debug(f"Métricas de {eval_name} para {model_name}: {eval_metrics}")
# else:
#     print_warning("No hay evaluadores offline (FQE/DRE) activados.")

# # Visualización de resultados
# coloured("\n==== VISUALIZACIÓN DE RESULTADOS ====", 'cyan', 'bold')

# # 1. Visualizar historial de entrenamiento para cada modelo
# for model_name, history in histories.items():
#     if not isinstance(history, dict) or not history:
#         continue
        
#     plt.figure(figsize=(12, 8))
    
#     # Gráfico de pérdida
#     plt.subplot(2, 2, 1)
#     if 'loss' in history:
#         plt.plot(history['loss'], label='Training Loss')
#     if 'val_loss' in history:
#         plt.plot(history['val_loss'], label='Validation Loss')
#     plt.title(f'{model_name} - Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)
    
#     # Gráfico de error absoluto medio
#     plt.subplot(2, 2, 2)
#     if 'mae' in history:
#         plt.plot(history['mae'], label='Training MAE')
#     if 'val_mae' in history:
#         plt.plot(history['val_mae'], label='Validation MAE')
#     plt.title(f'{model_name} - MAE')
#     plt.xlabel('Epochs')
#     plt.ylabel('MAE')
#     plt.legend()
#     plt.grid(True)
    
#     # Gráfico de pérdidas específicas para DRL (actor, critic)
#     plt.subplot(2, 2, 3)
#     for metric in ['actor_loss', 'critic_loss', 'q_loss', 'policy_loss', 'value_loss']:
#         if metric in history and history[metric]:
#             plt.plot(history[metric], label=metric.replace('_', ' ').title())
#     plt.title(f'{model_name} - DRL Losses')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)
    
#     # Gráfico de recompensas (si es aplicable)
#     plt.subplot(2, 2, 4)
#     for metric in ['reward', 'average_reward', 'episode_reward']:
#         if metric in history and history[metric]:
#             plt.plot(history[metric], label=metric.replace('_', ' ').title())
#     plt.title(f'{model_name} - Rewards')
#     plt.xlabel('Epochs')
#     plt.ylabel('Reward')
#     plt.legend()
#     plt.grid(True)
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(FIGURES_DIR, f'{model_name}_training.png'))
#     plt.close()

# # 2. Visualización de métricas clínicas
# if clinical_results:
#     plt.figure(figsize=(14, 8))
    
#     # Gráfico de tiempo en rango
#     models = list(clinical_results.keys())
#     tir_values = [clinical_results[m]['time_in_range'] for m in models]
#     tbr_values = [clinical_results[m]['time_below_range'] for m in models]
#     tsb_values = [clinical_results[m]['time_severe_below'] for m in models]  # Nuevo
#     tar_values = [clinical_results[m]['time_above_range'] for m in models]
#     tsa_values = [clinical_results[m]['time_severe_above'] for m in models]  # Nuevo

#     # Barras apiladas
#     plt.subplot(1, 2, 1)
#     bars_tsb = plt.bar(models, tsb_values, label='Hipoglucemia Severa (<54 mg/dL)', color='darkred')
#     bars_tbr = plt.bar(models, tbr_values, bottom=tsb_values, label='Hipoglucemia (54-70 mg/dL)', color='red')

#     # Posición para TIR
#     tsb_tbr = [tsb + tbr for tsb, tbr in zip(tsb_values, tbr_values)]
#     bars_tir = plt.bar(models, tir_values, bottom=tsb_tbr, label='Tiempo en Rango (70-180 mg/dL)', color='green')

#     # Posición para TAR
#     tsb_tbr_tir = [tsb + tbr + tir for tsb, tbr, tir in zip(tsb_values, tbr_values, tir_values)]
#     bars_tar = plt.bar(models, tar_values, bottom=tsb_tbr_tir, label='Hiperglucemia (180-250 mg/dL)', color='orange')

#     # Posición para TSA
#     tsb_tbr_tir_tar = [tsb + tbr + tir + tar for tsb, tbr, tir, tar in zip(tsb_values, tbr_values, tir_values, tar_values)]
#     bars_tsa = plt.bar(models, tsa_values, bottom=tsb_tbr_tir_tar, label='Hiperglucemia Severa (>250 mg/dL)', color='darkred')
    
#     plt.title('Distribución de Métricas Clínicas')
#     plt.xlabel('Modelo')
#     plt.ylabel('Porcentaje (%)')
#     plt.legend()
#     plt.ylim(0, 100)
#     plt.xticks(rotation=45)
#     plt.grid(True, axis='y')
    
#     # Comparación de tiempo en rango
#     plt.subplot(1, 2, 2)
#     plt.bar(models, tir_values, color='green')
#     plt.axhline(y=70, color='r', linestyle='--', label='Objetivo (70%)')
#     plt.title('Tiempo en Rango por Modelo')
#     plt.xlabel('Modelo')
#     plt.ylabel('Tiempo en Rango (%)')
#     plt.xticks(rotation=45)
#     plt.ylim(0, 100)
#     plt.legend()
#     plt.grid(True, axis='y')
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(FIGURES_DIR, 'clinical_metrics.png'))
#     plt.close()

# # 3. Visualización de evaluaciones offline (si existen)
# if 'offline_results' in locals() and offline_results:
#     # Asumiendo que 'active_evaluators.keys()' le da los nombres de los evaluadores como ['pt_fqe', 'pt_dr']
#     # y 'trained_models.keys()' le da los nombres base de los modelos como ['DDPG', 'SAC']
    
#     evaluator_names_for_plotting: List[str] = list(active_evaluators.keys())
#     # model_names_base: List[str] = list(trained_models.keys()) # Nombres base de los modelos

#     for evaluator_name in evaluator_names_for_plotting: # ej. "pt_fqe"
#         plt.figure(figsize=(10, 6))
        
#         # Datos para el gráfico del evaluador actual
#         plot_model_labels: List[str] = []
#         plot_values: List[float] = []
#         plot_lower_bounds: List[float] = []
#         plot_upper_bounds: List[float] = []

#         # Iterar sobre todas las claves combinadas en offline_results
#         # (originalmente 'models = list(offline_results.keys())')
#         for composite_key in offline_results.keys():
#             # Verificar si esta clave pertenece al evaluador actual que se está graficando
#             # Se asume que composite_key es "NombreModelo_NombreEvaluador"
#             # y 'evaluator_name' es el nombre del evaluador actual del bucle externo
#             if composite_key.endswith(f"_{evaluator_name}"):
#                 # Extraer la parte del nombre del modelo para la etiqueta
#                 # rpartition devuelve una tupla (parte_antes, separador, parte_después)
#                 model_name_part: str = composite_key.rpartition(f"_{evaluator_name}")[0]
                
#                 metrics_data: Optional[Dict[str, float]] = offline_results.get(composite_key)
                
#                 plot_model_labels.append(model_name_part) # Etiqueta para el eje x

#                 if isinstance(metrics_data, dict):
#                     est_value: float = metrics_data.get('estimated_value', 0.0)
#                     plot_values.append(est_value)
#                     # Usar est_value como default si los límites de confianza no están presentes
#                     plot_lower_bounds.append(metrics_data.get('confidence_lower', est_value))
#                     plot_upper_bounds.append(metrics_data.get('confidence_upper', est_value))
#                 elif isinstance(metrics_data, (float, int)):
#                     val: float = float(metrics_data)
#                     print_warning(f"El resultado offline para '{composite_key}' es un valor directo ({val}), no un dict. Se usará para el valor estimado y los límites.")
#                     plot_values.append(val)
#                     plot_lower_bounds.append(val)
#                     plot_upper_bounds.append(val)
#                 else: # metrics_data es None u otro tipo inesperado
#                     print_warning(f"Datos inesperados o métricas faltantes para '{composite_key}'. Se usarán valores por defecto (0.0).")
#                     plot_values.append(0.0)
#                     plot_lower_bounds.append(0.0)
#                     plot_upper_bounds.append(0.0)
        
#         if not plot_model_labels:
#             print_warning(f"No se encontraron datos para graficar para el evaluador '{evaluator_name}'.")
#             plt.close()
#             continue

#         # models = list(offline_results.keys()) # 'models' eran las claves compuestas
#         # values = [offline_results[m][evaluator].get('estimated_value', 0) for m in models]
#         # lower = [offline_results[m][evaluator].get('confidence_lower', 0) for m in models]
#         # upper = [offline_results[m][evaluator].get('confidence_upper', 0) for m in models]
#         # Ahora se usan plot_model_labels, plot_values, plot_lower_bounds, plot_upper_bounds

#         plt.bar(plot_model_labels, plot_values, color='skyblue')
#         # Asegurarse que los errores son no negativos
#         yerr_lower_values = [max(0, v - l) for v, l in zip(plot_values, plot_lower_bounds)]
#         yerr_upper_values = [max(0, u - v) for v, u in zip(plot_upper_bounds, plot_values)]

#         plt.errorbar(plot_model_labels, plot_values, yerr=[
#             yerr_lower_values,
#             yerr_upper_values
#         ], fmt='o', color='black', capsize=5)
        
#         plt.title(f'Evaluación {evaluator_name}')
#         plt.xlabel('Modelo')
#         plt.ylabel('Valor Estimado')
#         plt.xticks(rotation=45)
#         plt.grid(True, axis='y')
#         plt.tight_layout()
        
#         plt.savefig(os.path.join(FIGURES_DIR, f'{evaluator_name}_evaluation.png'))
#         plt.close()

# # Generación de Reporte
# coloured("\n==== GENERACIÓN DE REPORTE ====", 'cyan', 'bold')


# # Finalización del proceso

# coloured("\n==== PROCESO COMPLETADO ====", 'cyan', 'bold')
# coloured(f"Resultados guardados en: {RESULTS_SAVE_DIR}", 'green')
# coloured(f"Visualizaciones guardadas en: {FIGURES_DIR}", 'green')