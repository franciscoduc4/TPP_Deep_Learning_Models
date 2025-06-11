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
from typing import Dict, List, Tuple, Any
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
from custom.printer import coloured, print_debug, print_error, print_info, print_warning

# Configuración 
from config.params import FRAMEWORK, PROCESSING, MODELS, MODELS_USAGE, EVALUATE, EVALUATE_USAGE

# Procesamiento
from processing.pandas import preprocess_data as pd_preprocess, split_data as pd_split
from processing.polars import preprocess_data as pl_preprocess

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
(x_cgm_train, x_cgm_val, x_cgm_test, x_other_train, x_other_val, x_other_test, 
 x_subject_train, x_subject_val, x_subject_test, y_train, y_val, y_test, 
 x_subject_test, scaler_cgm, scaler_other, scaler_y) = (None, None, None, None, None, None, 
                                                        None, None, None, None, None, None, 
                                                        None, None, None, None)

if PROCESSING == "pandas":
    coloured("Procesando datos con pandas...", 'blue', 'bold')
    df_pd: pd.DataFrame = pd_preprocess()
    # No usamos pd_split, trabajamos con todo el dataset
    
    # Extraer características
    x_cgm = np.stack(df_pd['cgm_window'].to_numpy())
    x_other = df_pd.drop(['cgm_window', 'bolus'], axis=1).to_numpy()
    y = df_pd['bolus'].to_numpy()
    
    # # Normalizar datos si es necesario
    # scaler_cgm = StandardScaler().fit(x_cgm.reshape(x_cgm.shape[0], -1))
    # scaler_other = StandardScaler().fit(x_other)
    # scaler_y = StandardScaler().fit(y.reshape(-1, 1))
    
    # x_cgm = scaler_cgm.transform(x_cgm.reshape(x_cgm.shape[0], -1)).reshape(x_cgm.shape)
    # x_other = scaler_other.transform(x_other)
    y = y.reshape(-1, 1).flatten()
elif PROCESSING == "polars":
    coloured("Procesando datos con polars...", 'blue', 'bold')
    df_pl: pl.DataFrame = pl_preprocess()
    
    # Validar cobertura de datos contextuales
    coloured("\n==== VALIDACIÓN DE DATOS CONTEXTUALES ====", 'cyan', 'bold')
    contextual_coverage = validate_contextual_data_coverage(df_pl)
    # Preparar datos con división temporal apropiada
    coloured("\n==== PREPARACIÓN DE DATOS PARA DRL ====", 'cyan', 'bold')
    # Se asume que prepare_data_for_drl_training ahora retorna también los diccionarios de contexto
    (x_cgm_train, x_other_train, y_train, context_train,
     x_cgm_val, x_other_val, y_val, context_val,
     x_cgm_test, x_other_test, y_test, context_test) = prepare_data_for_drl_training(df_pl)
    
    # Mostrar información sobre características seleccionadas
    feature_groups = get_feature_groups()

# Mostrar información sobre los datos preparados
coloured("\n==== INFORMACIÓN DE DATOS PREPARADOS ====", 'cyan', 'bold')
coloured(f"Entrenamiento - CGM: {x_cgm_train.shape}, Otros: {x_other_train.shape}, Target: {y_train.shape}", 'green')
coloured(f"Validación    - CGM: {x_cgm_val.shape}, Otros: {x_other_val.shape}, Target: {y_val.shape}", 'green')
coloured(f"Test          - CGM: {x_cgm_test.shape}, Otros: {x_other_test.shape}, Target: {y_test.shape}", 'green')
if context_train is not None:
    coloured(f"Contexto Train - Claves: {list(context_train.keys())}, Muestra forma: {context_train[list(context_train.keys())[0]].shape if context_train else 'N/A'}", 'green')
if context_val is not None:
    coloured(f"Contexto Val   - Claves: {list(context_val.keys())}, Muestra forma: {context_val[list(context_val.keys())[0]].shape if context_val else 'N/A'}", 'green')
if context_test is not None:
    coloured(f"Contexto Test  - Claves: {list(context_test.keys())}, Muestra forma: {context_test[list(context_test.keys())[0]].shape if context_test else 'N/A'}", 'green')
# Mejorar características utilizando la función del framework seleccionado
coloured("\n==== GENERACIÓN DE CARACTERÍSTICAS ADICIONALES ====", 'cyan', 'bold')
x_cgm_train_enhanced, x_other_train_enhanced = enhance_features(x_cgm_train, x_other_train)
x_cgm_val_enhanced, x_other_val_enhanced = enhance_features(x_cgm_val, x_other_val)
x_cgm_test_enhanced, x_other_test_enhanced = enhance_features(x_cgm_test, x_other_test)

coloured("Forma de datos mejorados:", 'green')
coloured(f"  Train - CGM: {x_cgm_train_enhanced.shape}, Otros: {x_other_train_enhanced.shape}", 'green')
coloured(f"  Val   - CGM: {x_cgm_val_enhanced.shape}, Otros: {x_other_val_enhanced.shape}", 'green')
coloured(f"  Test  - CGM: {x_cgm_test_enhanced.shape}, Otros: {x_other_test_enhanced.shape}", 'green')

# Definir formas de entrada para los modelos
input_shapes = (x_cgm_train_enhanced.shape[1:], x_other_train_enhanced.shape[1:])
coloured(f"Formas de entrada para los modelos: CGM {input_shapes[0]}, Otros {input_shapes[1]}", 'green')

# Estructurar all_data para pasar a train_multiple_models
all_data = {
    'train': {
        'x_cgm': x_cgm_train_enhanced,
        'x_other': x_other_train_enhanced,
        'y': y_train,
        'context': context_train
    },
    'val': {
        'x_cgm': x_cgm_val_enhanced,
        'x_other': x_other_val_enhanced,
        'y': y_val,
        'context': context_val
    },
    'test': {
        'x_cgm': x_cgm_test_enhanced,
        'x_other': x_other_test_enhanced,
        'y': y_test,
        'context': context_test
    }
}

# Entrenamiento de modelos
coloured("\n==== ENTRENAMIENTO DE MODELOS ====", 'cyan', 'bold')

histories, predictions, clinical_metrics, trained_models = train_multiple_models(
    model_creators=use_models,
    input_shapes=input_shapes,
    all_data=all_data,
    models_dir=MODELS_SAVE_DIR
)

# Creación del Ensamble
coloured("\n==== CREACIÓN DEL ENSAMBLE ====", 'cyan', 'bold')
ensemble_prediction = None
ensemble_metrics = None
clinical_results = {}

# Inicializar simulador para métricas clínicas
simulator = GlucoseSimulator()

# Extraer valores iniciales de glucosa y carbohidratos del conjunto de prueba
initial_glucose = np.array([x_cgm_test_enhanced[i, -1, 0] for i in range(len(x_cgm_test_enhanced))])

# Manera más robusta de identificar la columna de carbohidratos
# Ahora se extrae del diccionario de contexto del conjunto de prueba
carb_intake_key = 'meal_carbs' # o 'carb_intake' si se renombra en prepare_drl_data
if context_test and carb_intake_key in context_test:
    carb_intake = context_test[carb_intake_key]
    if carb_intake.ndim > 1 and carb_intake.shape[1] == 1: # Asegurar que sea 1D
        carb_intake = carb_intake.flatten()
elif x_other_test_enhanced.shape[1] > 0:
    # Fallback si no está en el contexto, intentar buscar en x_other_test_enhanced
    # Esto requeriría conocer el índice o nombre de la columna de carbohidratos en x_other_test_enhanced
    print_warning(f"'{carb_intake_key}' no encontrado en context_test. Intentando fallback (puede ser incorrecto).")
    # Asumir un índice (esto es frágil, idealmente carb_intake siempre vendrá de context_test)
    # Por ejemplo, si 'meal_carbs_log1p' fuera la primera columna de x_other_test_enhanced:
    # carb_intake_idx = 0 # Esto es un placeholder, debe ajustarse si se usa este fallback
    # carb_intake = np.array([x_other_test_enhanced[i, carb_intake_idx] for i in range(len(x_other_test_enhanced))])
    # Es mejor asegurar que context_test contenga la ingesta de carbohidratos.
    # Por ahora, si no está en el contexto, se podría generar un error o un valor por defecto.
    print_error("No se pudo determinar la ingesta de carbohidratos para la evaluación clínica. Usando valores por defecto (0).")
    carb_intake = np.zeros(len(x_cgm_test_enhanced))
else:
    print_error("No se pudo determinar la ingesta de carbohidratos para la evaluación clínica. Usando valores por defecto (0).")
    carb_intake = np.zeros(len(x_cgm_test_enhanced))
# Evaluar métricas clínicas para cada modelo
for model_name, model_pred in predictions.items():
    clinical_metrics = evaluate_clinical_metrics(
        simulator=simulator,
        predictions=model_pred,
        initial_glucose=initial_glucose,
        carb_intake=carb_intake
    )
    clinical_results[model_name] = clinical_metrics
    
    coloured(f"\nMétricas clínicas para {model_name}:", 'green', 'bold')
    coloured(f"  Tiempo Severamente Bajo Rango: {clinical_metrics['time_severe_below']:.2f}%", 'red')
    coloured(f"  Tiempo Bajo Rango: {clinical_metrics['time_below_range']:.2f}%", 'yellow')
    coloured(f"  Tiempo en Rango: {clinical_metrics['time_in_range']:.2f}%", 'green')
    coloured(f"  Tiempo Sobre Rango: {clinical_metrics['time_above_range']:.2f}%", 'yellow')
    coloured(f"  Tiempo Severamente Sobre Rango: {clinical_metrics['time_severe_above']:.2f}%", 'red')
    
    # Guardar métricas clínicas
    with open(os.path.join(RESULTS_SAVE_DIR, f"{model_name}_clinical_metrics.json"), 'w') as f:
        json.dump(clinical_metrics, f, indent=2)

# Crear ensamble si hay más de un modelo
if len(predictions) > 1:
    coloured("\nCreando ensamble optimizado para métricas clínicas...", 'blue', 'bold')
    
    # Optimizar pesos para tiempo en rango
    weights, ensemble_prediction = optimize_ensemble_weights_clinical(
        predictions=predictions,
        initial_glucose=initial_glucose,
        carb_intake=carb_intake,
        simulator=simulator,
        y_true=y_test
    )
    
    # Calcular métricas para el ensamble
    ensemble_metrics = calculate_metrics(y_test, ensemble_prediction)
    ensemble_clinical = evaluate_clinical_metrics(
        simulator=simulator,
        predictions=ensemble_prediction,
        initial_glucose=initial_glucose,
        carb_intake=carb_intake
    )
    
    # Mostrar pesos y métricas del ensamble
    coloured("\nPesos del ensamble:", 'green', 'bold')
    for i, (model_name, weight) in enumerate(zip(predictions.keys(), weights)):
        coloured(f"  {model_name}: {weight:.4f}", 'green')
    
    coloured("\nMétricas clínicas del ensamble:", 'green', 'bold')
    coloured(f"  Tiempo en Rango: {ensemble_clinical['time_in_range']:.2f}%", 'green')
    coloured(f"  Tiempo Bajo Rango: {ensemble_clinical['time_below_range']:.2f}%", 'yellow')
    coloured(f"  Tiempo Sobre Rango: {ensemble_clinical['time_above_range']:.2f}%", 'yellow')
    
    # Guardar predicciones y métricas del ensamble
    clinical_results[CONST_ENSEMBLE] = ensemble_clinical
    np.save(os.path.join(RESULTS_SAVE_DIR, f"{CONST_ENSEMBLE}_predictions.npy"), ensemble_prediction)
    with open(os.path.join(RESULTS_SAVE_DIR, f"{CONST_ENSEMBLE}_metrics.json"), 'w') as f:
        json.dump({**ensemble_metrics, **ensemble_clinical}, f, indent=2)
else:
    coloured("No se puede crear ensamble con menos de 2 modelos", 'yellow', 'bold')
    
# Evaluación con FQE y Doubly Robust
coloured("\n==== EVALUACIÓN OFFLINE RL ====", 'cyan', 'bold')

# Seleccionar evaluadores activados
active_evaluators = {}
for eval_name, use in EVALUATE_USAGE.items():
    if use:
        evaluator_fn = EVALUATE[eval_name]
        if is_model_creator(evaluator_fn):
            evaluator_fn = evaluator_fn()
        active_evaluators[eval_name] = evaluator_fn
        coloured(f"Evaluador {eval_name} activado", 'green', 'bold')
    else:
        coloured(f"Evaluador {eval_name} desactivado", 'red', 'bold')

# Realizar evaluación si hay evaluadores activos
if active_evaluators:
    offline_results = {}
    
    # Evaluar cada modelo con cada evaluador activo
    for model_name, model in trained_models.items():
        coloured(f"\nEvaluando modelo {model_name}...", 'blue')
        model_results = {}
        
        for eval_name, evaluator_creator in active_evaluators.items():
            coloured(f"  Aplicando evaluador {eval_name}...", 'blue')
            eval_instance = evaluator_creator(input_shapes[0], input_shapes[1])
            
            # Entrenar evaluador con los datos de entrenamiento
            eval_instance.fit(
                x_cgm_train_enhanced, 
                x_other_train_enhanced, 
                y_train,
                validation_data=((x_cgm_val_enhanced, x_other_val_enhanced), y_val),
                epochs=50,
                batch_size=64
            )
            
            # Evaluar política del modelo usando directamente el modelo entrenado
            eval_metrics = eval_instance.evaluate_policy(
                policy=model,
                x_cgm_test=x_cgm_test_enhanced,
                x_other_test=x_other_test_enhanced,
                y_test=y_test,
                simulator=simulator
            )
            
            model_results[eval_name] = eval_metrics
            
            # Mostrar métricas principales
            coloured(f"    Valor Estimado: {eval_metrics.get('estimated_value', 0):.4f}", 'green')
            coloured(f"    Límite Inferior de Confianza: {eval_metrics.get('confidence_lower', 0):.4f}", 'yellow')
            coloured(f"    Límite Superior de Confianza: {eval_metrics.get('confidence_upper', 0):.4f}", 'yellow')
        
        offline_results[model_name] = model_results
        
        # Guardar resultados
        with open(os.path.join(RESULTS_SAVE_DIR, f"{model_name}_offline_eval.json"), 'w') as f:
            json.dump(model_results, f, indent=2)
    
    # Evaluar ensamble si existe
    if ensemble_prediction is not None:
        coloured("\nEvaluando ensamble...", 'blue')
        ensemble_offline_results = {}
        
        # Crear un wrapper temporal para el ensamble
        ensemble_wrapper = type('EnsembleWrapper', (), {
            'predict': lambda x_cgm, x_other: ensemble_prediction
        })()
        
        for eval_name, evaluator in active_evaluators.items():
            eval_instance = evaluator(input_shapes[0], input_shapes[1])
            eval_instance.fit(
                x_cgm_train_enhanced, x_other_train_enhanced, y_train,
                validation_data=((x_cgm_val_enhanced, x_other_val_enhanced), y_val)
            )
            
            eval_metrics = eval_instance.evaluate_policy(
                ensemble_wrapper, 
                x_cgm_test_enhanced, 
                x_other_test_enhanced, 
                y_test,
                simulator=simulator
            )
            
            ensemble_offline_results[eval_name] = eval_metrics
        
        offline_results[CONST_ENSEMBLE] = ensemble_offline_results
        
        # Guardar resultados
        with open(os.path.join(RESULTS_SAVE_DIR, f"{CONST_ENSEMBLE}_offline_eval.json"), 'w') as f:
            json.dump(ensemble_offline_results, f, indent=2)
else:
    coloured("No hay evaluadores offline activos", 'yellow', 'bold')

# Visualización de resultados
coloured("\n==== VISUALIZACIÓN DE RESULTADOS ====", 'cyan', 'bold')

# 1. Visualizar historial de entrenamiento para cada modelo
for model_name, history in histories.items():
    if not isinstance(history, dict) or not history:
        continue
        
    plt.figure(figsize=(12, 8))
    
    # Gráfico de pérdida
    plt.subplot(2, 2, 1)
    if 'loss' in history:
        plt.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Gráfico de error absoluto medio
    plt.subplot(2, 2, 2)
    if 'mae' in history:
        plt.plot(history['mae'], label='Training MAE')
    if 'val_mae' in history:
        plt.plot(history['val_mae'], label='Validation MAE')
    plt.title(f'{model_name} - MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    
    # Gráfico de pérdidas específicas para DRL (actor, critic)
    plt.subplot(2, 2, 3)
    for metric in ['actor_loss', 'critic_loss', 'q_loss', 'policy_loss', 'value_loss']:
        if metric in history and history[metric]:
            plt.plot(history[metric], label=metric.replace('_', ' ').title())
    plt.title(f'{model_name} - DRL Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Gráfico de recompensas (si es aplicable)
    plt.subplot(2, 2, 4)
    for metric in ['reward', 'average_reward', 'episode_reward']:
        if metric in history and history[metric]:
            plt.plot(history[metric], label=metric.replace('_', ' ').title())
    plt.title(f'{model_name} - Rewards')
    plt.xlabel('Epochs')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'{model_name}_training.png'))
    plt.close()

# 2. Visualización de métricas clínicas
if clinical_results:
    plt.figure(figsize=(14, 8))
    
    # Gráfico de tiempo en rango
    models = list(clinical_results.keys())
    tir_values = [clinical_results[m]['time_in_range'] for m in models]
    tbr_values = [clinical_results[m]['time_below_range'] for m in models]
    tsb_values = [clinical_results[m]['time_severe_below'] for m in models]  # Nuevo
    tar_values = [clinical_results[m]['time_above_range'] for m in models]
    tsa_values = [clinical_results[m]['time_severe_above'] for m in models]  # Nuevo

    # Barras apiladas
    plt.subplot(1, 2, 1)
    bars_tsb = plt.bar(models, tsb_values, label='Hipoglucemia Severa (<54 mg/dL)', color='darkred')
    bars_tbr = plt.bar(models, tbr_values, bottom=tsb_values, label='Hipoglucemia (54-70 mg/dL)', color='red')

    # Posición para TIR
    tsb_tbr = [tsb + tbr for tsb, tbr in zip(tsb_values, tbr_values)]
    bars_tir = plt.bar(models, tir_values, bottom=tsb_tbr, label='Tiempo en Rango (70-180 mg/dL)', color='green')

    # Posición para TAR
    tsb_tbr_tir = [tsb + tbr + tir for tsb, tbr, tir in zip(tsb_values, tbr_values, tir_values)]
    bars_tar = plt.bar(models, tar_values, bottom=tsb_tbr_tir, label='Hiperglucemia (180-250 mg/dL)', color='orange')

    # Posición para TSA
    tsb_tbr_tir_tar = [tsb + tbr + tir + tar for tsb, tbr, tir, tar in zip(tsb_values, tbr_values, tir_values, tar_values)]
    bars_tsa = plt.bar(models, tsa_values, bottom=tsb_tbr_tir_tar, label='Hiperglucemia Severa (>250 mg/dL)', color='darkred')
    
    plt.title('Distribución de Métricas Clínicas')
    plt.xlabel('Modelo')
    plt.ylabel('Porcentaje (%)')
    plt.legend()
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    
    # Comparación de tiempo en rango
    plt.subplot(1, 2, 2)
    plt.bar(models, tir_values, color='green')
    plt.axhline(y=70, color='r', linestyle='--', label='Objetivo (70%)')
    plt.title('Tiempo en Rango por Modelo')
    plt.xlabel('Modelo')
    plt.ylabel('Tiempo en Rango (%)')
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'clinical_metrics.png'))
    plt.close()

# 3. Visualización de evaluaciones offline (si existen)
if 'offline_results' in locals() and offline_results:
    evaluators = list(next(iter(offline_results.values())).keys())
    
    for evaluator in evaluators:
        plt.figure(figsize=(10, 6))
        
        models = list(offline_results.keys())
        values = [offline_results[m][evaluator].get('estimated_value', 0) for m in models]
        lower = [offline_results[m][evaluator].get('confidence_lower', 0) for m in models]
        upper = [offline_results[m][evaluator].get('confidence_upper', 0) for m in models]
        
        plt.bar(models, values, color='skyblue')
        plt.errorbar(models, values, yerr=[
            [v - l for v, l in zip(values, lower)],
            [u - v for v, u in zip(upper, values)]
        ], fmt='o', color='black', capsize=5)
        
        plt.title(f'Evaluación {evaluator}')
        plt.xlabel('Modelo')
        plt.ylabel('Valor Estimado')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        plt.tight_layout()
        
        plt.savefig(os.path.join(FIGURES_DIR, f'{evaluator}_evaluation.png'))
        plt.close()

# Generación de Reporte
coloured("\n==== GENERACIÓN DE REPORTE ====", 'cyan', 'bold')


# Finalización del proceso

coloured("\n==== PROCESO COMPLETADO ====", 'cyan', 'bold')
coloured(f"Resultados guardados en: {RESULTS_SAVE_DIR}", 'green')
coloured(f"Visualizaciones guardadas en: {FIGURES_DIR}", 'green')