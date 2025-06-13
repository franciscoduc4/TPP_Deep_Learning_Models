import os, sys
import numpy as np
import polars as pl
from typing import Dict, List, Tuple, Callable, Optional, Any, Union
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.optimize import minimize
from joblib import Parallel, delayed
from constants.constants import CONST_VAL_LOSS, CONST_LOSS, CONST_METRIC_MAE, CONST_METRIC_RMSE, CONST_METRIC_R2, CONST_MODELS, CONST_BEST_PREFIX, CONST_LOGS_DIR, CONST_DEFAULT_EPOCHS, CONST_DEFAULT_BATCH_SIZE, CONST_DEFAULT_SEED, CONST_FIGURES_DIR, CONST_MODEL_TYPES, CONST_FRAMEWORKS, CONST_DURATION_HOURS, CONTEXT_FEATURE_ORDER, HYPER_PENALTY_BASE, HYPO_PENALTY_BASE, MAX_REWARD, SEVERE_HYPER_PENALTY, SEVERE_HYPO_PENALTY, SEVERE_HYPOGLYCEMIA_THRESHOLD, HYPOGLYCEMIA_THRESHOLD, HYPERGLYCEMIA_THRESHOLD, SEVERE_HYPERGLYCEMIA_THRESHOLD, IDEAL_LOWER_BOUND, IDEAL_UPPER_BOUND
from custom.DeepReinforcementLearning.drl_pt import DRLModelWrapperPyTorch
from custom.printer import print_debug, print_header, print_info, print_success, print_warning, print_error
from validation.simulator import GlucoseSimulator

AllData = Dict[str, Optional[Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]]] 

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     simulator: GlucoseSimulator = None, initial_glucose=None, carb_intake=None,
                     context_data_for_clinical: Optional[Dict[str, np.ndarray]] = None, # AÑADIDO
                     model_wrapper_for_clinical: Optional[Any] = None
                     ) -> Dict[str, float]:
    """
    Calcula métricas de rendimiento para las predicciones del modelo.
    
    Parámetros:
    -----------
    y_true : np.ndarray
        Valores objetivo verdaderos (dosis de insulina reales)
    y_pred : np.ndarray
        Valores predichos por el modelo (dosis predichas)
    simulator : GlucoseSimulator, opcional
        Simulador para cálculo de métricas clínicas
    initial_glucose : np.ndarray, opcional
        Valores iniciales de glucosa para simulación
    carb_intake : np.ndarray, opcional
        Valores de ingesta de carbohidratos para simulación
        
    Retorna:
    --------
    Dict[str, float]
        Diccionario con métricas estándar y clínicas
    """
    
    if simulator is not None and initial_glucose is not None and carb_intake is not None and context_data_for_clinical is not None and model_wrapper_for_clinical is not None:
        try:
            # Para evaluación clínica, necesitamos predecir dosis usando el modelo y contexto completo, luego simular. y_pred aquí son las dosis ya predichas (posiblemente sin contexto completo).
            # La función evaluate_clinical del wrapper se encarga de la predicción contextual.
            
            # Asegurarse que initial_glucose y context_data_for_clinical tengan la misma cantidad de muestras que se usarán para la predicción.
            # Si y_pred es el resultado de model_wrapper.predict(x_cgm, x_other), entonces
            # x_cgm, x_other, context_data_for_clinical, initial_glucose deben estar alineados.
            
            # Nota: evaluate_clinical_metrics (la función global) espera dosis ya predichas.
            # ModelWrapper.evaluate_clinical hace la predicción internamente.
            # Aquí, si tenemos el model_wrapper, podemos llamar a su método evaluate_clinical.
            
            num_samples_for_clinical = len(initial_glucose) # Asumimos que initial_glucose marca las muestras
            
            # Necesitamos x_cgm y x_other para que el wrapper prediga internamente
            # Estos no se pasan directamente a calculate_metrics, lo cual es un problema para este flujo.
            # Por ahora, si model_wrapper_for_clinical está presente, asumimos que y_pred son las dosis que se usarían con el simulador, y que initial_glucose y carb_intake están alineados.
            
            # Opción 1: Usar la función global evaluate_clinical_metrics (requiere y_pred como dosis)
            # clinical_eval_metrics = evaluate_clinical_metrics(
            #     simulator, y_pred, initial_glucose, carb_intake
            # )
            
            # Opción 2: Usar el método del wrapper (más encapsulado, pero requiere x_cgm, x_other)
            # Esta función (calculate_metrics) no recibe x_cgm, x_other.
            # Esto indica una posible necesidad de refactorizar cómo se llaman las métricas clínicas.
            # Por ahora, vamos a asumir que y_pred son las dosis a simular y usamos la función global.
            # Si se quiere usar model_wrapper.evaluate_clinical(), esa llamada debería hacerse en otro lugar
            # donde x_cgm y x_other estén disponibles.

            # Usando la función global evaluate_clinical_metrics con las y_pred ya calculadas:
            clinical_eval_metrics = evaluate_clinical_metrics(
                 simulator, y_pred[:num_samples_for_clinical], 
                 initial_glucose[:num_samples_for_clinical], 
                 carb_intake[:num_samples_for_clinical] # Asumimos que carb_intake también está alineado
            )
        except Exception as e:
            print_warning(f"No se pudieron calcular las métricas clínicas: {e}")
    
    return clinical_eval_metrics

class ClinicalMetricsEvaluator:
    """
    Evaluador de métricas clínicas para modelos de dosificación de insulina.
    
    Calcula métricas relevantes clínicamente como Tiempo en Rango (TIR),
    Tiempo por Encima del Rango (TAR), Tiempo por Debajo del Rango (TBR), etc.
    """

    @staticmethod
    def calculate_time_metrics(glucose_values: np.ndarray) -> Dict[str, float]:
        """
        Calcula el porcentaje de tiempo en diferentes rangos glucémicos.

        Parámetros:
        -----------
        glucose_values : np.ndarray
            Valores de glucosa en mg/dL

        Retorna:
        --------
        Dict[str, float]
            Diccionario con porcentajes de tiempo en varios rangos.
        """
        if glucose_values.size == 0:
            return {
                'time_severe_below': 0.0,
                'time_below_range': 0.0,
                'time_in_range_low': 0.0,
                'time_in_ideal_range': 0.0,
                'time_in_range_high': 0.0,
                'time_above_range': 0.0,
                'time_severe_above': 0.0,
                'time_total_in_range': 0.0,
            }

        time_severe_below = np.mean(glucose_values < SEVERE_HYPOGLYCEMIA_THRESHOLD) * 100.0
        time_below_range = np.mean((glucose_values >= SEVERE_HYPOGLYCEMIA_THRESHOLD) & (glucose_values < HYPOGLYCEMIA_THRESHOLD)) * 100.0
        
        time_in_range_low = np.mean((glucose_values >= HYPOGLYCEMIA_THRESHOLD) & (glucose_values < IDEAL_LOWER_BOUND)) * 100.0
        time_in_ideal_range = np.mean((glucose_values >= IDEAL_LOWER_BOUND) & (glucose_values <= IDEAL_UPPER_BOUND)) * 100.0
        time_in_range_high = np.mean((glucose_values > IDEAL_UPPER_BOUND) & (glucose_values <= HYPERGLYCEMIA_THRESHOLD)) * 100.0
        
        time_above_range = np.mean((glucose_values > HYPERGLYCEMIA_THRESHOLD) & (glucose_values <= SEVERE_HYPERGLYCEMIA_THRESHOLD)) * 100.0
        time_severe_above = np.mean(glucose_values > SEVERE_HYPERGLYCEMIA_THRESHOLD) * 100.0

        time_total_in_range = time_in_range_low + time_in_ideal_range + time_in_range_high

        return {
            'time_severe_below': float(time_severe_below),
            'time_below_range': float(time_below_range),
            'time_in_range_low': float(time_in_range_low),
            'time_in_ideal_range': float(time_in_ideal_range),
            'time_in_range_high': float(time_in_range_high),
            'time_above_range': float(time_above_range),
            'time_severe_above': float(time_severe_above),
            'time_total_in_range': float(time_total_in_range), # TIR convencional 70-180
        }

    @staticmethod
    def calculate_glucose_variability(glucose_values: np.ndarray) -> Dict[str, float]:
        """
        Calcula métricas de variabilidad de glucosa.
        
        Parámetros:
        -----------
        glucose_values : np.ndarray
            Valores de glucosa en mg/dL
            
        Retorna:
        --------
        Dict[str, float]
            Diccionario con métricas de variabilidad (SD, CV, MAGE, etc.)
        """
        if glucose_values.size == 0:
            return {
                'mean_glucose': 0.0,
                'median_glucose': 0.0,
                'std_glucose': 0.0,
                'cv_glucose': 0.0,
                'min_glucose': 0.0,
                'max_glucose': 0.0,
                'glucose_range': 0.0,
                'mage': 0.0
            }

        mean_glucose_val = float(np.mean(glucose_values))
        metrics = {
            'mean_glucose': mean_glucose_val,
            'median_glucose': float(np.median(glucose_values)),
            'std_glucose': float(np.std(glucose_values)),
            'cv_glucose': float(np.std(glucose_values) / mean_glucose_val * 100) if mean_glucose_val > 0 else 0.0,
            'min_glucose': float(np.min(glucose_values)),
            'max_glucose': float(np.max(glucose_values)),
            'glucose_range': float(np.max(glucose_values) - np.min(glucose_values))
        }
        
        # Calcular MAGE (Mean Amplitude of Glycemic Excursions) de manera simplificada
        if len(glucose_values) > 1:
            diff = np.abs(np.diff(glucose_values))
            significant_excursions = diff > (np.std(glucose_values) * 1.0) # Umbral de 1 SD
            if np.any(significant_excursions):
                metrics['mage'] = float(np.mean(diff[significant_excursions]))
            else:
                metrics['mage'] = 0.0
        else:
            metrics['mage'] = 0.0
            
        return metrics
    
    @staticmethod
    def calculate_risk_indices(glucose_values: np.ndarray) -> Dict[str, float]:
        """
        Calcula índices de riesgo LBGI y HBGI.
        Adaptado de Kovatchev et al., Diabetes Care 2006.
        
        Parámetros:
        -----------
        glucose_values : np.ndarray
            Valores de glucosa en mg/dL.
            
        Retorna:
        --------
        Dict[str, float]
            Diccionario con 'lbgi', 'hbgi', y 'bgri'.
        """
        if glucose_values.size == 0:
            return {'lbgi': 0.0, 'hbgi': 0.0, 'bgri': 0.0}

        # Transformación f(bg) = 1.509 * (ln(bg)^1.084 - 5.381)
        # rl(bg) = f(bg) si f(bg) < 0, sino 0
        # rh(bg) = f(bg) si f(bg) > 0, sino 0
        # LBGI = mean(rl(bg)^2), HBGI = mean(rh(bg)^2)
        
        # Evitar log(0) o log(<0) por si acaso, aunque glucosa no debería ser <=0
        safe_glucose_values = np.maximum(glucose_values, 1.0)
        
        fbg = 1.509 * (np.log(safe_glucose_values)**1.084 - 5.381)
        
        rlbg = np.where(fbg < 0, fbg, 0)
        rhbg = np.where(fbg > 0, fbg, 0)
        
        lbgi = np.mean(rlbg**2)
        hbgi = np.mean(rhbg**2)
        
        return {
            'lbgi': float(lbgi),
            'hbgi': float(hbgi),
            'bgri': float(lbgi + hbgi)
        }


def evaluate_clinical_metrics(simulator: GlucoseSimulator, predictions: np.ndarray, initial_glucose: np.ndarray, 
                            carb_intake: np.ndarray, duration_hours: int = CONST_DURATION_HOURS) -> Dict[str, float]:
    """
    Evalúa métricas clínicas utilizando un simulador de glucosa.
    
    Parámetros:
    -----------
    simulator : GlucoseSimulator
        Objeto simulador de glucosa.
    predictions : np.ndarray
        Dosis de insulina predichas.
    initial_glucose : np.ndarray
        Valores iniciales de glucosa.
    carb_intake : np.ndarray
        Valores de ingesta de carbohidratos.
    duration_hours : int, opcional
        Duración de la simulación en horas (default: ver constantes).
            
    Retorna:
    --------
    Dict[str, float]
        Diccionario con métricas clínicas promediadas.
    """
    # from validation.clinical_metrics import ClinicalMetricsEvaluator # Importación ya está al inicio del archivo

    all_simulated_metrics: List[Dict[str, float]] = []
    
    # Asegurar que todos los arrays tengan la misma longitud
    num_samples = len(predictions)
    if not (len(initial_glucose) == num_samples and len(carb_intake) == num_samples):
        print_error(f"Discrepancia en longitudes de entrada para evaluate_clinical_metrics: "
                    f"predictions ({num_samples}), initial_glucose ({len(initial_glucose)}), carb_intake ({len(carb_intake)})")
        # Retornar métricas vacías o con ceros en caso de error de datos
        sample_metrics = ClinicalMetricsEvaluator.evaluate_clinical_metrics(np.array([]))
        return {k: 0.0 for k in sample_metrics.keys()}


    # Simular para cada predicción
    for i in range(num_samples):
        # Simular trayectoria de glucosa
        # El simulador predict_glucose_trajectory espera listas para dosis y carbs, y timestamps
        # Aquí asumimos que cada predicción es para un solo evento/paso de tiempo.
        # La simulación debe cubrir un horizonte de tiempo después de la dosis.
        
        # Para una simulación de un solo evento de dosificación:
        # insulin_doses = [predictions[i]]
        # carb_intakes_sim = [carb_intake[i]]
        # timestamps_sim = [0.0] # El evento ocurre al tiempo 0 relativo a la simulación
        
        # glucose_trajectory = simulator.predict_glucose_trajectory(
        #     initial_glucose=initial_glucose[i],
        #     insulin_doses=insulin_doses,
        #     carb_intakes=carb_intakes_sim,
        #     timestamps=timestamps_sim,
        #     prediction_horizon=duration_hours 
        # )
        
        # Usando el método step del simulador para un solo paso (o una secuencia si se define así)
        # Para evaluar el impacto de una dosis, necesitamos simular hacia adelante.
        # El método step actual es para un solo paso de 5min. Para un horizonte más largo,
        # necesitaríamos un bucle o un método de simulación de horizonte como predict_glucose_trajectory.
        
        # Reutilizando la lógica de predict_glucose_trajectory que parece más adecuada aquí:
        sim_insulin_doses = [predictions[i]]
        sim_carb_intakes = [carb_intake[i]] # Asumiendo que carb_intake[i] es el relevante para esta dosis
        sim_timestamps = [0.0] # La dosis y los carbohidratos ocurren al inicio de esta simulación particular

        glucose_trajectory = simulator.predict_glucose_trajectory(
            initial_glucose=initial_glucose[i],
            insulin_doses=sim_insulin_doses,
            carb_intakes=sim_carb_intakes,
            timestamps=sim_timestamps,
            prediction_horizon=duration_hours
        )
        
        if glucose_trajectory.size > 0:
            current_metrics = ClinicalMetricsEvaluator.evaluate_clinical_metrics(glucose_trajectory)
            all_simulated_metrics.append(current_metrics)
        else:
            # Añadir métricas vacías o con ceros si la trayectoria está vacía
            empty_metrics = ClinicalMetricsEvaluator.evaluate_clinical_metrics(np.array([]))
            all_simulated_metrics.append(empty_metrics)

    # Promediar las métricas de todas las simulaciones, ignorando NaNs si los hubiera
    if not all_simulated_metrics:
        print_warning("No se pudieron calcular métricas clínicas para ninguna muestra.")
        # Retornar un diccionario de métricas con NaNs o ceros según se prefiera
        # Usando la primera muestra como plantilla para las claves, si existe, o un conjunto predefinido
        # Esto requiere que ClinicalMetricsEvaluator.evaluate_clinical_metrics devuelva un dict plano
        dummy_metrics = ClinicalMetricsEvaluator.evaluate_clinical_metrics(np.array([])) # Obtiene claves con 0.0
        return {key: np.nan for key in dummy_metrics.keys()}


    # Convertir lista de diccionarios a un diccionario de listas
    aggregated_metrics: Dict[str, List[float]] = {k: [] for k in all_simulated_metrics[0].keys()}
    for m_dict in all_simulated_metrics:
        for key, value in m_dict.items():
            if isinstance(value, (int, float)): # Solo agregar valores numéricos
                 aggregated_metrics[key].append(value)
            # Ignorar sub-diccionarios como 'variability' si no se aplanan antes

    # Calcular el promedio para cada métrica, ignorando NaNs
    averaged_results: Dict[str, float] = {}
    for key, values_list in aggregated_metrics.items():
        if values_list: # Asegurarse de que la lista no esté vacía
            averaged_results[key] = float(np.nanmean(values_list))
        else:
            averaged_results[key] = np.nan # O 0.0 si se prefiere

    return averaged_results

def create_ensemble_prediction(predictions_dict: Dict[str, np.ndarray], 
                              weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Combina predicciones de múltiples modelos usando promedio ponderado.
    
    Parámetros:
    -----------
    predictions_dict : Dict[str, np.ndarray]
        Diccionario con predicciones de cada modelo
    weights : Optional[np.ndarray], opcional
        Pesos para cada modelo. Si es None, usa promedio simple (default: None)
        
    Retorna:
    --------
    np.ndarray
        Predicciones combinadas del ensemble
    """
    # Normalizar todos los arrays de predicciones a 1D
    normalized_preds = {}
    if not predictions_dict:
        print_warning("El diccionario de predicciones está vacío en create_ensemble_prediction.")
        return np.array([])

    for model_name, preds in predictions_dict.items():
        # Convertir a numpy si es necesario y asegurar que sea al menos 1D
        current_preds_np = np.atleast_1d(np.array(preds, dtype=float)) # Asegurar tipo float también
            
        # Asegurar que el array sea 1D después de la conversión inicial
        if current_preds_np.ndim > 1:
            current_preds_np = current_preds_np.reshape(-1)
        
        if current_preds_np.size == 0:
            print_warning(f"Predicciones vacías para el modelo {model_name} en create_ensemble_prediction. Se omitirá.")
            continue # Omitir este modelo si sus predicciones están vacías

        normalized_preds[model_name] = current_preds_np
    
    if not normalized_preds:
        print_warning("No hay predicciones normalizadas válidas para crear el ensemble.")
        return np.array([])

    # Verificar si todos los arrays tienen la misma longitud después de normalizar
    lengths = [len(p) for p in normalized_preds.values()] # Ahora 'p' es garantizado 1D o más
    if not lengths: # Si todos los modelos fueron omitidos
         print_warning("Todas las predicciones de los modelos estaban vacías. No se puede crear ensemble.")
         return np.array([])

    if len(set(lengths)) > 1:
        min_length = min(lengths)
        print_warning(f"Los modelos tienen longitudes de predicción diferentes: {lengths}. Se truncará a la longitud mínima: {min_length}")
        for model_name, p_arr in normalized_preds.items():
            normalized_preds[model_name] = p_arr[:min_length]
    
    # Apilar predicciones y aplicar pesos
    # Filtrar nuevamente en caso de que el truncamiento haya resultado en arrays vacíos para algunos (aunque improbable si min_length > 0)
    valid_preds_list = [p for p in normalized_preds.values() if p.size > 0]
    if not valid_preds_list:
        print_warning("No hay predicciones válidas después del truncamiento para crear el ensemble.")
        return np.array([])

    all_preds = np.stack(valid_preds_list)
    num_valid_models = all_preds.shape[0]

    if weights is None:
        final_weights = np.ones(num_valid_models) / num_valid_models
    else:
        if len(weights) != num_valid_models:
            print_warning(f"La longitud de los pesos ({len(weights)}) no coincide con el número de modelos válidos ({num_valid_models}). Se usarán pesos uniformes.")
            final_weights = np.ones(num_valid_models) / num_valid_models
        else:
            final_weights = weights
            
    return np.average(all_preds, axis=0, weights=final_weights)

def optimize_ensemble_weights_clinical(predictions: Dict[str, np.ndarray], 
                                     initial_glucose: np.ndarray,
                                     carb_intake: np.ndarray,
                                     simulator,
                                     y_true: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimiza pesos del ensemble basado en métricas clínicas.
    
    Parámetros:
    -----------
    predictions : Dict[str, np.ndarray]
        Diccionario con predicciones por modelo
    initial_glucose : np.ndarray
        Valores iniciales de glucosa
    carb_intake : np.ndarray
        Valores de ingesta de carbohidratos
    simulator : GlucoseSimulator
        Simulador para métricas clínicas
    y_true : np.ndarray, opcional
        Dosis de insulina reales (para evaluación)
        
    Retorna:
    --------
    Tuple[np.ndarray, np.ndarray]
        (pesos optimizados, predicción del ensemble)
    """
    def objective(weights):
        # Normalizar pesos para que sumen 1
        weights = weights / np.sum(weights)
        
        # Crear predicción del ensemble
        ensemble_pred = create_ensemble_prediction(predictions, weights)
        
        # Evaluar métricas clínicas
        metrics = evaluate_clinical_metrics(
            simulator=simulator,
            predictions=ensemble_pred,
            initial_glucose=initial_glucose,
            carb_intake=carb_intake
        )
        
        # Retornar negativo del tiempo en rango (para maximizarlo)
        return -metrics['time_in_range']
    
    # Pesos iniciales (ponderación equitativa)
    n_models = len(predictions)
    initial_weights = np.ones(n_models) / n_models
    
    # Límites y restricciones
    bounds = [(0, 1) for _ in range(n_models)]
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    
    # Optimizar pesos
    try:
        result = minimize(
            objective,
            initial_weights,
            bounds=bounds,
            constraints=constraints,
            method='SLSQP'
        )
        optimized_weights = result.x / np.sum(result.x)
    except Exception as e:
        print(f"Error en optimización: {e}")
        optimized_weights = initial_weights
    
    # Crear predicción final del ensemble
    ensemble_pred = create_ensemble_prediction(predictions, optimized_weights)
    
    return optimized_weights, ensemble_pred

def _apply_cgm_padding(cgm_diff: np.ndarray, x_cgm: np.ndarray) -> Optional[np.ndarray]:
    """Aplica padding a las diferencias CGM según dimensionalidad."""
    if cgm_diff.ndim == 3:
        return np.pad(cgm_diff, ((0, 0), (1, 0), (0, 0)), mode='constant', constant_values=0)
    elif cgm_diff.ndim == 2:
        padded = np.pad(cgm_diff, ((0, 0), (1, 0)), mode='constant', constant_values=0)
        if x_cgm.ndim == 3 and x_cgm.shape[2] == 1 and padded.ndim == 2:
            padded = padded[:, :, np.newaxis]
        return padded
    else:
        print_warning(f"Forma inesperada de cgm_diff: {cgm_diff.shape}. No se aplicará padding de diferencias.")
        return None

def _validate_and_fix_shape(cgm_diff_padded: np.ndarray, x_cgm: np.ndarray) -> Optional[np.ndarray]:
    """Valida y corrige la forma del array de diferencias CGM."""
    if cgm_diff_padded.shape == x_cgm.shape:
        return cgm_diff_padded
    
    print_warning(f"Discrepancia de formas después del padding: Original {x_cgm.shape}, Diff Padded {cgm_diff_padded.shape}. Se intentará ajustar o se devolverá original.")
    
    if (cgm_diff_padded.ndim == x_cgm.ndim and 
        cgm_diff_padded.shape[0] == x_cgm.shape[0] and 
        cgm_diff_padded.shape[1] == x_cgm.shape[1]):
        try:
            return cgm_diff_padded.reshape(x_cgm.shape)
        except ValueError:
            print_error("No se pudo ajustar la forma de cgm_diff_padded. Devolviendo características CGM originales.")
            return None
    else:
        print_error("No se pudo ajustar la forma de cgm_diff_padded debido a dimensiones incompatibles. Devolviendo características CGM originales.")
        return None

# def enhance_features(x_cgm: np.ndarray, x_other: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Mejora las características de entrada añadiendo derivadas y estadísticas.
    
#     Parámetros:
#     -----------
#     x_cgm : np.ndarray
#         Datos CGM de forma (muestras, pasos_tiempo, características)
#     x_other : np.ndarray
#         Otras características de forma (muestras, características)
        
#     Retorna:
#     --------
#     Tuple[np.ndarray, np.ndarray]
#         Tupla con (cgm_features_mejoradas, other_features_mejoradas)
#     """
#     if x_cgm.shape[1] <= 1:
#         print_warning("No hay suficientes pasos de tiempo en x_cgm para calcular diferencias. Retornando características originales.")
#         return x_cgm, x_other

#     # Calcular diferencia entre puntos de tiempo consecutivos (derivada)
#     cgm_diff = np.diff(x_cgm, axis=1)
    
#     # Aplicar padding según la dimensionalidad real
#     cgm_diff_padded = _apply_cgm_padding(cgm_diff, x_cgm)
#     if cgm_diff_padded is None:
#         return x_cgm, x_other

#     # Validar y corregir forma si es necesario
#     cgm_diff_padded = _validate_and_fix_shape(cgm_diff_padded, x_cgm)
#     if cgm_diff_padded is None:
#         return x_cgm, x_other

#     # Concatenar características originales con la derivada
#     try:
#         cgm_enhanced = np.concatenate((x_cgm, cgm_diff_padded), axis=2)
#     except ValueError as e:
#         print_error(f"Error al concatenar características CGM: {e}. Formas: x_cgm={x_cgm.shape}, cgm_diff_padded={cgm_diff_padded.shape}. Devolviendo originales.")
#         return x_cgm, x_other

#     return cgm_enhanced, x_other

def enhance_features(df: pl.DataFrame) -> pl.DataFrame: # Modificado
    """
    Añade características mejoradas al DataFrame.
    Ejemplo: diferencias, rolling means, etc., directamente sobre columnas del DataFrame.

    Parámetros:
    -----------
    df : pl.DataFrame
        DataFrame de entrada.

    Retorna:
    --------
    pl.DataFrame
        DataFrame con características mejoradas.
    """
    if df.is_empty():
        print_warning("DataFrame vacío en enhance_features, retornando original.")
        return df
    
    df_enhanced = df.clone()
    
    # Ejemplo: Calcular diferencia de glucosa (si 'glucose_last' existe)
    # Esto es solo un ejemplo, la lógica real de 'enhance_features' debe ser portada aquí.
    # La lógica original de enhance_features operaba sobre x_cgm y x_other (numpy)
    # Aquí, operaríamos sobre columnas del DataFrame.
    
    # Por ejemplo, si tienes una columna 'glucose_last' y quieres crear 'glucose_diff_5min'
    # if 'glucose_last' in df_enhanced.columns and 'cgm_22' in df_enhanced.columns: # cgm_22 sería glucose_last - 5min
    #    df_enhanced = df_enhanced.with_columns(
    #        (pl.col('glucose_last') - pl.col('cgm_22')).alias('glucose_trend_5min')
    #    )
    
    # La lógica original de `enhance_features` era:
    # cgm_diff = np.diff(x_cgm, axis=1) # Diferencias a lo largo del tiempo
    # ... padding ...
    # x_cgm_enhanced = np.concatenate([x_cgm, cgm_diff_padded], axis=-1)
    # Esto implicaría que las columnas CGM (cgm_0 a cgm_23) se usan para calcular diferencias
    # y estas diferencias se añaden como nuevas columnas.
    
    # Ejemplo conceptual para diferencias de CGM:
    # cgm_cols = [f'cgm_{i}' for i in range(24)] # Asumiendo que estas columnas existen
    # for i in range(1, len(cgm_cols)):
    #    if cgm_cols[i] in df_enhanced.columns and cgm_cols[i-1] in df_enhanced.columns:
    #        df_enhanced = df_enhanced.with_columns(
    #            (pl.col(cgm_cols[i]) - pl.col(cgm_cols[i-1])).alias(f'cgm_diff_{i}')
    #        )
    
    # print_info("enhance_features (DataFrame version) necesita implementar la lógica de mejora de características.")
    return df_enhanced

def get_model_type(model_name: str) -> str:
    """
    Determina el tipo de modelo basado en su nombre.
    
    Parámetros:
    -----------
    model_name : str
        Nombre del modelo
        
    Retorna:
    --------
    str
        Tipo de modelo: "dl", "rl" o "drl"
    """
    if any(x in model_name for x in ["monte_carlo", "policy_iteration", "q_learning", "sarsa", "value_iteration", "reinforce"]):
        return "rl"
    elif any(x in model_name for x in ["a2c", "a3c", "ddpg", "dqn", "ppo", "sac", "trpo"]):
        return "drl"
    else:
        return "dl"


def process_training_results(model_results: List[Dict], 
                            y_test: np.ndarray) -> Tuple[Dict[str, Dict], Dict[str, np.ndarray], Dict[str, Dict]]:
    """
    Procesa resultados de múltiples modelos entrenados.
    
    Parámetros:
    -----------
    model_results : List[Dict]
        Lista de resultados de modelos con keys 'name', 'history', 'predictions'
    y_test : np.ndarray
        Valores objetivo de prueba
        
    Retorna:
    --------
    Tuple[Dict[str, Dict], Dict[str, np.ndarray], Dict[str, Dict]]
        (historiales, predicciones, métricas) diccionarios
    """
    from config.params import FRAMEWORK
    
    # Procesar resultados secuencialmente cuando se usa JAX para evitar deadlocks
    if FRAMEWORK == "jax":
        print("\nCalculando métricas secuencialmente (compatible con JAX)...")
        metric_results = [
            calculate_metrics(
                y_test, 
                np.array(result['predictions'])
            ) for result in model_results
        ]
    else:
        # Para TensorFlow u otros frameworks, mantener el paralelismo
        print("\nCalculando métricas en paralelo...")
        with Parallel(n_jobs=-1, verbose=1) as parallel:
            metric_results = parallel(
                delayed(calculate_metrics)(
                    y_test, 
                    np.array(result['predictions'])
                ) for result in model_results
            )
    
    # Almacenar resultados
    histories = {}
    predictions = {}
    metrics = {}
    
    for result, metric in zip(model_results, metric_results):
        name = result['name']
        histories[name] = result['history']
        predictions[name] = np.array(result['predictions'])
        metrics[name] = metric
    
    return histories, predictions, metrics

def evaluate_policy_offline_ope(
    policy_model_wrapper: DRLModelWrapperPyTorch,
    ope_evaluator_creator: Callable, # e.g., create_fqe_evaluator or create_dre_evaluator
    ope_method_name: str, # "FQE" or "DRE"
    eval_data: AllData,
    cgm_input_dim: Tuple[int, ...],
    other_input_dim: Tuple[int, ...],
    training_config: Dict[str, Any]
) -> Dict[str, float]:
    """
    Evalúa una política DRL usando un método de OPE especificado (FQE o DRE).
    """
    print_header(f"Evaluación Offline ({ope_method_name}) para: {policy_model_wrapper.algorithm} (modelo: {getattr(policy_model_wrapper,'model_name','N/A')})")

    # Determine context_dim from eval_data if possible, or use a default/passed one
    context_dim_eval = 0
    if eval_data.get('train') and eval_data['train'].get('context') and CONTEXT_FEATURE_ORDER:
        context_dim_eval = len(CONTEXT_FEATURE_ORDER)

    ope_evaluator = ope_evaluator_creator(
        cgm_input_dim=cgm_input_dim,
        other_input_dim=other_input_dim
        # context_dim might be needed by OPE networks if they model context explicitly
        # hidden_dim, gamma, lr can be from training_config or OPE defaults
    )

    eval_train_data = eval_data.get('train')
    if not eval_train_data or eval_train_data.get('x_cgm') is None or eval_train_data.get('y') is None:
        print_error(f"Datos de entrenamiento para {ope_method_name} incompletos.")
        return {}

    print_info(f"Entrenando evaluador {ope_method_name}...")
    # OPE evaluators (FQE/DRE) fit their internal models (Q-nets, behavior policy nets)
    # y_actions are the actions from the behavior policy in the dataset
    ope_fit_history = ope_evaluator.fit(
        x_cgm=eval_train_data['x_cgm'],
        x_other=eval_train_data.get('x_other'),
        y_actions=eval_train_data['y'],
        # validation_data can be passed if OPE fit supports it
        epochs=training_config.get('ope_epochs', 10),
        batch_size=training_config.get('ope_batch_size', training_config.get('batch_size', CONST_DEFAULT_BATCH_SIZE))
    )
    print_info(f"Entrenamiento de evaluador {ope_method_name} completado. Historial: {ope_fit_history}")

    eval_test_data = eval_data.get('test')
    if not eval_test_data or eval_test_data.get('x_cgm') is None or eval_test_data.get('y') is None:
        print_error(f"Datos de prueba para evaluación {ope_method_name} incompletos.")
        return {}

    print_info(f"Evaluando política con {ope_method_name}...")
    # The policy_model_wrapper is the 'policy' argument for ope_evaluator.evaluate_policy
    # It will use policy_model_wrapper.predict() or select_action() internally.
    ope_results = ope_evaluator.evaluate_policy(
        policy=policy_model_wrapper,
        x_cgm_test=eval_test_data['x_cgm'],
        x_other_test=eval_test_data.get('x_other'),
        y_actions_test=eval_test_data['y'], # Behavior actions for comparison/context
        context_test_data=eval_test_data.get('context')
        # simulator can be added if OPE's evaluate_policy uses it for additional metrics
    )
    print_success(f"Resultados de evaluación {ope_method_name}: {ope_results}")

    return ope_results