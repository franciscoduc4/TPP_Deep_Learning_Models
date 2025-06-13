import numpy as np
import pandas as pd
import polars as pl
from typing import Dict, List, Any, Callable, Optional, Tuple, Union
import matplotlib.pyplot as plt
import os

from config.feature_selection import RECOMMENDED_FEATURES
from custom.printer import print_error, print_warning
from validation.ClinicalMetrics import ClinicalMetricsEvaluator
from validation.simulator import GlucoseSimulator
from validation.metrics import evaluate_glucose_control
from constants.constants import (
    CONTEXT_FEATURE_ORDER, SEVERE_HYPOGLYCEMIA_THRESHOLD, HYPOGLYCEMIA_THRESHOLD, 
    HYPERGLYCEMIA_THRESHOLD, SEVERE_HYPERGLYCEMIA_THRESHOLD, IDEAL_LOWER_BOUND, IDEAL_UPPER_BOUND
)

def validate_dosing_model_pl(
    model: Any,
    test_data: pl.DataFrame,
    patient_params: Dict[str, Dict[str, float]],
    # Add context_data_full if you decide to pass the dictionary directly
    # context_data_full: Optional[Dict[str, np.ndarray]] = None, 
    output_dir: str = "validation_results",
    visualize: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Valida un modelo de dosificación de insulina basado en el impacto en los niveles de glucosa.
    
    Parámetros:
    -----------
    model : Any
        Modelo entrenado que implementa predict_with_context
    test_data : pl.DataFrame
        Datos de prueba con columnas para CGM, carbohidratos, etc. (polars DataFrame)
    patient_params : Dict[str, Dict[str, float]]
        Parámetros específicos por paciente (sensibilidad, ratio, etc.)
    output_dir : str, opcional
        Directorio para guardar resultados (default: "validation_results")
    visualize : bool, opcional
        Si generar visualizaciones (default: True)
        
    Retorna:
    --------
    Dict[str, Dict[str, float]]
        Métricas de control glucémico por paciente
    """
    results = {}
    # Asegurar que 'subject_id' existe
    if "subject_id" not in test_data.columns:
        print_error("La columna 'subject_id' no se encuentra en test_data para validación (Polars).")
        return {"error": {"error_metric": -1.0}}
        
    patient_ids = test_data["subject_id"].unique().to_list()
    
    # Definir las columnas de características CGM y otras que se esperan del preprocesamiento
    # Esto debe alinearse con lo que produce `prepare_features_for_drl`
    # Ejemplo: cgm_window_col = 'cgm_window_array' # o como se llame la columna con la secuencia CGM
    # other_feature_cols = [...] # lista de nombres de otras columnas de características

    for patient_id in patient_ids:
        print(f"Validando paciente {patient_id}...")
        
        patient_data_pl = test_data.filter(pl.col("subject_id") == patient_id)
        
        if patient_data_pl.is_empty():
            print_warning(f"No hay datos para el paciente {patient_id} en validate_dosing_model_pl.")
            continue

        # Ordenar datos por timestamp si existe
        if "timestamp" in patient_data_pl.columns:
            patient_data_pl = patient_data_pl.sort(by="timestamp")
        else:
            print_warning("Columna 'timestamp' no encontrada para ordenar datos del paciente.")

        patient_specific_params = patient_params.get(str(patient_id), patient_params.get('default', {}))
        if not patient_specific_params:
             print_warning(f"No se encontraron parámetros para el paciente {patient_id}, usando defaults del simulador.")
             simulator = GlucoseSimulator()
        else:
            simulator = GlucoseSimulator(**patient_specific_params)
        
        patient_metrics_list = []
        
        # Iterar sobre cada fila (cada punto de decisión)
        for i in range(len(patient_data_pl)):
            current_sample_pl = patient_data_pl.row(i, named=True)

            # Extraer x_cgm y x_other
            # Esto depende de cómo estén almacenadas las secuencias CGM y otras features en el DataFrame
            # Asumimos que hay una columna 'cgm_features_array' y 'other_features_array' o nombres individuales
            
            # Placeholder: Necesitas una lógica robusta para extraer x_cgm y x_other de `current_sample_pl`
            # Por ejemplo, si 'cgm_0' a 'cgm_23' son columnas:
            cgm_cols = [f'cgm_{k}' for k in range(24)] # Ajustar según el número real de características CGM
            x_cgm_sample_list = []
            for cgm_col_name in cgm_cols:
                if cgm_col_name in current_sample_pl:
                    x_cgm_sample_list.append(current_sample_pl[cgm_col_name])
                else:
                    # print_warning(f"Columna CGM '{cgm_col_name}' no encontrada para paciente {patient_id}, muestra {i}. Usando 0.")
                    x_cgm_sample_list.append(0.0) # O manejar de otra forma

            # Asumiendo que las características CGM son una secuencia temporal, y cada cgm_k es un punto en el tiempo.
            # Si cada cgm_k es una característica diferente en el mismo punto de tiempo, la lógica cambia.
            # Para DDPG, se espera (timesteps, cgm_features_per_timestep)
            # Si 'cgm_0'...'cgm_23' son valores en diferentes timesteps para UNA característica CGM:
            x_cgm_sample = np.array(x_cgm_sample_list, dtype=np.float32).reshape(-1, 1) # (timesteps, 1)
            if x_cgm_sample.shape[0] == 0 : # Si no hay columnas cgm
                 x_cgm_sample = np.zeros((12,1), dtype=np.float32) # Fallback a un shape esperado
                 print_warning(f"No se encontraron columnas CGM para paciente {patient_id}, muestra {i}. Usando placeholder.")


            # Para x_other, necesitas identificar las columnas relevantes
            # Ejemplo:
            other_feature_names_from_selection = [
                f for f in RECOMMENDED_FEATURES 
                if not (f.startswith('cgm_') and f[4:].isdigit()) and f not in CONTEXT_FEATURE_ORDER and f in current_sample_pl
            ] # Excluir las que se pasan explícitamente y las CGM
            
            x_other_sample_list = [current_sample_pl.get(f, 0.0) for f in other_feature_names_from_selection]
            x_other_sample = np.array(x_other_sample_list, dtype=np.float32)
            if x_other_sample.size == 0: # Si no hay otras features
                # El modelo DDPG espera other_input_dim[0] > 0 si se usa.
                # Si other_input_dim es (0,), entonces un array vacío está bien.
                # Ajustar según la configuración del modelo DDPG.
                # Por ahora, si es (0,), un array vacío está bien. Si no, podría dar error.
                # print_warning(f"No se encontraron 'otras características' para paciente {patient_id}, muestra {i}. Usando array vacío.")
                pass


            # Extraer contexto explícito
            # CONTEXT_FEATURE_ORDER = ['current_glucose', 'carb_intake', 'iob', 'sleep_quality', 'work_intensity', 'exercise_intensity']
            def get_context_val(key: str, default: float) -> float:
                col_name = key
                # Mapeo de nombres si es necesario, ej: 'meal_carbs' para 'carb_intake'
                if key == 'carb_intake' and 'meal_carbs' in current_sample_pl: col_name = 'meal_carbs'
                if key == 'current_glucose' and 'glucose_last' in current_sample_pl: col_name = 'glucose_last'
                
                val = current_sample_pl.get(col_name)
                if val is None:
                    # print_warning(f"Contexto '{key}' (col: {col_name}) no encontrado para paciente {patient_id}, muestra {i}. Usando default {default}.")
                    return default
                return float(val)

            current_glucose_val = get_context_val('current_glucose', 150.0)
            carb_intake_val = get_context_val('carb_intake', 0.0)
            iob_val = get_context_val('iob', 0.0)
            sleep_quality_val = get_context_val('sleep_quality', 0.0)
            work_intensity_val = get_context_val('work_intensity', 0.0)
            exercise_intensity_val = get_context_val('exercise_intensity', 0.0)

            try:
                predicted_dose = model.predict_with_context(
                    x_cgm=x_cgm_sample, 
                    x_other=x_other_sample,
                    current_glucose=current_glucose_val,
                    carb_intake=carb_intake_val,
                    iob=iob_val,
                    sleep_quality=sleep_quality_val,
                    work_intensity=work_intensity_val,
                    exercise_intensity=exercise_intensity_val
                    # target_glucose no se usa en DDPG predict_with_context actualmente
                )
            except Exception as e:
                print_error(f"Error en model.predict_with_context para paciente {patient_id}, muestra {i}: {e}")
                predicted_dose = 0.0 # Fallback

            # Simular trayectoria de glucosa
            # El simulador step espera una sola dosis y un solo carb_intake para el paso actual
            next_glucose, reward, done, _ = simulator.step(
                action_insulin=predicted_dose,
                current_glucose=current_glucose_val, # Glucosa al inicio del paso
                carb_intake=carb_intake_val # Carbs que afectan este paso
            )
            
            # Aquí, la métrica se basa en el resultado de un solo paso.
            # Para métricas como TIR sobre una trayectoria, necesitarías simular más tiempo.
            # Por ahora, usaremos el 'next_glucose' para una evaluación simple.
            # Si quieres TIR, etc., necesitarías llamar a simulator.predict_glucose_trajectory
            # y luego a ClinicalMetricsEvaluator.
            
            # Ejemplo de métrica simple basada en el siguiente estado de glucosa:
            metrics_step = {
                'sim_next_glucose': next_glucose,
                'sim_reward': reward,
                'sim_predicted_dose': predicted_dose
            }
            # Para métricas clínicas completas, necesitarías una trayectoria:
            glucose_trajectory = simulator.predict_glucose_trajectory(
                initial_glucose=current_glucose_val,
                insulin_doses=[predicted_dose], # Dosis para el periodo
                carb_intakes=[carb_intake_val], # Carbs para el periodo
                timestamps=[0], # Tiempo relativo de la dosis/carbs
                prediction_horizon=6 # Simular por 6 horas
            )
            clinical_eval_metrics = ClinicalMetricsEvaluator.evaluate_clinical_metrics(glucose_trajectory)
            patient_metrics_list.append(clinical_eval_metrics)

        if patient_metrics_list:
            # Promediar las métricas clínicas sobre todos los puntos de decisión del paciente
            avg_patient_metrics: Dict[str, float] = {}
            for key in patient_metrics_list[0].keys():
                if isinstance(patient_metrics_list[0][key], dict): # Métricas de variabilidad
                    avg_patient_metrics[key] = {} # type: ignore
                    for sub_key in patient_metrics_list[0][key].keys():
                        avg_patient_metrics[key][sub_key] = np.mean([m[key][sub_key] for m in patient_metrics_list]) # type: ignore
                else:
                    avg_patient_metrics[key] = np.mean([m[key] for m in patient_metrics_list])
            results[str(patient_id)] = avg_patient_metrics
        else:
            print_warning(f"No se generaron métricas para el paciente {patient_id} en validate_dosing_model_pl.")

    return results

def validate_dosing_model_pd(
    model: Any,
    test_data: pd.DataFrame,
    patient_params: Dict[str, Dict[str, float]],
    # context_data_full: Optional[Dict[str, np.ndarray]] = None,
    output_dir: str = "validation_results",
    visualize: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Valida un modelo de dosificación de insulina basado en el impacto en los niveles de glucosa.
    
    Parámetros:
    -----------
    model : Any
        Modelo entrenado que implementa predict_with_context
    test_data : pd.DataFrame
        Datos de prueba con columnas para CGM, carbohidratos, etc. (pandas DataFrame)
    patient_params : Dict[str, Dict[str, float]]
        Parámetros específicos por paciente (sensibilidad, ratio, etc.)
    output_dir : str, opcional
        Directorio para guardar resultados (default: "validation_results")
    visualize : bool, opcional
        Si generar visualizaciones (default: True)
        
    Retorna:
    --------
    Dict[str, Dict[str, float]]
        Métricas de control glucémico por paciente
    """
    results = {}
    if "subject_id" not in test_data.columns:
        print_error("La columna 'subject_id' no se encuentra en test_data para validación (Pandas).")
        return {"error": {"error_metric": -1.0}}

    for patient_id, patient_data_pd in test_data.groupby("subject_id"):
        patient_id_str = str(patient_id)
        print(f"Validando paciente {patient_id_str}...")

        if patient_data_pd.empty:
            print_warning(f"No hay datos para el paciente {patient_id_str} en validate_dosing_model_pd.")
            continue
        
        if "timestamp" in patient_data_pd.columns:
            patient_data_pd = patient_data_pd.sort_values("timestamp")
        else:
            print_warning("Columna 'timestamp' no encontrada para ordenar datos del paciente.")

        patient_specific_params = patient_params.get(patient_id_str, patient_params.get('default', {}))
        if not patient_specific_params:
             print_warning(f"No se encontraron parámetros para el paciente {patient_id_str}, usando defaults del simulador.")
             simulator = GlucoseSimulator()
        else:
            simulator = GlucoseSimulator(**patient_specific_params)
        
        patient_metrics_list = []
        
        for i, current_sample_pd_series in patient_data_pd.iterrows():
            current_sample_pd = current_sample_pd_series.to_dict()

            cgm_cols = [f'cgm_{k}' for k in range(24)] 
            x_cgm_sample_list = [current_sample_pd.get(cgm_col_name, 0.0) for cgm_col_name in cgm_cols]
            x_cgm_sample = np.array(x_cgm_sample_list, dtype=np.float32).reshape(-1, 1)
            if x_cgm_sample.shape[0] == 0 :
                 x_cgm_sample = np.zeros((12,1), dtype=np.float32)
                 print_warning(f"No se encontraron columnas CGM para paciente {patient_id_str}, muestra {i}. Usando placeholder.")

            other_feature_names_from_selection = [
                f for f in RECOMMENDED_FEATURES 
                if not (f.startswith('cgm_') and f[4:].isdigit()) and f not in CONTEXT_FEATURE_ORDER and f in current_sample_pd
            ]
            x_other_sample_list = [current_sample_pd.get(f, 0.0) for f in other_feature_names_from_selection]
            x_other_sample = np.array(x_other_sample_list, dtype=np.float32)

            def get_context_val_pd(key: str, default: float) -> float:
                col_name = key
                if key == 'carb_intake' and 'meal_carbs' in current_sample_pd: col_name = 'meal_carbs'
                if key == 'current_glucose' and 'glucose_last' in current_sample_pd: col_name = 'glucose_last'
                val = current_sample_pd.get(col_name)
                if pd.isna(val) or val is None: # pd.isna maneja NaNs de pandas
                    # print_warning(f"Contexto '{key}' (col: {col_name}) no encontrado o NaN para paciente {patient_id_str}, muestra {i}. Usando default {default}.")
                    return default
                return float(val)

            current_glucose_val = get_context_val_pd('current_glucose', 150.0)
            carb_intake_val = get_context_val_pd('carb_intake', 0.0)
            iob_val = get_context_val_pd('iob', 0.0)
            sleep_quality_val = get_context_val_pd('sleep_quality', 0.0)
            work_intensity_val = get_context_val_pd('work_intensity', 0.0)
            exercise_intensity_val = get_context_val_pd('exercise_intensity', 0.0)

            try:
                predicted_dose = model.predict_with_context(
                    x_cgm=x_cgm_sample, 
                    x_other=x_other_sample,
                    current_glucose=current_glucose_val,
                    carb_intake=carb_intake_val,
                    iob=iob_val,
                    sleep_quality=sleep_quality_val,
                    work_intensity=work_intensity_val,
                    exercise_intensity=exercise_intensity_val
                )
            except Exception as e:
                print_error(f"Error en model.predict_with_context para paciente {patient_id_str}, muestra {i}: {e}")
                predicted_dose = 0.0

            glucose_trajectory = simulator.predict_glucose_trajectory(
                initial_glucose=current_glucose_val,
                insulin_doses=[predicted_dose],
                carb_intakes=[carb_intake_val],
                timestamps=[0],
                prediction_horizon=6 
            )
            clinical_eval_metrics = ClinicalMetricsEvaluator.evaluate_clinical_metrics(glucose_trajectory)
            patient_metrics_list.append(clinical_eval_metrics)

        if patient_metrics_list:
            avg_patient_metrics: Dict[str, float] = {}
            for key in patient_metrics_list[0].keys():
                if isinstance(patient_metrics_list[0][key], dict):
                    avg_patient_metrics[key] = {} # type: ignore
                    for sub_key in patient_metrics_list[0][key].keys():
                        avg_patient_metrics[key][sub_key] = np.mean([m[key][sub_key] for m in patient_metrics_list]) # type: ignore
                else:
                    avg_patient_metrics[key] = np.mean([m[key] for m in patient_metrics_list])
            results[patient_id_str] = avg_patient_metrics
        else:
            print_warning(f"No se generaron métricas para el paciente {patient_id_str} en validate_dosing_model_pd.")
            
    return results

def validate_dosing_model(
    model: Any,
    test_data: Union[pd.DataFrame, pl.DataFrame],
    patient_params: Dict[str, Dict[str, float]],
    output_dir: str = "validation_results",
    visualize: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Valida un modelo de dosificación de insulina basado en el impacto en los niveles de glucosa.
    
    Parámetros:
    -----------
    model : Any
        Modelo entrenado que implementa predict_with_context
    test_data : Union[pd.DataFrame, pl.DataFrame]
        Datos de prueba con columnas para CGM, carbohidratos, etc. (pandas o polars)
    patient_params : Dict[str, Dict[str, float]]
        Parámetros específicos por paciente (sensibilidad, ratio, etc.)
    output_dir : str, opcional
        Directorio para guardar resultados (default: "validation_results")
    visualize : bool, opcional
        Si generar visualizaciones (default: True)
        
    Retorna:
    --------
    Dict[str, Dict[str, float]]
        Métricas de control glucémico por paciente
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # Determinar si estamos trabajando con polars o pandas
    is_polars = isinstance(test_data, pl.DataFrame)

    results = validate_dosing_model_pl(model, test_data, patient_params, output_dir, visualize) if is_polars else validate_dosing_model_pd(model, test_data.to_pandas(), patient_params, output_dir, visualize)
   
    # Calcular y guardar métricas globales
    global_metrics = {}
    for metric in results[list(results.keys())[0]].keys():
        global_metrics[metric] = np.mean([r[metric] for r in results.values()])
    
    results["global"] = global_metrics
    
    # Guardar resultados en CSV
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.to_csv(os.path.join(output_dir, "validation_metrics.csv"))
    
    # Generar gráfico comparativo de tiempo en rango
    if visualize:
        plt.figure(figsize=(12, 6))
        
        # Extraer TIR para cada paciente
        patient_ids = [pid for pid in results.keys() if pid != "global"]
        tir_values = [results[pid]["time_in_range"] for pid in patient_ids]
        
        # Añadir TIR global
        patient_ids.append("Global")
        tir_values.append(results["global"]["time_in_range"])
        
        # Crear gráfico
        bars = plt.bar(patient_ids, tir_values, color='skyblue')
        bars[-1].set_color('navy')  # Destacar la barra global
        
        plt.axhline(y=70, color='r', linestyle='--', label='Objetivo Mínimo (70%)')
        plt.title('Tiempo en Rango por Paciente')
        plt.xlabel('ID de Paciente')
        plt.ylabel('Tiempo en Rango (%)')
        plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, "time_in_range_comparison.png"))
        plt.close()
    
    return results

def validate_model_with_simulator(model: Any, test_data: Dict[str, np.array], simulator: GlucoseSimulator) -> Dict[str, float]:
    """
    Valida la precisión de un modelo de dosificación de insulina simulando el impacto en los niveles de glucosa.
    
    Parámetros:
    -----------
    model : Any
        Modelo entrenado que implementa predict_with_context
    test_data : Dict[str, np.ndarray]
        Datos de prueba con claves 'x_cgm' y 'x_other' (numpy arrays)
    simulator : GlucoseSimulator
        Simulador de dinámica de glucosa para validar dosis de insulina
        
    Retorna:
    --------
    Dict[str, float]
        Métricas de control glucémico como tiempo en rango, eventos de hipoglucemia e hiperglucemia
    """
    time_in_range_percentages = []
    time_below_range = []
    time_severe_below = []
    time_above_range = []
    time_severe_above = []
    
    for i in range(len(test_data['x_cgm'])):
        # Obtener glucosa inicial y otros datos contextuales
        initial_glucose = test_data['x_cgm'][i][-1][-1]  # última lectura de glucosa
        carb_intake = test_data['x_other'][i][0]  # Suponiendo que la ingesta de carbohidratos es la primera característica
        
        # Obtener dosis predicha por el modelo
        predicted_dose = model.predict(test_data['x_cgm'][i:i+1], test_data['x_other'][i:i+1])[0]
        
        # Simular trayectoria de glucosa por 6 horas con intervalos de 5 min
        glucose_trajectory = simulator.predict_glucose_trajectory(
            initial_glucose=initial_glucose,
            insulin_doses=[predicted_dose],
            carb_intakes=[carb_intake],
            timestamps=[0],  # Dosis administrada en tiempo 0
            prediction_horizon=6  # Simular 6 horas hacia adelante
        )
        
        # Calcular tiempo en cada rango
        severe_below = glucose_trajectory < SEVERE_HYPOGLYCEMIA_THRESHOLD
        below_range = np.logical_and(
            glucose_trajectory >= SEVERE_HYPOGLYCEMIA_THRESHOLD,
            glucose_trajectory < HYPOGLYCEMIA_THRESHOLD
        )
        in_range = np.logical_and(
            glucose_trajectory >= HYPOGLYCEMIA_THRESHOLD, 
            glucose_trajectory <= HYPERGLYCEMIA_THRESHOLD
        )
        above_range = np.logical_and(
            glucose_trajectory > HYPERGLYCEMIA_THRESHOLD,
            glucose_trajectory <= SEVERE_HYPERGLYCEMIA_THRESHOLD
        )
        severe_above = glucose_trajectory > SEVERE_HYPERGLYCEMIA_THRESHOLD
        
        # Acumular métricas
        time_in_range_percentages.append(np.mean(in_range) * 100)  # Porcentaje
        time_below_range.append(np.mean(below_range) * 100)  # Porcentaje
        time_severe_below.append(np.mean(severe_below) * 100)  # Porcentaje
        time_above_range.append(np.mean(above_range) * 100)  # Porcentaje
        time_severe_above.append(np.mean(severe_above) * 100)  # Porcentaje
    
    return {
        'mean_time_in_range': np.mean(time_in_range_percentages),
        'time_below_range': np.mean(time_below_range),
        'time_severe_below': np.mean(time_severe_below),
        'time_above_range': np.mean(time_above_range),
        'time_severe_above': np.mean(time_severe_above),
        'hypo_events': np.sum([severe_below.any() for severe_below in time_severe_below]),
        'hyper_events': np.sum([severe_above.any() for severe_above in time_severe_above])
    }