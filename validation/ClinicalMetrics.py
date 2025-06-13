import numpy as np
from typing import Dict, Union

from constants.constants import (
    SEVERE_HYPOGLYCEMIA_THRESHOLD, HYPOGLYCEMIA_THRESHOLD,
    HYPERGLYCEMIA_THRESHOLD, SEVERE_HYPERGLYCEMIA_THRESHOLD,
    IDEAL_LOWER_BOUND, IDEAL_UPPER_BOUND
)

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
                'time_severe_below': 0.0, 'time_below_range': 0.0,
                'time_in_range_low': 0.0, 'time_in_ideal_range': 0.0, 'time_in_range_high': 0.0,
                'time_above_range': 0.0, 'time_severe_above': 0.0,
                'time_total_in_range': 0.0
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
            'time_total_in_range': float(time_total_in_range),
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
                'mean_glucose': 0.0, 'median_glucose': 0.0, 'std_glucose': 0.0,
                'cv_glucose': 0.0, 'min_glucose': 0.0, 'max_glucose': 0.0,
                'glucose_range': 0.0, 'mage': 0.0
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
            peaks = glucose_values[1:][np.diff(glucose_values) > 0]
            nadirs = glucose_values[1:][np.diff(glucose_values) < 0]
            excursions = []
            if len(peaks) > 0 and len(nadirs) > 0:
                # Simplificación: tomar diferencias entre picos y nadirs consecutivos si alternan
                # Una implementación más robusta requeriría identificar excursiones válidas (p.ej. > 1 SD)
                last_event_was_peak = False # Placeholder
                # Esta parte de MAGE es compleja y a menudo requiere librerías especializadas o una lógica más detallada.
                # Por ahora, una aproximación muy simple o dejarlo como 0.0 si no es crítico.
                # Placeholder para MAGE:
                diffs = np.abs(np.diff(glucose_values))
                significant_diffs = diffs[diffs > np.std(glucose_values)] # Ejemplo de filtro
                metrics['mage'] = float(np.mean(significant_diffs)) if len(significant_diffs) > 0 else 0.0
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
        
        lbgi = np.mean(rlbg**2) if rlbg.size > 0 else 0.0
        hbgi = np.mean(rhbg**2) if rhbg.size > 0 else 0.0
        
        return {
            'lbgi': float(lbgi),
            'hbgi': float(hbgi),
            'bgri': float(lbgi + hbgi)
        }

    @staticmethod
    def evaluate_clinical_metrics(glucose_values: np.ndarray) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Evalúa todas las métricas clínicas para una serie de valores de glucosa.
        
        Parámetros:
        -----------
        glucose_values : np.ndarray
            Valores de glucosa en mg/dL
            
        Retorna:
        --------
        Dict[str, Union[float, Dict[str, float]]]
            Diccionario con todas las métricas clínicas
        """
        if not isinstance(glucose_values, np.ndarray) or glucose_values.ndim == 0 or glucose_values.size == 0:
            # Retornar ceros o NaNs si no hay datos válidos
            time_metrics = ClinicalMetricsEvaluator.calculate_time_metrics(np.array([]))
            variability = ClinicalMetricsEvaluator.calculate_glucose_variability(np.array([]))
            risk_indices = ClinicalMetricsEvaluator.calculate_risk_indices(np.array([]))
        else:
            time_metrics = ClinicalMetricsEvaluator.calculate_time_metrics(glucose_values)
            variability = ClinicalMetricsEvaluator.calculate_glucose_variability(glucose_values)
            risk_indices = ClinicalMetricsEvaluator.calculate_risk_indices(glucose_values)

        # Combinar todos los diccionarios de métricas
        all_metrics = {**time_metrics, **variability, **risk_indices}
        return all_metrics
    
    @staticmethod
    def calculate_risk_index(glucose_values: np.ndarray) -> Dict[str, float]:
        """
        Calcula índices de riesgo basados en valores de glucosa.
        Esta es una implementación alternativa y podría ser diferente de Kovatchev.
        
        Parámetros:
        -----------
        glucose_values : np.ndarray
            Valores de glucosa en mg/dL
            
        Retorna:
        --------
        Dict[str, float]
            Índices de riesgo LBGI (bajo) y HBGI (alto)
        """
        if glucose_values.size == 0:
            return {'lbgi': 0.0, 'hbgi': 0.0, 'bgri': 0.0}
            
        # Convertir de mg/dL a mmol/L para fórmulas estándar si es necesario
        # Esta implementación parece usar mg/dL directamente para una fórmula simplificada
        # glucose_mmol = glucose_values / 18.0 
        
        # Función de transformación para índice de riesgo (ejemplo, puede variar)
        def risk_function(bg_mgdl: np.ndarray) -> np.ndarray:
            # Ejemplo de función de riesgo, no necesariamente estándar Kovatchev
            # Usaremos la de Kovatchev como en calculate_risk_indices para consistencia
            safe_bg = np.maximum(bg_mgdl, 1.0)
            fbg = 1.509 * (np.log(safe_bg)**1.084 - 5.381)
            return fbg

        transformed_risk_values = risk_function(glucose_values)
        
        # Separar riesgos altos y bajos
        # rl(bg) = f(bg) si f(bg) < 0, sino 0
        # rh(bg) = f(bg) si f(bg) > 0, sino 0
        rl = np.where(transformed_risk_values < 0, transformed_risk_values, 0)
        rh = np.where(transformed_risk_values > 0, transformed_risk_values, 0)
        
        # Calcular LBGI y HBGI (Kovatchev usa la media de los cuadrados de rl y rh)
        lbgi = np.mean(rl**2) if rl.size > 0 else 0.0
        hbgi = np.mean(rh**2) if rh.size > 0 else 0.0
        
        return {
            'lbgi': float(lbgi),  # Low Blood Glucose Index
            'hbgi': float(hbgi),  # High Blood Glucose Index
            'bgri': float(lbgi + hbgi)  # Blood Glucose Risk Index
        }