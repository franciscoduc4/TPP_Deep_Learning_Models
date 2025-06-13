from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import polars as pl
# Imports for JAX/Flax are not used in this specific file after modifications,
# but kept if other wrappers in the same original file might use them.
# import jax
# import jax.numpy as jnp
# import flax.linen as nn

from constants.constants import CONST_DEFAULT_EPOCHS, CONST_DEFAULT_BATCH_SIZE
from validation.ClinicalMetrics import ClinicalMetricsEvaluator
from validation.simulator import GlucoseSimulator
from custom.printer import print_error, print_warning

class ModelWrapper:
    """
    Clase base para encapsular modelos de aprendizaje profundo y por refuerzo.
    
    Proporciona una interfaz unificada para inicialización, entrenamiento y predicción.
    Todos los modelos deben heredar de esta clase.
    """
    
    def __init__(self, feature_config: Optional[Dict[str, List[str]]] = None) -> None:
        self.early_stopping: Optional[Dict[str, Any]] = None
        self.feature_config = feature_config

    def start(self, train_df: pl.DataFrame, rng_key: Optional[Any] = None) -> Any: 
        """
        Inicializa el modelo con los datos de entrenamiento.

        Parámetros:
        -----------
        train_df : pl.DataFrame
            DataFrame de entrenamiento.
        rng_key : Optional[Any], opcional
            Clave para generación aleatoria (default: None).
        """
        raise NotImplementedError("El método start debe ser implementado por las subclases.")

    def fit(self, train_df: pl.DataFrame,
             val_df: Optional[pl.DataFrame] = None,
             epochs: int = CONST_DEFAULT_EPOCHS, 
             batch_size: int = CONST_DEFAULT_BATCH_SIZE, 
             verbose: int = 1) -> Dict[str, List[float]]:
        """
        Entrena el modelo con los datos proporcionados.

        Parámetros:
        -----------
        train_df : pl.DataFrame
            DataFrame de entrenamiento.
        val_df : Optional[pl.DataFrame], opcional
            DataFrame de validación (default: None).
        epochs : int, opcional
            Número de épocas de entrenamiento.
        batch_size : int, opcional
            Tamaño de lote (relevante para cómo se procesan los datos del DataFrame).
        verbose : int, opcional
            Nivel de verbosidad.
        """
        raise NotImplementedError("El método fit debe ser implementado por las subclases.")

    def predict(self, df: pl.DataFrame) -> np.ndarray: # Modificado
        """
        Realiza predicciones con el modelo entrenado.

        Parámetros:
        -----------
        df : pl.DataFrame
            DataFrame con los datos para predicción.

        Retorna:
        --------
        np.ndarray
            Predicciones del modelo (ej: dosis de insulina).
        """
        raise NotImplementedError("El método predict debe ser implementado por las subclases.")

    def predict_with_context(self, current_sample_df: pl.DataFrame) -> float: # Modificado
        """
        Realiza una predicción de dosis basada en una muestra de datos actual (una fila de DataFrame).
        
        Parámetros:
        -----------
        current_sample_df : pl.DataFrame
            Un DataFrame que contiene una única fila con todas las características 
            necesarias (CGM histórico, 'glucose_last', 'meal_carbs', 'insulin_on_board', 
            'sleep_quality', 'work_intensity', 'exercise_intensity', y otras 'other_features').
            
        Retorna:
        --------
        float
            Dosis de insulina recomendada en unidades.
        """
        raise NotImplementedError("El método predict_with_context debe ser implementado por las subclases.")

    def evaluate(self, df: pl.DataFrame, target_column: str = 'bolus_log1p') -> Dict[str, float]: # Modificado
        """
        Evalúa el modelo con datos de prueba.

        Parámetros:
        -----------
        df : pl.DataFrame
            DataFrame de prueba.
        target_column : str, opcional
            Nombre de la columna objetivo en el DataFrame.
        """
        preds_np = self.predict(df)
        if target_column in df.columns and len(df[target_column]) == len(preds_np):
            # ... (cálculo de métricas como MSE, MAE)
            mse = np.mean((df[target_column].to_numpy() - preds_np)**2)
            return {'mse_loss': mse}
        print_warning("No se pudo calcular MSE en evaluate: target_column no encontrada o longitudes no coinciden.")
        return {'mse_loss': float('nan')}

    def evaluate_clinical(self, 
                          df: pl.DataFrame, # Modificado
                          simulator: GlucoseSimulator, 
                          simulation_hours: int = 24) -> Dict[str, float]:
        """
        Evalúa el modelo utilizando métricas clínicas basadas en simulación.

        Parámetros:
        -----------
        df : pl.DataFrame
            DataFrame (ej: test_df) que contiene las muestras para simulación.
            Debe incluir columnas para 'glucose_last' (initial_glucose), 'meal_carbs' (carb_intake),
            y todas las demás características necesarias para model.predict_with_context().
        simulator : GlucoseSimulator
            Instancia del simulador de glucosa.
        simulation_hours : int, opcional
            Duración de cada simulación en horas.

        Retorna:
        --------
        Dict[str, float]
            Diccionario con métricas clínicas promediadas.
        """
        if df.is_empty():
            print_warning("DataFrame vacío proporcionado a evaluate_clinical.")
            return {}

        all_simulated_metrics_list = self._process_all_samples(df, simulator, simulation_hours)
        
        if not all_simulated_metrics_list:
            print_warning("No se pudieron calcular métricas clínicas para ninguna muestra.")
            return {}

        return self._aggregate_metrics(all_simulated_metrics_list)

    def _process_all_samples(self, df: pl.DataFrame, simulator: GlucoseSimulator, simulation_hours: int) -> List[Dict[str, float]]:
        """Procesa todas las muestras del DataFrame y recopila métricas."""
        all_simulated_metrics_list: List[Dict[str, float]] = []

        for i in range(len(df)):
            metrics = self._process_single_sample(df, i, simulator, simulation_hours)
            if metrics:
                all_simulated_metrics_list.append(metrics)

        return all_simulated_metrics_list

    def _process_single_sample(self, df: pl.DataFrame, sample_index: int, simulator: GlucoseSimulator, simulation_hours: int) -> Optional[Dict[str, float]]:
        """Procesa una muestra individual y retorna sus métricas."""
        current_sample_df_row = df.slice(sample_index, 1)
        
        initial_glucose_val = current_sample_df_row.select(pl.col('glucose_last')).item()
        carb_intake_val = current_sample_df_row.select(pl.col('meal_carbs')).item()

        predicted_dose = self._get_predicted_dose(current_sample_df_row, sample_index)
        if predicted_dose is None:
            return None

        glucose_trajectory = self._simulate_glucose_trajectory(
            simulator, initial_glucose_val, predicted_dose, carb_intake_val, simulation_hours
        )

        return self._calculate_trajectory_metrics(glucose_trajectory, sample_index)

    def _get_predicted_dose(self, current_sample_df_row: pl.DataFrame, sample_index: int) -> Optional[float]:
        """Obtiene la dosis predicha para una muestra, manejando errores."""
        try:
            return self.predict_with_context(current_sample_df_row)
        except Exception as e:
            print_error(f"Error al llamar a predict_with_context en evaluate_clinical para la muestra {sample_index}: {e}")
            return None

    def _simulate_glucose_trajectory(self, simulator: GlucoseSimulator, initial_glucose: float, 
                                   predicted_dose: float, carb_intake: float, simulation_hours: int) -> Optional[List[float]]:
        """Simula la trayectoria de glucosa usando el simulador."""
        return simulator.predict_glucose_trajectory(
            initial_glucose=initial_glucose,
            insulin_doses=[predicted_dose],
            carb_intakes=[carb_intake],
            timestamps=[0.0],
            prediction_horizon=simulation_hours
        )

    def _calculate_trajectory_metrics(self, glucose_trajectory: Optional[List[float]], sample_index: int) -> Optional[Dict[str, float]]:
        """Calcula métricas para una trayectoria de glucosa."""
        if glucose_trajectory is None or len(glucose_trajectory) == 0:
            print_warning(f"Trayectoria de glucosa vacía o None para la muestra {sample_index}.")
            return None

        metrics = ClinicalMetricsEvaluator.evaluate_trajectory(glucose_trajectory)
        return metrics if metrics else None

    def _aggregate_metrics(self, all_simulated_metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Agrega todas las métricas calculadas."""
        aggregated_metrics: Dict[str, List[float]] = {k: [] for k in all_simulated_metrics_list[0].keys()}
        
        for m_dict in all_simulated_metrics_list:
            for key, value in m_dict.items():
                if isinstance(value, (int, float)):
                    aggregated_metrics[key].append(value)

        averaged_results: Dict[str, float] = {}
        for key, values_list in aggregated_metrics.items():
            if values_list:
                averaged_results[key] = float(np.nanmean(values_list))
            else:
                averaged_results[key] = np.nan

        return averaged_results

    def save(self, path: str) -> None:
        """
        Guarda el modelo en la ruta especificada.

        Parámetros:
        -----------
        path : str
            Ruta donde guardar el modelo.
        
        Retorna:
        --------
        None
        """
        raise NotImplementedError("El método save debe ser implementado por las subclases.")

    def load(self, path: str) -> None:
        """
        Carga el modelo desde la ruta especificada.

        Parámetros:
        -----------
        path : str
            Ruta desde donde cargar el modelo.

        Retorna:
        --------
        None
        """
        raise NotImplementedError("El método load debe ser implementado por las subclases.")

    def _predict_insulin_doses_with_full_context(self, 
                                                 x_cgm_batch: np.ndarray, 
                                                 x_other_batch: np.ndarray, 
                                                 context_batch: Dict[str, np.ndarray]) -> List[float]:
        """
        Predice dosis de insulina para un lote de muestras usando contexto completo.

        Parámetros:
        -----------
        x_cgm_batch : np.ndarray
            Datos CGM para predicción (N_muestras, ventana, características_cgm).
        x_other_batch : np.ndarray
            Otras características para predicción (N_muestras, características_other).
        context_batch : Dict[str, np.ndarray]
            Diccionario donde cada valor es un array de N_muestras para una característica contextual
            (ej: 'current_glucose', 'carb_intake', 'iob', 'activity_level', etc.).

        Retorna:
        --------
        List[float]
            Lista de dosis de insulina predichas para cada muestra.
        """
        predicted_doses: List[float] = []
        num_samples = len(x_cgm_batch)

        for i in range(num_samples):
            context_values = self._extract_context_values(context_batch, i)
            dose = self._predict_single_dose(x_cgm_batch, x_other_batch, i, context_values)
            predicted_doses.append(float(dose))
        return predicted_doses

    def _extract_context_values(self, context_batch: Dict[str, np.ndarray], sample_index: int) -> Dict[str, Optional[float]]:
        """
        Extrae valores de contexto para una muestra específica.
        
        Parámetros:
        -----------
        context_batch : Dict[str, np.ndarray]
            Diccionario de arrays de contexto.
        sample_index : int
            Índice de la muestra a extraer.
            
        Retorna:
        --------
        Dict[str, Optional[float]]
            Valores de contexto extraídos para la muestra.
        """
        # Valores requeridos con defaults
        required_values = {
            'current_glucose': self._get_required_context_value(context_batch, 'current_glucose', sample_index, 0.0),
            'carb_intake': self._get_required_context_value(context_batch, 'carb_intake', sample_index, 0.0),
            'iob': self._get_required_context_value(context_batch, 'iob', sample_index, 0.0)
        }
        
        # Valores opcionales
        optional_values = {
            'activity_level': self._get_optional_context_value(context_batch, 'activity_level', sample_index),
            'stress_level': self._get_optional_context_value(context_batch, 'stress_level', sample_index),
            'work_intensity': self._get_optional_context_value(context_batch, 'work_intensity', sample_index),
            'sleep_quality': self._get_optional_context_value(context_batch, 'sleep_quality', sample_index),
            'target_glucose': self._get_optional_context_value(context_batch, 'target_glucose', sample_index)
        }
        
        return {**required_values, **optional_values}

    def _get_required_context_value(self, context_batch: Dict[str, np.ndarray], key: str, index: int, default: float) -> float:
        """Obtiene un valor de contexto requerido con default."""
        return float(context_batch[key][index]) if key in context_batch else default

    def _get_optional_context_value(self, context_batch: Dict[str, np.ndarray], key: str, index: int) -> Optional[float]:
        """Obtiene un valor de contexto opcional que puede ser None."""
        if key not in context_batch or context_batch[key][index] is None:
            return None
        return float(context_batch[key][index])

    def add_early_stopping(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True) -> None:
        """
        Añade configuración de early stopping al modelo.
        
        Parámetros:
        -----------
        patience : int, opcional
            Número de épocas a esperar para detener el entrenamiento si no hay mejora (default: 10).
        min_delta : float, opcional
            Cambio mínimo en la métrica monitoreada que califica como una mejora (default: 0.0).
        restore_best_weights : bool, opcional
            Si se deben restaurar los pesos del modelo de la mejor época al finalizar el early stopping (default: True).
        """
        self.early_stopping = {
            'patience': patience,
            'min_delta': min_delta,
            'restore_best_weights': restore_best_weights,
            'best_loss': float('inf'), # O -float('inf') si se monitorea una métrica a maximizar
            'best_params': None, # Para almacenar los pesos del mejor modelo
            'wait': 0 # Contador de épocas sin mejora
        }