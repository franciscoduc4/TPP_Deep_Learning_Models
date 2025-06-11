from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
# Imports for JAX/Flax are not used in this specific file after modifications,
# but kept if other wrappers in the same original file might use them.
# import jax
# import jax.numpy as jnp
# import flax.linen as nn

from constants.constants import CONST_DEFAULT_EPOCHS, CONST_DEFAULT_BATCH_SIZE
from validation.evaluator import ClinicalMetricsEvaluator
from validation.simulator import GlucoseSimulator
from custom.printer import print_warning

class ModelWrapper:
    """
    Clase base para encapsular modelos de aprendizaje profundo y por refuerzo.
    
    Proporciona una interfaz unificada para inicialización, entrenamiento y predicción.
    Todos los modelos deben heredar de esta clase.
    """
    
    def __init__(self) -> None:
        """
        Constructor base para ModelWrapper.
        Inicializa la configuración de early stopping.
        """
        self.early_stopping: Optional[Dict[str, Any]] = None

    def start(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, 
                 rng_key: Optional[Any] = None) -> Any:
        """
        Inicializa el modelo con los datos de entrada.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrada.
        x_other : np.ndarray
            Otras características de entrada.
        y : np.ndarray
            Valores objetivo.
        rng_key : Optional[Any], opcional
            Clave para generación aleatoria (default: None).
            
        Retorna:
        --------
        Any
            Estado inicial del modelo o parámetros.
        """
        raise NotImplementedError("El método start debe ser implementado por las subclases.")
    
    def fit(self, x: Union[np.ndarray, List[np.ndarray]], y: np.ndarray,
           validation_data: Optional[Tuple[Union[np.ndarray, List[np.ndarray]], np.ndarray]] = None,
           epochs: int = CONST_DEFAULT_EPOCHS, batch_size: int = CONST_DEFAULT_BATCH_SIZE, verbose: int = 1) -> Dict[str, List[float]]:
        """
        Entrena el modelo con los datos proporcionados.
        
        Parámetros:
        -----------
        x : Union[np.ndarray, List[np.ndarray]]
            Datos de entrada. Si es una lista, se espera [x_cgm, x_other].
        y : np.ndarray
            Valores objetivo.
        validation_data : Optional[Tuple[Union[np.ndarray, List[np.ndarray]], np.ndarray]], opcional
            Datos de validación como ((x_cgm_val, x_other_val), y_val) o (x_val, y_val) (default: None).
        epochs : int, opcional
            Número de épocas de entrenamiento (default: ver constantes).
        batch_size : int, opcional
            Tamaño de lote (default: ver constantes).
        verbose : int, opcional
            Nivel de verbosidad (0=silencioso, 1=progreso).
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historial de entrenamiento con métricas.
        """
        raise NotImplementedError("El método fit debe ser implementado por las subclases.")

    def predict(self, x_cgm: np.ndarray, x_other: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones con el modelo entrenado.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM para predicción.
        x_other : np.ndarray
            Otras características para predicción.
            
        Retorna:
        --------
        np.ndarray
            Predicciones del modelo.
        """
        raise NotImplementedError("El método predict debe ser implementado por las subclases.")
    
    def predict_with_context(self, x_cgm: np.ndarray, x_other: np.ndarray, 
                        current_glucose: float,
                        carb_intake: float, 
                        iob: float,
                        activity_level: Optional[float] = None,
                        stress_level: Optional[float] = None,
                        work_intensity: Optional[float] = None, 
                        sleep_quality: Optional[float] = None,
                        target_glucose: Optional[float] = None) -> float:
        """
        Realiza predicciones con el modelo entrenado, considerando contexto adicional.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM para predicción (ej: ventana de historial reciente).
        x_other : np.ndarray
            Otras características generales para predicción.
        current_glucose : float
            Nivel actual de glucosa en mg/dL.
        carb_intake : float
            Ingesta de carbohidratos en gramos.
        iob : float
            Insulina a bordo en unidades.
        activity_level : Optional[float], opcional
            Nivel de actividad física (ej: mapeado desde exercise_intensity).
        stress_level : Optional[float], opcional
            Nivel de estrés general.
        work_intensity : Optional[float], opcional
            Intensidad del trabajo.
        sleep_quality : Optional[float], opcional
            Calidad del sueño.
        target_glucose : Optional[float], opcional
            Nivel objetivo de glucosa en mg/dL.
                
        Retorna:
        --------
        float
            Dosis de insulina recomendada en unidades.
        """
        raise NotImplementedError("El método predict_with_context debe ser implementado por las subclases.")
    
    def evaluate(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray,
                 context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Evalúa el modelo con datos de prueba.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de prueba.
        x_other : np.ndarray
            Otras características de prueba.
        y : np.ndarray
            Valores objetivo reales.
        context : Optional[Dict[str, Any]], opcional
            Contexto adicional para la evaluación (default: None).
            
        Retorna:
        --------
        Dict[str, float]
            Diccionario de métricas de evaluación (ej: {'mse_loss': valor}).
        """
        # Implementación base simple, las subclases pueden sobreescribirla.
        preds_np = self.predict(x_cgm, x_other)
        if y is not None and len(y) == len(preds_np):
            mse = float(np.mean((preds_np.flatten() - y.flatten()) ** 2))
            return {'mse_loss': mse}
        return {'mse_loss': float('nan')}

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

    def _predict_single_dose(self, x_cgm_batch: np.ndarray, x_other_batch: np.ndarray, 
                           sample_index: int, context_values: Dict[str, Optional[float]]) -> float:
        """
        Predice dosis para una sola muestra.
        
        Parámetros:
        -----------
        x_cgm_batch : np.ndarray
            Datos CGM del lote.
        x_other_batch : np.ndarray
            Otras características del lote.
        sample_index : int
            Índice de la muestra.
        context_values : Dict[str, Optional[float]]
            Valores de contexto extraídos.
            
        Retorna:
        --------
        float
            Dosis predicha.
        """
        return self.predict_with_context(
            x_cgm=x_cgm_batch[sample_index:sample_index+1], 
            x_other=x_other_batch[sample_index:sample_index+1],
            current_glucose=context_values['current_glucose'],
            carb_intake=context_values['carb_intake'],
            iob=context_values['iob'],
            activity_level=context_values['activity_level'],
            stress_level=context_values['stress_level'],
            work_intensity=context_values['work_intensity'],
            sleep_quality=context_values['sleep_quality'],
            target_glucose=context_values['target_glucose']
        )

    def _simulate_glucose_trajectories(self, simulator: GlucoseSimulator, 
                                     initial_glucose: np.ndarray,
                                     predicted_doses: List[float], 
                                     carb_intake_for_sim: np.ndarray, # Carbs para el simulador
                                     simulation_hours: int) -> List[np.ndarray]:
        """
        Simula trayectorias de glucosa usando las dosis predichas.
        
        Parámetros:
        -----------
        simulator : GlucoseSimulator
            Simulador de glucosa para generar trayectorias.
        initial_glucose : np.ndarray
            Valores iniciales de glucosa para la simulación.
        predicted_doses : List[float]
            Dosis de insulina predichas para cada muestra.
        carb_intake_for_sim : np.ndarray
            Ingesta de carbohidratos para cada muestra (usada por el simulador).
        simulation_hours : int
            Duración de la simulación en horas.
        
        Retorna:
        --------
        List[np.ndarray]
            Lista de trayectorias de glucosa simuladas.
        """
        glucose_trajectories: List[np.ndarray] = []
        for i in range(len(initial_glucose)):
            # Asegurarse que carb_intake_for_sim tiene el valor correcto para la muestra i
            current_carb_intake = float(carb_intake_for_sim[i]) if i < len(carb_intake_for_sim) else 0.0
            
            glucose_trajectory = simulator.predict_glucose_trajectory(
                initial_glucose=float(initial_glucose[i]),
                insulin_doses=[predicted_doses[i]],
                carb_intakes=[current_carb_intake], # Usar el carbohidrato de la muestra actual
                timestamps=[0], # Asumiendo que la dosis y los carbohidratos son en t=0 para la simulación
                prediction_horizon=simulation_hours
            )
            glucose_trajectories.append(glucose_trajectory)
        return glucose_trajectories
    
    def _aggregate_metrics(self, glucose_trajectories: List[np.ndarray]) -> Dict[str, float]:
        """
        Calcula y agrega métricas clínicas para todas las trayectorias.
        
        Parámetros:
        -----------
        glucose_trajectories : List[np.ndarray]
            Lista de trayectorias de glucosa simuladas.
        
        Retorna:
        --------
        Dict[str, float]
            Diccionario con métricas clínicas promediadas.
        """
        accumulated_metrics: Dict[str, List[float]] = {}
        for trajectory in glucose_trajectories:
            # ClinicalMetricsEvaluator.evaluate_clinical_metrics espera una trayectoria 1D
            # y devuelve un diccionario de métricas escalares.
            metrics = ClinicalMetricsEvaluator.evaluate_clinical_metrics(trajectory.flatten())
            self._update_metrics_dict(accumulated_metrics, metrics)
        
        # Promediar métricas
        averaged_metrics: Dict[str, float] = {}
        for key, values in accumulated_metrics.items():
            if values: # Asegurarse de que la lista no esté vacía
                averaged_metrics[key] = float(np.mean(values))
            else:
                averaged_metrics[key] = float('nan') # O 0.0, según preferencia
        return averaged_metrics
    
    def _update_metrics_dict(self, all_metrics: Dict[str, List[float]], 
                           metrics: Dict[str, Any]) -> None:
        """
        Actualiza el diccionario de métricas con nuevos valores.
        
        Parámetros:
        -----------
        all_metrics : Dict[str, List[float]]
            Diccionario acumulativo de métricas (lista de valores por clave).
        metrics : Dict[str, Any]
            Nuevas métricas a agregar (valores escalares).
        
        Retorna:
        --------
        None
        """
        for key, value in metrics.items():
            if not isinstance(value, (int, float, np.number)):
                # print_warning(f"Métrica '{key}' con valor no numérico '{value}' será ignorada en _update_metrics_dict.")
                continue # Ignorar métricas no numéricas para promediar

            if key not in all_metrics:
                all_metrics[key] = []
            all_metrics[key].append(float(value))
    
    def evaluate_clinical(self, 
                          simulator: GlucoseSimulator, 
                          x_cgm: np.ndarray, 
                          x_other: np.ndarray, 
                          context_data: Dict[str, np.ndarray], # Diccionario de arrays de contexto
                          initial_glucose: np.ndarray, 
                          ground_truth_insulin: Optional[np.ndarray] = None, 
                          simulation_hours: int = 24) -> Dict[str, float]:
        """
        Evalúa el modelo utilizando métricas clínicas con un simulador de glucosa.
        
        Parámetros:
        -----------
        simulator : GlucoseSimulator
            Simulador de glucosa.
        x_cgm : np.ndarray
            Datos CGM para predicción.
        x_other : np.ndarray
            Otras características para predicción.
        context_data : Dict[str, np.ndarray]
            Diccionario con arrays de datos contextuales. Claves deben incluir
            'current_glucose', 'carb_intake', 'iob', y opcionalmente otras como
            'activity_level', 'stress_level', 'work_intensity', 'sleep_quality', 'target_glucose'.
            Cada valor debe ser un np.ndarray de la misma longitud que x_cgm.
        initial_glucose : np.ndarray
            Valores iniciales de glucosa para simulación.
        ground_truth_insulin : Optional[np.ndarray], opcional
            Dosis reales de insulina para comparación (actualmente no usado en cálculo de métricas).
        simulation_hours : int, opcional
            Duración de la simulación en horas (default: 24).
                
        Retorna:
        --------
        Dict[str, float]
            Métricas clínicas de evaluación.
        """
        # Paso 1: Predecir dosis de insulina usando el contexto completo
        predicted_doses = self._predict_insulin_doses_with_full_context(x_cgm, x_other, context_data)
        
        # carb_intake para el simulador debe venir de context_data
        carb_intake_for_sim = context_data.get('carb_intake', np.zeros_like(initial_glucose))
        if len(carb_intake_for_sim) != len(initial_glucose):
            print_warning("Longitud de carb_intake_for_sim no coincide con initial_glucose. Usando ceros.")
            carb_intake_for_sim = np.zeros_like(initial_glucose)

        # Paso 2: Simular trayectorias de glucosa
        glucose_trajectories = self._simulate_glucose_trajectories(
            simulator, initial_glucose, predicted_doses, carb_intake_for_sim, simulation_hours)
        
        # Paso 3: Calcular y devolver métricas clínicas
        aggregated_metrics = self._aggregate_metrics(glucose_trajectories)

        if ground_truth_insulin is not None:
            # Opcional: calcular MSE si hay insulina real
            if len(predicted_doses) == len(ground_truth_insulin):
                mse_doses = np.mean((np.array(predicted_doses) - ground_truth_insulin)**2)
                aggregated_metrics['dose_mse'] = float(mse_doses)
            else:
                print_warning("Longitudes de dosis predichas e insulina real no coinciden para MSE.")
        
        return aggregated_metrics
    
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