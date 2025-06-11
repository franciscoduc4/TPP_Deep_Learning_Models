from typing import Dict, List, Tuple, Any, Optional, Callable, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm # Usar tqdm.auto para compatibilidad con notebooks y scripts
import os 

from config.models_config import EARLY_STOPPING_POLICY
from constants.constants import (
    CONST_DEFAULT_BATCH_SIZE, CONST_DEFAULT_EPOCHS, 
    IDEAL_LOWER_BOUND, IDEAL_UPPER_BOUND, 
    CONST_MODEL_INIT_ERROR, CONST_DEFAULT_SEED, CONST_EPSILON,
    CONST_LOSS, CONST_VAL_LOSS, CONST_AVERAGE_REWARD, 
    CONST_ACTOR_LOSS, CONST_CRITIC_LOSS 
)
from custom.model_wrapper import ModelWrapper
from custom.printer import print_critical, print_debug, print_error, print_info, print_success, print_warning
# Asumiendo que ReplayBuffer está definido en alguna parte accesible
# from models.utils.replay_buffer import ReplayBuffer 

# Mensajes de error y advertencia
MSG_CTX_UNPACK_ERROR = "Error al desempaquetar datos de validación con contexto: {}"
MSG_CTX_MISSING_IN_FIT = "Contexto de entrenamiento no proporcionado a fit, pero el modelo subyacente podría requerirlo."
MSG_CTX_MISSING_IN_VAL = "Contexto de validación no proporcionado, pero el modelo subyacente podría requerirlo para evaluación."
MSG_NO_RUN_TRAINING_STEP = "El modelo DRL subyacente no tiene 'run_training_step'. Se intentará un bucle genérico de actualización o se advertirá."
MSG_NO_EVAL_PERFORMANCE = "El modelo DRL subyacente no tiene 'evaluate_performance'. Se usará MSE de acción para validación si y_val está disponible, o se advertirá."
MSG_INIT_OPTIMIZER_NO_PARAMS = "El modelo DRL {} no tiene parámetros entrenables. El optimizador no se inicializará."
MSG_INIT_OPTIMIZER_SUCCESS = "Optimizador {} inicializado con LR={} para el modelo DRL."
MSG_INIT_OPTIMIZER_FAIL = CONST_MODEL_INIT_ERROR.format("inicializar optimizador (modelo no es nn.Module o es None)")
MSG_EARLY_STOPPING_CONFIGURED = "Early stopping configurado para DRL con paciencia={}."
MSG_EARLY_STOPPING_ACTIVATED = "Early stopping activado en época {}."
MSG_TRAINING_COMPLETE = "Entrenamiento de {} (DRL) completado."
MSG_RESTORE_BEST_WEIGHTS_DRL = "Restaurando mejores pesos del modelo DRL (métrica val: {:.4f})."
MSG_SAVE_DRL_SUCCESS = "Modelo DRL ({}) guardado en: {}"
MSG_LOAD_DRL_SUCCESS = "Modelo DRL ({}) cargado desde: {}"
MSG_PREDICT_CTX_NOT_IMPLEMENTED_UNDERLYING = "El modelo DRL subyacente ({}) no implementa 'predict_with_context'. No se puede realizar la predicción contextual detallada."
MSG_PREDICT_CTX_NOT_IMPLEMENTED_ERROR = "El modelo DRL subyacente ({}) debe implementar 'predict_with_context' para manejar el contexto detallado."
MSG_ATTR_ERROR_BUFFER = "El modelo DRL subyacente no tiene un atributo 'buffer' o 'replay_buffer'."
MSG_ATTR_ERROR_CGM_DIM = "El modelo DRL subyacente debe tener 'cgm_input_dim' (timesteps, features)."
MSG_ATTR_ERROR_OTHER_DIM = "El modelo DRL subyacente debe tener 'other_input_dim' (features_len,)."
MSG_UNSUPPORTED_INPUT_LIST = "Lista de entrada x debe tener 1 o 2 arrays (x_cgm, [x_other])."
MSG_UNSUPPORTED_INPUT_LIST = "Lista de entrada x debe tener 1 o 2 arrays (x_cgm, [x_other])."
MSG_SELECT_ACTION_ROLLOUT_FAIL = "El modelo DRL subyacente ({}) no implementa 'select_action' o falló la llamada."

class DRLModelWrapperPyTorch(ModelWrapper, nn.Module):
    """
    Wrapper para modelos de aprendizaje por refuerzo profundo implementados en PyTorch.
    Delega la funcionalidad al modelo DRL subyacente (ej. DDPG, SAC).
    
    Parámetros:
    -----------
    model_or_cls : Union[Callable[..., nn.Module], nn.Module]
        Clase del modelo DRL a instanciar o instancia ya creada del modelo.
        Se espera que este modelo contenga la lógica específica del algoritmo DRL.
    algorithm : str, opcional
        Nombre del algoritmo DRL (default: "generic").
    **model_kwargs : dict
        Argumentos para el constructor del modelo (usado solo si se pasa una clase de modelo).
    """
    
    def __init__(self, model_or_cls: Union[Callable[..., nn.Module], nn.Module], 
                 algorithm: str = "generic", **model_kwargs: Any) -> None:
        ModelWrapper.__init__(self) 
        nn.Module.__init__(self)    
        
        self.is_class = isinstance(model_or_cls, type) 
    
        if self.is_class:
            self.model_cls: Optional[Callable[..., nn.Module]] = model_or_cls
            self.model: Optional[nn.Module] = None 
        else:
            self.model_cls = type(model_or_cls)
            self.model: Optional[nn.Module] = model_or_cls # type: ignore
            
        self.model_kwargs: Dict[str, Any] = model_kwargs
        self.algorithm: str = model_kwargs.get('algorithm', algorithm)
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.model is not None and isinstance(self.model, nn.Module):
            self.model = self.model.to(self.device)
            
        seed = model_kwargs.get('seed', CONST_DEFAULT_SEED)
        if seed is None: 
            seed = CONST_DEFAULT_SEED
        self.rng: np.random.Generator = np.random.default_rng(seed)
        
        self.optimizer: Optional[optim.Optimizer] = None

    @property
    def replay_buffer(self) -> Any:
        """
        Accede al buffer de experiencia del modelo DRL subyacente.
        
        Retorna:
        --------
        Any
            Buffer de experiencia del modelo.
            
        Levanta:
        -------
        AttributeError
            Si el modelo no está inicializado o no tiene un buffer accesible.
        """
        if self.model is None:
            raise AttributeError(CONST_MODEL_INIT_ERROR.format("acceder al buffer de repetición"))
        
        if hasattr(self.model, 'buffer'):
            return self.model.buffer # type: ignore
        elif hasattr(self.model, 'replay_buffer'):
            return self.model.replay_buffer # type: ignore
        else:
            raise AttributeError(MSG_ATTR_ERROR_BUFFER)

    def start(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, 
             rng_key: Optional[Any] = None) -> Any:
        """
        Inicializa el modelo DRL subyacente si aún no está instanciado,
        y luego llama a su método de inicialización si existe.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrada.
        x_other : np.ndarray
            Otras características de entrada.
        y : np.ndarray
            Valores objetivo (pueden ser acciones o no usados directamente en DRL).
        rng_key : Optional[Any], opcional
            Clave para generación aleatoria (default: None), más común en JAX.
            
        Retorna:
        --------
        Any
            El modelo inicializado o su estado/parámetros.
        """
        self._instantiate_model_if_needed()
        self._validate_model_exists()
        self._ensure_model_on_device()
        
        return self._initialize_model_with_data(x_cgm, x_other, y, rng_key)

    def _instantiate_model_if_needed(self) -> None:
        """Instancia el modelo DRL si aún no está creado."""
        if self.model is None and self.model_cls is not None:
            print_info(f"Instanciando modelo DRL: {self.model_cls.__name__}")
            if 'seed' not in self.model_kwargs and hasattr(self.rng, '_bit_generator'):
                 self.model_kwargs['seed'] = self.rng._bit_generator.seed_seq.entropy[0] # type: ignore
            self.model = self.model_cls(**self.model_kwargs) # type: ignore

    def _validate_model_exists(self) -> None:
        """Valida que el modelo esté inicializado."""
        if self.model is None:
            raise ValueError(CONST_MODEL_INIT_ERROR.format("inicializar (modelo es None)"))

    def _ensure_model_on_device(self) -> None:
        """Asegura que el modelo esté en el dispositivo correcto."""
        if isinstance(self.model, nn.Module):
            self.model = self.model.to(self.device)

    def _initialize_model_with_data(self, x_cgm: np.ndarray, x_other: np.ndarray, 
                                   y: np.ndarray, rng_key: Optional[Any]) -> Any:
        """Inicializa el modelo con los datos proporcionados."""
        if hasattr(self.model, 'start'):
            return self.model.start(x_cgm, x_other, y, rng_key=rng_key) # type: ignore
        elif hasattr(self.model, 'initialize'):
            state_dim_info = self._build_state_dim_info(x_cgm, x_other)
            action_dim = y.shape[-1] if y.ndim > 1 else 1
            return self.model.initialize(state_dim_info, action_dim, rng_key=rng_key) # type: ignore
            
        print_warning(f"El modelo DRL subyacente ({type(self.model).__name__}) no tiene método 'start' ni 'initialize'. Se devuelve el modelo tal cual.")
        return self.model

    def _build_state_dim_info(self, x_cgm: np.ndarray, x_other: np.ndarray) -> Dict[str, Tuple[int, ...]]:
        """Construye información de dimensiones del estado."""
        cgm_input_shape = x_cgm.shape[1:] if x_cgm.ndim > 1 else (0,)
        other_input_shape = x_other.shape[1:] if x_other.ndim > 1 and x_other.shape[1] > 0 else (0,)
        return {'cgm_shape': cgm_input_shape, 'other_shape': other_input_shape}

    def fit(self, 
            x: Union[np.ndarray, List[np.ndarray]], 
            y: np.ndarray,
            context_data_train: Optional[Dict[str, np.ndarray]] = None,
            validation_data: Optional[Tuple[Union[np.ndarray, List[np.ndarray]], np.ndarray, Optional[Dict[str, np.ndarray]]]] = None,
            epochs: int = CONST_DEFAULT_EPOCHS, 
            batch_size: int = CONST_DEFAULT_BATCH_SIZE, 
            verbose: int = 1) -> Dict[str, List[float]]:
        """
        Entrena el modelo DRL. Esta función orquesta el bucle de entrenamiento DRL.
        Delega la lógica de actualización de la época al método `_run_drl_training_epoch`,
        que a su vez debe delegar al método `run_training_step` del modelo DRL subyacente.

        Parámetros:
        -----------
        x : Union[np.ndarray, List[np.ndarray]]
            Datos de entrada. Si es lista: [x_cgm, x_other].
        y : np.ndarray
            Valores objetivo (ej: acciones históricas para BC o DRL offline).
        context_data_train : Optional[Dict[str, np.ndarray]], opcional
            Contexto para los datos de entrenamiento (ej: {'carb_intake': array, 'iob': array}).
        validation_data : Optional[Tuple[Union[np.ndarray, List[np.ndarray]], np.ndarray, Optional[Dict[str, np.ndarray]]]], opcional
            Datos de validación: (x_val, y_val, context_data_val).
        epochs : int, opcional
            Número de épocas.
        batch_size : int, opcional
            Tamaño de lote para actualizaciones del agente DRL (pasado a `run_training_step`).
        verbose : int, opcional
            Nivel de verbosidad.

        Retorna:
        --------
        Dict[str, List[float]]
            Historial de entrenamiento.
        """
        x_cgm, x_other = self._unpack_input_data(x)
        self._initialize_training_components(x_cgm, x_other, y)
        
        x_cgm_val, x_other_val, y_val, context_data_val = self._unpack_validation_data_with_context(validation_data)
        do_validation = x_cgm_val is not None
        
        history = self._setup_history_drl()
        self._setup_early_stopping_drl(verbose)

        epoch_iterator = tqdm(range(epochs), desc=f"Entrenando {self.algorithm} (DRL)", 
                              disable=(verbose == 0), unit="época")

        for epoch in epoch_iterator:
            epoch_metrics = self._run_drl_training_epoch(
                x_cgm, x_other, y, context_data_train, batch_size, epoch
            )
            current_epoch_loss = self._process_epoch_metrics_drl(epoch_metrics, history)
            
            current_val_metric_for_es = self._process_validation_epoch(
                do_validation, x_cgm_val, x_other_val, y_val, context_data_val, history
            )

            self._log_epoch_progress_drl(epoch, epochs, current_epoch_loss, 
                                       getattr(self, '_last_val_metrics', None), history, epoch_iterator)

            if self.early_stopping and self._check_early_stopping_drl(current_val_metric_for_es):
                if verbose > 0:
                    print_info(MSG_EARLY_STOPPING_ACTIVATED.format(epoch + 1))
                break
        
        self._restore_best_weights_drl(verbose)
        
        if verbose > 0:
            print_success(MSG_TRAINING_COMPLETE.format(self.algorithm))
        return history

    def predict(self, x_cgm: np.ndarray, x_other: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones (selección de acciones determinísticas) con el modelo DRL.
        Delega al método `select_action` o `actor` del modelo DRL subyacente.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM para predicción.
        x_other : np.ndarray
            Otras características para predicción.
            
        Retorna:
        --------
        np.ndarray
            Acciones predichas por el modelo.
        """
        if self.model is None:
            raise ValueError(CONST_MODEL_INIT_ERROR.format("predecir"))
    
        self.eval() 
        x_cgm_t = torch.FloatTensor(x_cgm).to(self.device)
        x_other_t = torch.FloatTensor(x_other).to(self.device)
        
        with torch.no_grad():
            if hasattr(self.model, 'select_action'):
                # Asumimos que select_action(state_cgm, state_other, add_noise=False) es para inferencia determinista
                actions_t = self.model.select_action(x_cgm_t, x_other_t, add_noise=False) # type: ignore
            elif hasattr(self.model, 'actor') and callable(self.model.actor): # type: ignore
                actions_t = self.model.actor(x_cgm_t, x_other_t) # type: ignore
            elif callable(self.model): 
                # Fallback si el modelo mismo es el actor (menos común para DRL complejos)
                print_warning(f"El modelo DRL ({type(self.model).__name__}) no tiene 'select_action' o 'actor'. Se intentará llamar directamente al modelo.")
                actions_t = self.model(x_cgm_t, x_other_t) # type: ignore
            else:
                raise NotImplementedError(f"El modelo DRL ({type(self.model).__name__}) no tiene un método 'select_action', 'actor', ni es llamable para predicción.")
            
            actions_np = actions_t.cpu().numpy()
            # Asegurar que la salida sea (N,) o (N, action_dim)
            return actions_np.reshape(len(x_cgm), -1).squeeze() if actions_np.ndim > 0 else actions_np

    def predict_with_context(self, x_cgm: np.ndarray, x_other: np.ndarray, 
                        current_glucose: float, carb_intake: float, iob: float,
                        exercise_intensity: Optional[float] = None, stress_level: Optional[float] = None,
                        work_intensity: Optional[float] = None, sleep_quality: Optional[float] = None,
                        target_glucose: Optional[float] = None) -> float:
        """
        Realiza predicciones con el modelo DRL entrenado, considerando contexto adicional.
        Este método es para inferencia determinística (sin ruido de exploración).

        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM para predicción (ej: [1, timesteps, cgm_features] o [timesteps, cgm_features]).
        x_other : np.ndarray
            Otras características para predicción (ej: [1, other_features_len] o [other_features_len]).
        current_glucose : float
            Nivel actual de glucosa en mg/dL.
        carb_intake : float
            Ingesta de carbohidratos en gramos.
        iob : float
            Insulina a bordo en unidades.
        activity_level : Optional[float], opcional
            Nivel de actividad física.
        stress_level : Optional[float], opcional
            Nivel de estrés general.
        work_intensity : Optional[float], opcional
            Intensidad del trabajo.
        sleep_quality : Optional[float], opcional
            Calidad del sueño.
        target_glucose : Optional[float], opcional
            Nivel objetivo de glucosa (puede no ser usado por todos los modelos DRL).
                
        Retorna:
        --------
        float
            Dosis de insulina recomendada en unidades.
        """
        self._validate_model_exists()
        self._ensure_model_on_device()
        
        if hasattr(self.model, 'predict_with_context') and callable(getattr(self.model, 'predict_with_context')):
            # Delegar al método del modelo subyacente
            return self.model.predict_with_context(
                x_cgm=x_cgm, x_other=x_other,
                current_glucose=current_glucose, carb_intake=carb_intake, iob=iob,
                exercise_intensity=exercise_intensity, # Usar el nuevo parámetro
                stress_level=stress_level,
                work_intensity=work_intensity, sleep_quality=sleep_quality,
                target_glucose=target_glucose
            )
        else:
            # Fallback o error si el modelo subyacente no lo implementa
            print_error(MSG_PREDICT_CTX_NOT_IMPLEMENTED_UNDERLYING.format(self.algorithm_name))
            raise NotImplementedError(MSG_PREDICT_CTX_NOT_IMPLEMENTED_ERROR.format(self.algorithm_name))
    
    def select_action_for_rollout(self, 
                                  x_cgm_sample: np.ndarray, 
                                  x_other_sample: np.ndarray, 
                                  context_dict: Dict[str, float], 
                                  add_noise: bool = True) -> np.ndarray:
        """
        Selecciona una acción para la interacción con el entorno durante el entrenamiento/rollout.
        Delega al método `select_action` del modelo DRL subyacente.

        Parámetros:
        -----------
        x_cgm_sample : np.ndarray
            Muestra de datos CGM (ventana, ej. forma (timesteps, cgm_features)).
        x_other_sample : np.ndarray
            Muestra de otras características (ej. forma (other_features_len,)).
        context_dict : Dict[str, float]
            Diccionario con valores de contexto escalares para la muestra actual.
        add_noise : bool, opcional
            Si se debe añadir ruido para exploración (default: True).

        Retorna:
        --------
        np.ndarray
            Acción seleccionada (ej. array de dosis de insulina).
        
        Lanza:
        ------
        NotImplementedError
            Si el modelo subyacente no tiene un método `select_action` compatible.
        """
        self._validate_model_exists()
        if not hasattr(self.model, 'select_action'):
            print_error(MSG_SELECT_ACTION_ROLLOUT_FAIL.format(type(self.model).__name__))
            raise NotImplementedError(MSG_SELECT_ACTION_ROLLOUT_FAIL.format(type(self.model).__name__))
        
        try:
            # state_tuple es (x_cgm_sample, x_other_sample)
            action = self.model.select_action(
                state_tuple=(x_cgm_sample, x_other_sample),
                context_dict=context_dict,
                add_noise=add_noise
            )
            return action
        except Exception as e:
            print_error(f"Error al llamar a self.model.select_action: {e}")
            raise NotImplementedError(MSG_SELECT_ACTION_ROLLOUT_FAIL.format(type(self.model).__name__))
    
    def evaluate(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray,
                 context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Evalúa el modelo DRL.
        Delega al método `evaluate_policy` o similar del modelo DRL subyacente si existe,
        o calcula métricas básicas como MSE de acción si `y` es provisto.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de prueba.
        x_other : np.ndarray
            Otras características de prueba.
        y : np.ndarray
            Acciones de referencia (ej: dosis históricas). Puede ser None si la evaluación es puramente basada en simulación/recompensa.
        context : Optional[Dict[str, Any]], opcional
            Contexto adicional para evaluación (ej: simulador, datos contextuales completos para `evaluate_policy`).
            
        Retorna:
        --------
        Dict[str, float]
            Métricas de evaluación.
        """
        if self.model is None:
            raise ValueError(CONST_MODEL_INIT_ERROR.format("evaluar"))
        
        metrics: Dict[str, float] = {}
        
        # Intentar evaluación delegada primero
        metrics = self._try_delegated_evaluation(x_cgm, x_other, y, context)
        
        # Si no hay métricas, calcular MSE como fallback
        if not metrics:
            metrics = self._calculate_fallback_metrics(x_cgm, x_other, y)
        
        return metrics if metrics else {'evaluation_status': 0.0}

    def _try_delegated_evaluation(self, x_cgm: np.ndarray, x_other: np.ndarray, 
                                 y: np.ndarray, context: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Intenta la evaluación delegada usando evaluate_policy del modelo subyacente."""
        metrics: Dict[str, float] = {}
        
        if not hasattr(self.model, 'evaluate_policy'):
            return metrics
            
        if context is None:
            print_warning(f"El modelo {type(self.model).__name__} tiene 'evaluate_policy' pero no se proporcionó contexto para la evaluación delegada.")
            return metrics
            
        try:
            model_eval_metrics = self.model.evaluate_policy(x_cgm, x_other, y, context_dict=context) # type: ignore
            if isinstance(model_eval_metrics, dict):
                metrics.update(model_eval_metrics)
        except Exception as e:
            print_warning(f"Error al llamar a self.model.evaluate_policy con contexto: {e}")
            
        return metrics

    def _calculate_fallback_metrics(self, x_cgm: np.ndarray, x_other: np.ndarray, 
                                   y: np.ndarray) -> Dict[str, float]:
        """Calcula métricas de fallback usando MSE si y está disponible."""
        if y is None:
            print_warning(f"Evaluación DRL para {type(self.model).__name__} no produjo métricas (ej: 'y' no proporcionado, 'evaluate_policy' no disponible/utilizable, o error en su ejecución).")
            return {}
            
        predicted_actions_np = self.predict(x_cgm, x_other)
        if len(y) != len(predicted_actions_np):
            print_warning("Longitudes de y (objetivo) y acciones predichas no coinciden para calcular MSE.")
            return {}
            
        action_mse = float(np.mean((predicted_actions_np.flatten() - y.flatten()) ** 2))
        return {'action_mse': action_mse}

    def save(self, path: str) -> None:
        """
        Guarda el estado del modelo DRL (incluyendo el modelo subyacente) y del optimizador.
        
        Parámetros:
        -----------
        path : str
            Ruta para guardar el checkpoint.
        """
        if self.model is None or not isinstance(self.model, nn.Module):
            raise ValueError(CONST_MODEL_INIT_ERROR.format("guardar (modelo no es nn.Module o es None)"))

        save_content: Dict[str, Any] = {
            'model_state_dict': self.model.state_dict(), # Estado del modelo DRL subyacente
            'model_kwargs': self.model_kwargs, # Kwargs usados para crear el modelo DRL
            'algorithm': self.algorithm,
            'torch_rng_state': torch.get_rng_state()
        }
        if self.optimizer is not None:
            save_content['optimizer_state_dict'] = self.optimizer.state_dict()
        if self.device.type == 'cuda':
            save_content['torch_cuda_rng_state'] = torch.cuda.get_rng_state_all()

        torch.save(save_content, path)
        print_success(MSG_SAVE_DRL_SUCCESS.format(self.algorithm, path))

    def load(self, path: str) -> None:
        """
        Carga el estado del modelo DRL (incluyendo el modelo subyacente) y del optimizador.
        
        Parámetros:
        -----------
        path : str
            Ruta desde donde cargar el checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self._load_model_from_checkpoint(checkpoint)
        self._load_optimizer_from_checkpoint(checkpoint)
        self._load_rng_states_from_checkpoint(checkpoint)
        
        self.algorithm = checkpoint.get('algorithm', self.algorithm)
        print_success(MSG_LOAD_DRL_SUCCESS.format(self.algorithm, path))
        self.eval() # Poner en modo evaluación después de cargar

    def _load_model_from_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Carga el modelo desde el checkpoint."""
        # Re-instanciar el modelo DRL subyacente si es necesario (si se guardó la clase)
        if self.model is None and self.model_cls is not None:
            loaded_kwargs = checkpoint.get('model_kwargs', self.model_kwargs)
            if 'seed' not in loaded_kwargs and hasattr(self.rng, '_bit_generator'):
                 loaded_kwargs['seed'] = self.rng._bit_generator.seed_seq.entropy[0] # type: ignore
            self.model = self.model_cls(**loaded_kwargs) # type: ignore
        
        if self.model is None or not isinstance(self.model, nn.Module):
             raise ValueError(CONST_MODEL_INIT_ERROR.format("cargar (modelo no es nn.Module o es None después de creación)"))

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)

    def _load_optimizer_from_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Carga el estado del optimizador desde el checkpoint."""
        if 'optimizer_state_dict' not in checkpoint:
            return
            
        if self.optimizer is None: # Intentar inicializar si no existe
             self._initialize_optimizer_from_model_params()
             
        if self.optimizer: # Chequear de nuevo si la inicialización fue exitosa
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                print_warning(f"No se pudo cargar el estado del optimizador: {e}. Puede ser necesario re-inicializarlo o asegurar que los parámetros coinciden.")
        else:
            print_warning("Optimizador no inicializado, no se puede cargar su estado.")

    def _load_rng_states_from_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Carga los estados de generadores aleatorios desde el checkpoint."""
        if 'torch_rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['torch_rng_state'].cpu()) # Asegurar que se carga al CPU primero si es necesario
        if 'torch_cuda_rng_state' in checkpoint and self.device.type == 'cuda':
            torch.cuda.set_rng_state_all(checkpoint['torch_cuda_rng_state'])

    def forward(self, # type: ignore[override]
            glucose: float, carb_intake: float, iob: float,
            activity_level: float, stress_level: float, 
            work_intensity: float, sleep_quality: float,
            target_glucose_range: Optional[Tuple[float, float]] = None
            ) -> float:
        """
        Interfaz principal para obtener una dosis de insulina recomendada usando contexto completo.
        Este método construye las entradas `x_cgm` y `x_other` necesarias y luego llama
        a `predict_with_context` para obtener la dosis.

        Parámetros:
        -----------
        glucose : float
            Nivel actual de glucosa en sangre (mg/dL).
        carb_intake : float
            Carbohidratos consumidos (gramos).
        iob : float
            Insulina a bordo (unidades).
        activity_level : float
            Nivel de actividad física (ej: 0-10). Se usa 0.0 si es None.
        stress_level : float
            Nivel de estrés (ej: 0-10). Se usa 0.0 si es None.
        work_intensity : float
            Intensidad del trabajo (ej: 0-10). Se usa 0.0 si es None.
        sleep_quality : float
            Calidad del sueño (ej: 0-4). Se usa 0.0 si es None.
        target_glucose_range : Optional[Tuple[float, float]], opcional
            Rango objetivo de glucosa (min, max) en mg/dL. Si es None, se usa un default.

        Retorna:
        --------
        float
            Dosis de insulina recomendada.
        """
        if self.model is None:
            raise ValueError(CONST_MODEL_INIT_ERROR.format("realizar una predicción forward"))

        effective_target_glucose: Optional[float]
        if target_glucose_range:
            effective_target_glucose = (target_glucose_range[0] + target_glucose_range[1]) / 2.0
        elif hasattr(self.model, 'default_target_glucose'): # El modelo subyacente puede tener un default
            effective_target_glucose = self.model.default_target_glucose # type: ignore
        else: 
            # Se utiliza el punto medio del rango ideal como valor por defecto si no se especifica otro.
            effective_target_glucose = (IDEAL_LOWER_BOUND + IDEAL_UPPER_BOUND) / 2.0

        # Construir x_cgm (historial CGM)
        # El modelo subyacente debe exponer sus dimensiones de entrada esperadas
        if not hasattr(self.model, 'cgm_input_dim') or not getattr(self.model, 'cgm_input_dim', None):
            raise AttributeError(MSG_ATTR_ERROR_CGM_DIM)
        cgm_timesteps, cgm_features = self.model.cgm_input_dim # type: ignore
        # Crear un historial CGM simple usando el valor actual de glucosa
        # Esto es una simplificación; en un caso real, se usaría el historial real.
        x_cgm_hist_np = np.full((1, cgm_timesteps, cgm_features), float(glucose), dtype=np.float32)

        # Construir x_other (otras características)
        if not hasattr(self.model, 'other_input_dim') or not getattr(self.model, 'other_input_dim', None):
             raise AttributeError(MSG_ATTR_ERROR_OTHER_DIM)
        other_features_len = self.model.other_input_dim[0] # type: ignore
        # Crear un array base para x_other. El modelo subyacente en predict_with_context
        # decidirá cómo usar estas o las características del diccionario de contexto.
        x_other_base_np = np.zeros((1, other_features_len), dtype=np.float32)
        
        # Llamar a predict_with_context, que es el método que realmente interactúa
        # con la lógica de predicción contextual del modelo DRL subyacente.
        return self.predict_with_context(
            x_cgm=x_cgm_hist_np, 
            x_other=x_other_base_np, # x_other puede ser un placeholder si el contexto lo maneja todo
            current_glucose=float(glucose), 
            carb_intake=float(carb_intake), 
            iob=float(iob),
            exercise_intensity=float(activity_level) if activity_level is not None else None,
            stress_level=float(stress_level) if stress_level is not None else None,
            work_intensity=float(work_intensity) if work_intensity is not None else None,
            sleep_quality=float(sleep_quality) if sleep_quality is not None else None,
            target_glucose=effective_target_glucose
        )

    def _unpack_input_data(self, x: Union[np.ndarray, List[np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        """Desempaqueta los datos de entrada x en x_cgm y x_other."""
        if isinstance(x, list):
            return self._unpack_list_input(x)
        
        # Si x es un solo ndarray, se asume que es x_cgm
        x_cgm = x
        other_dim = self._get_other_input_dimension()
        return x_cgm, np.zeros((x_cgm.shape[0], other_dim), dtype=np.float32)

    def _unpack_list_input(self, x: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Desempaqueta entrada cuando x es una lista."""
        if len(x) == 2:
            return self._handle_two_element_list(x)
        elif len(x) == 1:
            return self._handle_single_element_list(x)
        raise ValueError(MSG_UNSUPPORTED_INPUT_LIST)

    def _handle_two_element_list(self, x: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Maneja lista con dos elementos."""
        x_cgm, x_other = x[0], x[1]
        if x_other is None and self.model and hasattr(self.model, 'other_input_dim'):
            other_dim = getattr(self.model, 'other_input_dim', [0])[0]
            x_other = np.zeros((x_cgm.shape[0], other_dim), dtype=np.float32)
        return x_cgm, x_other

    def _handle_single_element_list(self, x: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Maneja lista con un elemento."""
        x_cgm = x[0]
        other_dim = self._get_other_input_dimension()
        return x_cgm, np.zeros((x_cgm.shape[0], other_dim), dtype=np.float32)

    def _get_other_input_dimension(self) -> int:
        """Obtiene la dimensión de entrada 'other' del modelo."""
        if not self.model or not hasattr(self.model, 'other_input_dim'):
            return 0
            
        other_dim_attr = getattr(self.model, 'other_input_dim', [0])
        if isinstance(other_dim_attr, (list, tuple)) and len(other_dim_attr) > 0:
            return other_dim_attr[0]
        elif isinstance(other_dim_attr, int):
            return other_dim_attr
        return 0


    def _unpack_validation_data_with_context(self, 
        validation_data: Optional[Tuple[Union[np.ndarray, List[np.ndarray]], Optional[np.ndarray], Optional[Dict[str, np.ndarray]]]]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, np.ndarray]]]:
        """
        Desempaqueta los datos de validación, incluyendo el contexto opcional.
        Permite que y_val sea None, ya que en DRL la evaluación puede ser por recompensa.
        """
        if validation_data is None:
            return None, None, None, None
        try:
            x_val_data: Union[np.ndarray, List[np.ndarray]]
            y_val: Optional[np.ndarray]
            context_val: Optional[Dict[str, np.ndarray]]

            if len(validation_data) == 2: # (x_val_data, y_val_or_context)
                x_val_data, y_val_or_context = validation_data # type: ignore
                if isinstance(y_val_or_context, np.ndarray) or y_val_or_context is None:
                    y_val = y_val_or_context
                    context_val = None
                elif isinstance(y_val_or_context, dict): # Asumir (x_val_data, context_val) y y_val es None
                    y_val = None
                    context_val = y_val_or_context
                else:
                    raise ValueError("Formato de validation_data (2 elementos) no reconocido.")
            elif len(validation_data) == 3: # (x_val_data, y_val, context_val)
                (x_val_data, y_val, context_val) = validation_data # type: ignore
            else:
                raise ValueError("validation_data debe ser una tupla de 2 o 3 elementos.")

            x_cgm_val, x_other_val = self._unpack_input_data(x_val_data)
            return x_cgm_val, x_other_val, y_val, context_val
        except Exception as e:
            print_warning(MSG_CTX_UNPACK_ERROR.format(e))
            return None, None, None, None

    def _initialize_training_components(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray) -> None:
        """Inicializa el modelo (si es necesario) y el optimizador."""
        if self.model is None: # Si el modelo no fue pre-instanciado
            self.start(x_cgm, x_other, y) # Llama a start para instanciar/inicializar self.model
        
        # Inicializar optimizador después de que self.model esté definitivamente disponible
        self._initialize_optimizer_from_model_params()


    def _initialize_optimizer_from_model_params(self, learning_rate: Optional[float] = None) -> None:
        """
        Inicializa el optimizador usando los parámetros del modelo DRL subyacente.
        Usa 'learning_rate' de model_kwargs si está disponible, sino el argumento o un default.
        """
        if self.model is not None and isinstance(self.model, nn.Module):
            model_params = list(self.model.parameters())
            if not model_params:
                print_warning(MSG_INIT_OPTIMIZER_NO_PARAMS.format(self.algorithm))
                self.optimizer = None
                return
            
            # Priorizar LR del argumento, luego de model_kwargs, luego default
            lr_to_use: float = learning_rate if learning_rate is not None \
                else self.model_kwargs.get('learning_rate', 1e-4)

            opt_class_name: str = self.model_kwargs.get('optimizer_cls', 'Adam')
            
            optimizer_cls_map: Dict[str, torch.Type[optim.Optimizer]] = {
                'Adam': optim.Adam, 
                'AdamW': optim.AdamW, 
                'RMSprop': optim.RMSprop,
                'SGD': optim.SGD
            }
            selected_optimizer_cls = optimizer_cls_map.get(opt_class_name, optim.Adam)
            
            # Algunos optimizadores pueden tener kwargs adicionales (ej: weight_decay para AdamW)
            opt_kwargs_from_model = self.model_kwargs.get('optimizer_kwargs', {})

            try:
                self.optimizer = selected_optimizer_cls(model_params, lr=lr_to_use, **opt_kwargs_from_model)
                print_info(MSG_INIT_OPTIMIZER_SUCCESS.format(opt_class_name, lr_to_use))
            except TypeError as e:
                print_warning(f"Error al inicializar optimizador {opt_class_name} con kwargs {opt_kwargs_from_model}: {e}. Intentando sin kwargs adicionales.")
                self.optimizer = selected_optimizer_cls(model_params, lr=lr_to_use)
                print_info(MSG_INIT_OPTIMIZER_SUCCESS.format(opt_class_name, lr_to_use) + " (sin kwargs adicionales)")

        else:
            print_warning(MSG_INIT_OPTIMIZER_FAIL)
            self.optimizer = None
            
    def _setup_history_drl(self) -> Dict[str, List[float]]:
        """Configura el diccionario de historial para el entrenamiento DRL."""
        return {
            CONST_LOSS: [], CONST_ACTOR_LOSS: [], CONST_CRITIC_LOSS: [],
            CONST_AVERAGE_REWARD: [], CONST_VAL_LOSS: [] # CONST_VAL_LOSS se usa para early stopping
        }

    def _setup_early_stopping_drl(self, verbose: int) -> None:
        """Configura el early stopping para DRL."""
        # Usar valores de model_kwargs si están presentes, sino los de EARLY_STOPPING_POLICY
        patience = self.model_kwargs.get('patience', EARLY_STOPPING_POLICY.get('drl_patience', 20))
        min_delta = self.model_kwargs.get('min_delta', EARLY_STOPPING_POLICY.get('drl_min_delta', 0.001)) # Ajustado min_delta
        restore_best = self.model_kwargs.get('restore_best_weights', True)

        if self.early_stopping is None: # Solo configurar si no existe
            self.add_early_stopping(
                patience=patience,
                min_delta=min_delta,
                restore_best_weights=restore_best
            )
            if verbose > 0 and self.early_stopping is not None:
                print_info(MSG_EARLY_STOPPING_CONFIGURED.format(self.early_stopping['patience']))
        
        # Siempre resetear el estado de early stopping al inicio de fit
        if self.early_stopping is not None: 
            self.early_stopping['best_loss'] = float('inf') # Asume que se minimiza una pérdida o -recompensa
            self.early_stopping['wait'] = 0 
            self.early_stopping['best_params'] = None


    def _run_drl_training_epoch(self, x_cgm_data: np.ndarray, x_other_data: np.ndarray, 
                               y_data: np.ndarray, context_data_train: Optional[Dict[str, np.ndarray]],
                               agent_update_batch_size: int, epoch_num: int) -> Dict[str, float]:
        """
        Ejecuta una 'época' de entrenamiento DRL, delegando al método `run_training_step`
        del modelo DRL subyacente.
        """
        if not self._validate_training_components():
            return self._get_empty_metrics()

        self.train()
        epoch_metrics_sum = self._initialize_epoch_metrics()
        
        if not hasattr(self.model, 'run_training_step'):
            print_warning(MSG_NO_RUN_TRAINING_STEP + f" Modelo: {type(self.model).__name__}")
            return {k: float('nan') for k in epoch_metrics_sum}

        num_updates = self._execute_training_steps(
            x_cgm_data, x_other_data, y_data, context_data_train, 
            agent_update_batch_size, epoch_num, epoch_metrics_sum
        )
        
        return self._compute_averaged_metrics(epoch_metrics_sum, num_updates)

    def _validate_training_components(self) -> bool:
        """Valida que el modelo y optimizador estén inicializados."""
        if self.model is None or self.optimizer is None:
            print_warning("Modelo DRL u optimizador no inicializados. Saltando época de entrenamiento DRL.")
            return False
        return True

    def _get_empty_metrics(self) -> Dict[str, float]:
        """Retorna métricas vacías con valores NaN."""
        return {
            CONST_LOSS: float('nan'), 
            CONST_ACTOR_LOSS: float('nan'), 
            CONST_CRITIC_LOSS: float('nan'), 
            CONST_AVERAGE_REWARD: float('nan')
        }

    def _initialize_epoch_metrics(self) -> Dict[str, float]:
        """Inicializa el diccionario de métricas de época."""
        return {
            CONST_LOSS: 0.0, 
            CONST_ACTOR_LOSS: 0.0, 
            CONST_CRITIC_LOSS: 0.0, 
            CONST_AVERAGE_REWARD: 0.0
        }

    def _execute_training_steps(self, x_cgm_data: np.ndarray, x_other_data: np.ndarray,
                              y_data: np.ndarray, context_data_train: Optional[Dict[str, np.ndarray]],
                              agent_update_batch_size: int, epoch_num: int,
                              epoch_metrics_sum: Dict[str, float]) -> int:
        """Ejecuta los pasos de entrenamiento y acumula métricas."""
        steps_per_epoch = self.model_kwargs.get('drl_steps_per_epoch', 1)
        num_updates = 0
        
        for _ in range(steps_per_epoch):
            step_metrics = self.model.run_training_step( # type: ignore
                x_cgm_data=x_cgm_data, 
                x_other_data=x_other_data, 
                y_data=y_data,
                context_data=context_data_train,
                batch_size=agent_update_batch_size,
                optimizer=self.optimizer
            )
            
            if not self._process_step_metrics(step_metrics, epoch_metrics_sum, epoch_num):
                if num_updates == 0:
                    num_updates = 1  # Para evitar división por cero
                break
                
            num_updates += 1
            
        return num_updates

    def _process_step_metrics(self, step_metrics: Any, epoch_metrics_sum: Dict[str, float], 
                            epoch_num: int) -> bool:
        """Procesa las métricas de un paso y las acumula. Retorna True si es exitoso."""
        if isinstance(step_metrics, dict):
            for key in epoch_metrics_sum:
                epoch_metrics_sum[key] += step_metrics.get(key, 0.0)
            return True
        else:
            print_warning(f"run_training_step del modelo {type(self.model).__name__} no devolvió un diccionario de métricas en la época {epoch_num}.")
            return False

    def _compute_averaged_metrics(self, epoch_metrics_sum: Dict[str, float], 
                                num_updates: int) -> Dict[str, float]:
        """Calcula las métricas promediadas de la época."""
        if num_updates > 0:
            return {k: v / num_updates for k, v in epoch_metrics_sum.items()}
        else:
            return {k: float('nan') for k in epoch_metrics_sum}


    def _validate_drl_agent(self, x_cgm_val: np.ndarray, x_other_val: np.ndarray, 
                            y_val: Optional[np.ndarray], context_data_val: Optional[Dict[str, np.ndarray]]
                           ) -> Dict[str, float]:
        """
        Valida el agente DRL, delegando a `evaluate_performance` del modelo subyacente.
        Si `y_val` está presente y `evaluate_performance` no devuelve métricas de recompensa,
        puede calcular MSE de acción como fallback.
        """
        if self.model is None: 
            return {CONST_AVERAGE_REWARD: -float('inf'), CONST_VAL_LOSS: float('inf')}
        
        self.eval()
        val_metrics_result = self._get_default_validation_metrics()
        
        # Evaluar con el modelo si tiene método de evaluación
        self._evaluate_with_model_performance(x_cgm_val, x_other_val, y_val, context_data_val, val_metrics_result)
        
        # Calcular MSE como fallback si es necesario
        self._calculate_mse_fallback(x_cgm_val, x_other_val, y_val, val_metrics_result)
        
        self.train()
        return val_metrics_result

    def _get_default_validation_metrics(self) -> Dict[str, float]:
        """Retorna métricas de validación por defecto."""
        return {
            CONST_AVERAGE_REWARD: -float('inf'),
            CONST_VAL_LOSS: float('inf')
        }

    def _evaluate_with_model_performance(self, x_cgm_val: np.ndarray, x_other_val: np.ndarray, 
                                       y_val: Optional[np.ndarray], context_data_val: Optional[Dict[str, np.ndarray]],
                                       val_metrics_result: Dict[str, float]) -> None:
        """Evalúa usando el método evaluate_performance del modelo si existe."""
        if not hasattr(self.model, 'evaluate_performance'):
            print_warning(MSG_NO_EVAL_PERFORMANCE + f" Modelo: {type(self.model).__name__}")
            return
            
        try:
            perf_metrics = self.model.evaluate_performance( # type: ignore
                x_cgm_val=x_cgm_val, 
                x_other_val=x_other_val, 
                y_val=y_val,
                context_data=context_data_val
            )
            self._process_performance_metrics(perf_metrics, val_metrics_result)
        except Exception as e:
            print_warning(f"Error al llamar a self.model.evaluate_performance: {e}")

    def _process_performance_metrics(self, perf_metrics: Any, val_metrics_result: Dict[str, float]) -> None:
        """Procesa las métricas devueltas por evaluate_performance."""
        if isinstance(perf_metrics, dict):
            val_metrics_result.update(perf_metrics)
        elif isinstance(perf_metrics, (float, int, np.number)):
            self._handle_numeric_performance_metric(perf_metrics, val_metrics_result)
        else:
            print_warning(f"evaluate_performance de {type(self.model).__name__} devolvió un tipo inesperado: {type(perf_metrics)}")

    def _handle_numeric_performance_metric(self, perf_metrics: Union[float, int, np.number], 
                                         val_metrics_result: Dict[str, float]) -> None:
        """Maneja métricas numéricas simples de evaluate_performance."""
        if perf_metrics > -CONST_EPSILON:  # Probablemente una recompensa
            val_metrics_result[CONST_AVERAGE_REWARD] = float(perf_metrics)
            val_metrics_result[CONST_VAL_LOSS] = -float(perf_metrics)  # Para early stopping
        else:  # Probablemente una pérdida
            val_metrics_result[CONST_VAL_LOSS] = float(perf_metrics)

    def _calculate_mse_fallback(self, x_cgm_val: np.ndarray, x_other_val: np.ndarray, 
                              y_val: Optional[np.ndarray], val_metrics_result: Dict[str, float]) -> None:
        """Calcula MSE como fallback si y_val está disponible y faltan métricas."""
        if not self._should_calculate_mse_fallback(y_val, val_metrics_result):
            return
            
        actions_pred_val = self.predict(x_cgm_val, x_other_val)
        if len(y_val) != len(actions_pred_val): # type: ignore
            print_warning("Longitudes de y_val y acciones predichas no coinciden para MSE de validación.")
            return
            
        action_mse_val = float(np.mean((actions_pred_val.flatten() - y_val.flatten())**2)) # type: ignore
        val_metrics_result['action_mse_val'] = action_mse_val
        
        # Si no hay otra métrica de pérdida, usar MSE para early stopping
        if val_metrics_result.get(CONST_VAL_LOSS, float('inf')) == float('inf'):
            val_metrics_result[CONST_VAL_LOSS] = action_mse_val

    def _should_calculate_mse_fallback(self, y_val: Optional[np.ndarray], 
                                     val_metrics_result: Dict[str, float]) -> bool:
        """Determina si se debe calcular MSE como fallback."""
        if y_val is None:
            return False
            
        reward_missing = val_metrics_result.get(CONST_AVERAGE_REWARD, -float('inf')) == -float('inf')
        loss_missing = val_metrics_result.get(CONST_VAL_LOSS, float('inf')) == float('inf')
        
        return reward_missing or loss_missing


    def _process_epoch_metrics_drl(self, epoch_metrics: Dict[str, float], history: Dict[str, List[float]]) -> float:
        """Procesa y registra las métricas de la época DRL en el historial."""
        # La 'pérdida principal' para DRL puede ser la del crítico, o una combinada.
        # Usar CONST_LOSS de epoch_metrics si está, sino default a critic_loss o actor_loss.
        main_loss = epoch_metrics.get(CONST_LOSS, 
                                      epoch_metrics.get(CONST_CRITIC_LOSS, 
                                                        epoch_metrics.get(CONST_ACTOR_LOSS, 0.0)))
        
        history[CONST_LOSS].append(main_loss)
        history[CONST_ACTOR_LOSS].append(epoch_metrics.get(CONST_ACTOR_LOSS, 0.0))
        history[CONST_CRITIC_LOSS].append(epoch_metrics.get(CONST_CRITIC_LOSS, 0.0))
        history[CONST_AVERAGE_REWARD].append(epoch_metrics.get(CONST_AVERAGE_REWARD, 0.0))
        return main_loss # Devuelve la pérdida principal de la época de entrenamiento

    def _log_epoch_progress_drl(self, epoch: int, epochs: int, train_loss: float, 
                               val_metrics: Optional[Dict[str, float]], 
                               history: Dict[str, List[float]],
                               pbar: tqdm) -> None:
        """Registra el progreso de la época DRL en la barra de progreso y opcionalmente en logs."""
        log_msg_parts = [f"Época {epoch + 1}/{epochs}"]
        
        # Añadir métricas de entrenamiento
        self._add_training_metrics_to_log(history, log_msg_parts)
        
        # Añadir métricas de validación
        self._add_validation_metrics_to_log(val_metrics, log_msg_parts)
        
        # Actualizar barra de progreso y log opcional
        log_msg = " - ".join(log_msg_parts)
        pbar.set_description(log_msg)
        
        if self._should_log_detailed(epoch):
            print_debug(log_msg)

    def _add_training_metrics_to_log(self, history: Dict[str, List[float]], 
                                   log_msg_parts: List[str]) -> None:
        """Añade métricas de entrenamiento al mensaje de log."""
        actor_l_train = history[CONST_ACTOR_LOSS][-1] if history[CONST_ACTOR_LOSS] else float('nan')
        critic_l_train = history[CONST_CRITIC_LOSS][-1] if history[CONST_CRITIC_LOSS] else float('nan')
        avg_reward_train = history[CONST_AVERAGE_REWARD][-1] if history[CONST_AVERAGE_REWARD] else float('nan')

        if self._is_valid_metric(actor_l_train):
            log_msg_parts.append(f"ActorL: {actor_l_train:.4f}")
        if self._is_valid_metric(critic_l_train):
            log_msg_parts.append(f"CriticL: {critic_l_train:.4f}")
        if self._is_valid_metric(avg_reward_train):
            log_msg_parts.append(f"RecompProm: {avg_reward_train:.2f}")

    def _add_validation_metrics_to_log(self, val_metrics: Optional[Dict[str, float]], 
                                     log_msg_parts: List[str]) -> None:
        """Añade métricas de validación al mensaje de log."""
        if not val_metrics:
            return
            
        avg_reward_val = val_metrics.get(CONST_AVERAGE_REWARD)
        val_loss_metric = val_metrics.get(CONST_VAL_LOSS)
        action_mse_val = val_metrics.get('action_mse_val')

        if self._is_valid_metric(avg_reward_val):
            log_msg_parts.append(f"RecompVal: {avg_reward_val:.2f}")
        if self._is_valid_loss_metric(val_loss_metric):
            log_msg_parts.append(f"ValPerd: {val_loss_metric:.4f}")
        if self._is_valid_loss_metric(action_mse_val):
            log_msg_parts.append(f"ValMSEAcc: {action_mse_val:.4f}")

    def _is_valid_metric(self, metric: Optional[float]) -> bool:
        """Verifica si una métrica es válida para mostrar."""
        return metric is not None and not np.isnan(metric) and abs(metric) > CONST_EPSILON

    def _is_valid_loss_metric(self, metric: Optional[float]) -> bool:
        """Verifica si una métrica de pérdida es válida para mostrar."""
        return metric is not None and not np.isnan(metric)

    def _should_log_detailed(self, epoch: int) -> bool:
        """Determina si se debe hacer log detallado en esta época."""
        return hasattr(self, 'verbose') and getattr(self, 'verbose', 0) > 1 and epoch % 10 == 0


    def _check_early_stopping_drl(self, current_val_metric: float) -> bool:
        """
        Comprueba early stopping para DRL. 
        `current_val_metric` debe ser una métrica donde menor es mejor (ej: pérdida, -recompensa).
        """
        if not self.early_stopping: return False
        if np.isnan(current_val_metric): # No se puede tomar decisión con NaN
            print_warning("Métrica de validación para early stopping es NaN. No se aplicará early stopping en esta época.")
            return False

        es_config = self.early_stopping
        # current_val_metric es la que se quiere minimizar (ej: val_loss, o -avg_reward_val)
        if current_val_metric < es_config['best_loss'] - es_config['min_delta']:
            es_config['best_loss'] = current_val_metric
            es_config['wait'] = 0
            if es_config['restore_best_weights'] and self.model and isinstance(self.model, nn.Module):
                try:
                    # Guardar parámetros en CPU para evitar problemas de memoria GPU si hay muchos modelos
                    es_config['best_params'] = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                except Exception as e:
                    print_warning(f"No se pudieron guardar los mejores pesos para DRL (early stopping): {e}")
                    es_config['best_params'] = None # No se pudieron guardar los pesos
        else:
            es_config['wait'] += 1
        
        return es_config['wait'] >= es_config['patience']

    def _restore_best_weights_drl(self, verbose: int) -> None:
        """Restaura los mejores pesos para DRL si early stopping los guardó."""
        if not self.early_stopping: return
        
        es_config = self.early_stopping
        if es_config['restore_best_weights'] and es_config['best_params'] is not None and \
           self.model and isinstance(self.model, nn.Module):
            if verbose > 0:
                # Asegurar que best_loss no es inf antes de formatear
                best_loss_display = es_config['best_loss'] if es_config['best_loss'] != float('inf') else float('nan')
                print_info(MSG_RESTORE_BEST_WEIGHTS_DRL.format(best_loss_display))
            
            # Cargar los parámetros guardados (que están en CPU) al dispositivo del modelo
            # Es importante que las claves coincidan perfectamente.
            try:
                # Crear un nuevo state_dict en el dispositivo correcto antes de cargar
                device_state_dict = {k: v.to(self.device) for k,v in es_config['best_params'].items()}
                self.model.load_state_dict(device_state_dict)
                # self.model = self.model.to(self.device) # Asegurar que el modelo está en el dispositivo (ya debería estarlo)
            except Exception as e:
                print_critical(f"Error crítico al restaurar los mejores pesos del modelo DRL: {e}")
                print_warning("El modelo podría no estar en su mejor estado.")
        elif es_config['restore_best_weights'] and es_config['best_params'] is None and verbose > 0:
            print_warning("Early stopping estaba configurado para restaurar pesos, pero no se guardaron mejores pesos (posiblemente debido a errores o no mejora).")
