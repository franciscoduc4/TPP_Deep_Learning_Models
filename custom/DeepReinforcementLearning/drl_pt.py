import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
from tqdm.auto import tqdm
import polars as pl
from config.models_config import BUFFER_CONFIG, EARLY_STOPPING_POLICY
from constants.constants import (
    CONST_ACTOR_LOSS, CONST_AVERAGE_REWARD, CONST_CRITIC_LOSS, CONST_DEFAULT_BATCH_SIZE, CONST_DEFAULT_EPOCHS, CONST_DEFAULT_SEED, CONST_LOSS, CONST_MODEL_INIT_ERROR, CONTEXT_FEATURE_ORDER,
    IDEAL_LOWER_BOUND, IDEAL_UPPER_BOUND, SEVERE_HYPOGLYCEMIA_THRESHOLD, HYPOGLYCEMIA_THRESHOLD, 
    HYPERGLYCEMIA_THRESHOLD, SEVERE_HYPER_PENALTY, HYPO_PENALTY_BASE, 
    HYPER_PENALTY_BASE, MAX_REWARD, SUBJECT_ID_COL
)
from custom.model_wrapper import ModelWrapper
from custom.printer import print_critical, print_debug, print_error, print_info, print_success, print_warning
from models.utils.replay_buffer import ReplayBuffer
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
    cgm_input_dim : Optional[Tuple[int, ...]], opcional
        Dimensiones de la entrada CGM (ej: (timesteps, num_cgm_features)).
        Requerido si se pasa una clase de modelo.
    other_input_dim : Optional[Tuple[int, ...]], opcional
        Dimensiones de otras características (ej: (num_other_features,)).
        Requerido si se pasa una clase de modelo.
    context_dim : Optional[int], opcional
        Dimensión de las características de contexto.
        Requerido si se pasa una clase de modelo.
    **model_kwargs : dict
        Argumentos adicionales para el constructor del modelo.
    """
    
    def __init__(self, model_or_cls: Union[Callable[..., nn.Module], nn.Module], 
                 algorithm: str = "generic", 
                 feature_config: Optional[Dict[str, List[str]]] = None, # Para nombres de columnas
                 model_kwargs: Optional[Dict[str, Any]] = None) -> None: # model_kwargs es opcional
        ModelWrapper.__init__(self, feature_config=feature_config) # Pasa feature_config al padre
        nn.Module.__init__(self)

        self.model_cls = None
        self.model: Optional[nn.Module] = None # El modelo DRL subyacente (DDPG, TD3BC, etc.)
        self.algorithm = algorithm
        # Guardar una copia de los kwargs originales destinados al constructor del modelo subyacente
        self.underlying_model_constructor_kwargs = model_kwargs.copy() if model_kwargs else {}
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Asegurar que feature_config esté inicializado (ya lo hace ModelWrapper.__init__)
        # self.feature_config ya está disponible desde ModelWrapper
        self.cgm_cols = self.feature_config.get('cgm_features', [])
        self.other_cols = self.feature_config.get('other_features', [])
        self.context_cols = self.feature_config.get('explicit_context_features', CONTEXT_FEATURE_ORDER) 
        self.target_col = self.feature_config.get('target_variable', 'bolus_log1p') 
        self.reward_col = self.feature_config.get('reward_variable', 'reward')

        # Dimensiones para el estado (aplanado) y acción
        self.state_dim = len(self.cgm_cols) + len(self.other_cols) + len(self.context_cols)
        self.action_dim = self.underlying_model_constructor_kwargs.get('action_dim', BUFFER_CONFIG.get('action_dim', 1))
        
        # Dimensiones específicas para modelos que las necesiten (como TD3BC)
        self.cgm_input_dim_tuple = (len(self.cgm_cols), 1) if self.cgm_cols else (0,0)
        self.other_input_dim_tuple = (len(self.other_cols),) if self.other_cols else (0,)

        self.prepared_model_constructor_args: Dict[str, Any] = {}

        if callable(model_or_cls) and isinstance(model_or_cls, type) and issubclass(model_or_cls, nn.Module):
            self.model_cls = model_or_cls
            # Preparar los argumentos para el constructor del modelo subyacente
            self.prepared_model_constructor_args = self.underlying_model_constructor_kwargs.copy()

            if self.algorithm == "DDPG":
                self.prepared_model_constructor_args['state_dim'] = self.state_dim
            elif self.algorithm == "TD3+BC":
                self.prepared_model_constructor_args['cgm_input_dim'] = self.cgm_input_dim_tuple
                self.prepared_model_constructor_args['other_input_dim'] = (len(self.other_cols) + len(self.context_cols),)
            # Añadir más casos para otros algoritmos si es necesario
            
        elif isinstance(model_or_cls, nn.Module):
            self.model = model_or_cls.to(self.device)
            # Si se pasa una instancia, se asume que ya está configurada correctamente.
            # Se puede intentar inferir state_dim si es relevante y no se calculó antes.
            if hasattr(self.model, 'state_dim') and isinstance(getattr(self.model, 'state_dim'), int):
                 self.state_dim = getattr(self.model, 'state_dim')
            # Similar para action_dim, cgm_input_dim_tuple, other_input_dim_tuple si es necesario
        else:
            raise ValueError("model_or_cls debe ser una clase de modelo nn.Module o una instancia de nn.Module.")
        
        self._instantiate_model_if_needed() # Asegurar que el modelo se instancia si se pasó una clase
        
        self.optimizer = None 
        buffer_size = self.underlying_model_constructor_kwargs.get('buffer_size', BUFFER_CONFIG['buffer_size'])
        # El ReplayBuffer usa state_dim (dimensión del estado aplanado) y action_dim
        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, buffer_size, seed=CONST_DEFAULT_SEED)
        self.history: Dict[str, List[float]] = {CONST_LOSS: [], CONST_AVERAGE_REWARD: []}
        self.early_stopping_config = None

    def _instantiate_model_if_needed(self) -> None:
         if self.model is None and self.model_cls is not None:
             try:
                 self.model = self.model_cls(**self.prepared_model_constructor_args)
                 self.model.to(self.device)
             except TypeError as e:
                 print_error(f"Error al instanciar el modelo {self.model_cls.__name__}: {e}")
                 print_debug(f"Argumentos proporcionados: {self.prepared_model_constructor_args}")
                 raise
             except Exception as e:
                 print_error(f"Error inesperado al instanciar el modelo {self.model_cls.__name__}: {e}")
                 raise

    def _build_state_from_row(self, row: pl.Series) -> Optional[np.ndarray]:
        """Construye un vector de estado aplanado desde una fila de DataFrame (Polars Series)."""
        try:
            # Convertir la Polars Series a un diccionario para facilitar la extracción
            row_dict = row.to_dicts()[0] # asumiendo que la Series viene de una fila de DataFrame (slice(0,1).to_series()) o similar
                                       # Si row ya es un dict, no es necesario to_dicts()

            cgm_values = [row_dict.get(col, 0.0) for col in self.cgm_cols]
            other_values = [row_dict.get(col, 0.0) for col in self.other_cols]
            context_values = [row_dict.get(col, 0.0) for col in self.context_cols] # Usar CONTEXT_FEATURE_ORDER
            
            # Aplanar y concatenar
            state_list = cgm_values + other_values + context_values
            return np.array(state_list, dtype=np.float32)
        except Exception as e:
            print_error(f"Error construyendo estado desde la fila: {e}. Fila: {row.to_dicts()}")
            # print_debug(f"Columnas CGM esperadas: {self.cgm_cols}")
            # print_debug(f"Columnas Other esperadas: {self.other_cols}")
            # print_debug(f"Columnas Context esperadas: {self.context_cols}")
            return None

    def start(self, train_df: pl.DataFrame, rng_key: Optional[Any] = None) -> Any:
        """
        Inicializa el modelo DRL, incluyendo dimensiones y buffer si es necesario.
        
        Parámetros:
        -----------
        train_df : pl.DataFrame
            DataFrame de entrenamiento para inferir dimensiones si es necesario.
        rng_key : Optional[Any]
            Semilla o clave aleatoria para reproducibilidad (para PyTorch será un int).
            
        Retorna:
        --------
        Any
            Clave o estado inicializado (para compatibilidad con interfaz).
        """
        try:
            # Configurar semilla para reproducibilidad
            if rng_key is not None:
                torch.manual_seed(rng_key)
                np.random.seed(rng_key)
            
            # Verificar que el DataFrame no esté vacío
            if train_df.is_empty():
                print_warning("El DataFrame de entrenamiento está vacío. No se puede inicializar el modelo.")
                return None
                
            # Instanciar el modelo si aún no está instanciado
            self._instantiate_model_if_needed()
            
            if self.model is None:
                print_error("No se pudo instanciar el modelo DRL.")
                return None
                
            # Verificar que las columnas necesarias existan en el DataFrame
            missing_cols = []
            all_required_cols = self.cgm_cols + self.other_cols + self.context_cols + [self.target_col]
            
            df_columns = train_df.columns
            for col in all_required_cols:
                if col not in df_columns:
                    missing_cols.append(col)
                    
            if missing_cols:
                print_error(f"Faltan las siguientes columnas en train_df: {', '.join(missing_cols)}")
                return None
                
            # Configurar el dispositivo del modelo
            self.model.to(self.device)
            print_info(f"Modelo DRL {self.algorithm} movido a dispositivo: {self.device}")
            
            # Inicializar el replay buffer con las dimensiones correctas
            if hasattr(self, 'replay_buffer') and self.replay_buffer is not None:
                print_info(f"Replay buffer ya inicializado con capacidad {self.replay_buffer.capacity}.")
            else:
                buffer_size = self.underlying_model_constructor_kwargs.get('buffer_size', BUFFER_CONFIG['buffer_size'])
                self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, buffer_size, seed=CONST_DEFAULT_SEED)
                print_info(f"Replay buffer inicializado: state_dim={self.state_dim}, action_dim={self.action_dim}, capacity={buffer_size}")
                
            # Configurar early stopping si está definido en la configuración
            if EARLY_STOPPING_POLICY.get('enabled', False):
                self.early_stopping_config = EARLY_STOPPING_POLICY
                print_info(MSG_EARLY_STOPPING_CONFIGURED.format(self.early_stopping_config.get('patience', 'N/A')))
                
            # Inicializar historial de entrenamiento
            self.history = {
                CONST_LOSS: [],
                CONST_AVERAGE_REWARD: [],
                CONST_ACTOR_LOSS: [],
                CONST_CRITIC_LOSS: []
            }
            
            # Validar que el modelo tiene los métodos necesarios
            required_methods = ['update', 'select_action']
            missing_methods = []
            for method in required_methods:
                if not hasattr(self.model, method) or not callable(getattr(self.model, method)):
                    missing_methods.append(method)
                    
            if missing_methods:
                print_error(f"El modelo DRL {self.algorithm} no implementa los métodos requeridos: {', '.join(missing_methods)}")
                return None
                
            # Imprimir información del modelo inicializado
            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print_info(f"Modelo DRL {self.algorithm} inicializado con {num_params:,} parámetros entrenables.")
            print_info(f"Dimensiones: state_dim={self.state_dim}, action_dim={self.action_dim}")
            
            print_success(f"Modelo DRL {self.algorithm} listo para entrenamiento.")
            
            # Retornar algo para compatibilidad (podría ser útil para tracking)
            return {
                'model_ready': True,
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'device': str(self.device),
                'num_parameters': num_params
            }
            
        except Exception as e:
            print_error(f"Error en start() del modelo DRL {self.algorithm}: {e}")
            return None
    
    def _build_state_from_row_dict(self, row_dict: Dict[str, Any]) -> Optional[np.ndarray]:
        """Construye un vector de estado aplanado desde un diccionario de fila."""
        try:
            cgm_values = [row_dict.get(col, 0.0) for col in self.cgm_cols]
            other_values = [row_dict.get(col, 0.0) for col in self.other_cols]
            context_values = [row_dict.get(col, 0.0) for col in self.context_cols] # Usar CONTEXT_FEATURE_ORDER
            
            state_list = cgm_values + other_values + context_values
            return np.array(state_list, dtype=np.float32)
        except Exception as e:
            print_error(f"Error construyendo estado desde dict de fila: {e}. Fila: {row_dict}")
            return None

    def fit(self, train_df: pl.DataFrame,
            val_df: Optional[pl.DataFrame] = None,
            epochs: int = CONST_DEFAULT_EPOCHS, 
            batch_size: int = CONST_DEFAULT_BATCH_SIZE, 
            verbose: int = 1) -> Dict[str, List[float]]:
        self._instantiate_model_if_needed()
        if self.model is None:
            print_error("Modelo DRL no instanciado. No se puede entrenar.")
            return {}

        if not hasattr(self.model, 'parameters') or not callable(self.model.parameters):
             print_error(MSG_INIT_OPTIMIZER_FAIL)
             return {}
             
        # Inicializar optimizador si no existe (para actor y crítico dentro del modelo DRL)
        # El modelo DRL (ej. DDPG) es responsable de sus propios optimizadores.
        # Esta sección podría eliminarse si el modelo DDPG/TD3BC maneja sus optimizadores internamente.
        # if self.optimizer is None and hasattr(self.model, 'parameters') and callable(self.model.parameters):
        #     try:
        #         params_to_optimize = list(self.model.parameters())
        #         if params_to_optimize:
        #             lr = self.underlying_model_constructor_kwargs.get('learning_rate', self.underlying_model_constructor_kwargs.get('actor_lr', 1e-3)) # Tomar LR específico si existe
        #             self.optimizer = optim.Adam(params_to_optimize, lr=lr)
        #             if verbose > 0: print_info(MSG_INIT_OPTIMIZER_SUCCESS.format("Adam", lr))
        #         else:
        #             if verbose > 0: print_warning(MSG_INIT_OPTIMIZER_NO_PARAMS.format(self.algorithm))
        #     except Exception as e:
        #         print_error(f"Error al inicializar el optimizador: {e}")

        if train_df.is_empty():
            print_error("DataFrame de entrenamiento vacío.")
            return {}
        
        # Popular el Replay Buffer
        # Asumimos que train_df tiene columnas: [features_estado_actual...], 'action', 'reward', [features_siguiente_estado...]
        # Necesitamos construir 'state', 'action', 'reward', 'next_state', 'done'
        print_info(f"Populando Replay Buffer con {len(train_df)} transiciones...")
        for i in tqdm(range(len(train_df) - 1), desc="Populando Replay Buffer"): # -1 para tener siempre un next_state
            current_row_series = train_df.row(i, named=True) # Devuelve un diccionario
            next_row_series = train_df.row(i + 1, named=True)

            # Convertir dict a Polars Series temporalmente para _build_state_from_row si es necesario
            # o adaptar _build_state_from_row para tomar dict.
            # Para simplificar, asumimos que _build_state_from_row puede manejar un dict.
            
            state = self._build_state_from_row_dict(current_row_series)
            next_state = self._build_state_from_row_dict(next_row_series)

            if state is None or next_state is None:
                print_warning(f"Saltando transición en el índice {i} debido a un error al construir el estado.")
                continue # Saltar esta transición si el estado no se pudo construir

            action = np.array([current_row_series.get(self.target_col, 0.0)], dtype=np.float32) # Acción tomada
            reward = float(current_row_series.get(self.reward_col, 0.0)) # Recompensa obtenida
            
            # 'done' es más complejo. Si el siguiente estado pertenece a un sujeto diferente o es el final de un episodio.
            # Simplificación: no 'done' a menos que sea la última transición del dataset.
            done = (i == len(train_df) - 2) 
            if current_row_series.get(SUBJECT_ID_COL) != next_row_series.get(SUBJECT_ID_COL):
                done = True

            self.replay_buffer.add(state, action, reward, next_state, done)
        
        if len(self.replay_buffer) < batch_size:
            print_warning(f"Replay buffer tiene menos muestras ({len(self.replay_buffer)}) que batch_size ({batch_size}). El entrenamiento podría no ser efectivo.")
            # return self.history # O continuar si el modelo puede manejarlo

        # Bucle de entrenamiento DRL
        if not hasattr(self.model, 'update') or not callable(getattr(self.model, 'update')):
            print_error(f"El modelo DRL {self.algorithm} no tiene un método 'update'. No se puede entrenar.")
            return self.history

        for epoch in range(epochs):
            epoch_losses = []
            epoch_rewards = [] # Si el método update devuelve recompensas o si se calculan
            
            # El número de pasos de actualización por época puede variar.
            # Por ejemplo, puede ser len(train_df) // batch_size o un número fijo.
            num_updates_per_epoch = max(1, len(self.replay_buffer) // batch_size) if len(self.replay_buffer) > 0 else 1
            
            for _ in tqdm(range(num_updates_per_epoch), desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                if len(self.replay_buffer) < batch_size:
                    print_warning(f"No hay suficientes muestras en el buffer ({len(self.replay_buffer)}) para el tamaño de batch ({batch_size}). Saltando actualización.")
                    continue

                states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
                
                # Convertir a tensores y mover al dispositivo
                states_tensor = torch.FloatTensor(states).to(self.device)
                actions_tensor = torch.FloatTensor(actions).to(self.device)
                rewards_tensor = torch.FloatTensor(rewards).to(self.device)
                next_states_tensor = torch.FloatTensor(next_states).to(self.device)
                dones_tensor = torch.FloatTensor(dones.astype(np.float32)).to(self.device) # Convertir bool a float para PyTorch

                # Llamar al método update del modelo DRL subyacente
                # La firma de update puede variar, así que adaptamos según el algoritmo
                if self.algorithm == "DDPG":
                    loss_dict = self.model.update((states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor))
                elif self.algorithm == "TD3+BC":
                    # TD3BC espera x_cgm, x_other, etc. por separado. Necesitamos reconstruirlos desde 'states_tensor'.
                    # Esto asume que _build_state_from_row_dict concatena CGM, luego Other, luego Context.
                    cgm_len = len(self.cgm_cols)
                    other_len = len(self.other_cols)
                    # context_len = len(self.context_cols) # No se usa directamente aquí, ya que está en x_other_full

                    x_cgm_batch = states_tensor[:, :cgm_len]
                    x_other_batch = states_tensor[:, cgm_len:] # Esto incluye 'other' y 'context'
                    
                    next_x_cgm_batch = next_states_tensor[:, :cgm_len]
                    next_x_other_batch = next_states_tensor[:, cgm_len:]

                    # Reshape CGM si es necesario (ej. para CNNs)
                    if self.cgm_input_dim_tuple != (0,0) and len(self.cgm_input_dim_tuple) == 2:
                        x_cgm_batch = x_cgm_batch.view(-1, self.cgm_input_dim_tuple[0], self.cgm_input_dim_tuple[1])
                        next_x_cgm_batch = next_x_cgm_batch.view(-1, self.cgm_input_dim_tuple[0], self.cgm_input_dim_tuple[1])
                    
                    loss_dict = self.model.update((x_cgm_batch, x_other_batch, actions_tensor, rewards_tensor, next_x_cgm_batch, next_x_other_batch, dones_tensor))
                else:
                    # Para otros algoritmos, asumimos que aceptan el estado aplanado directamente
                    loss_dict = self.model.update((states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor))

                if loss_dict:
                    if CONST_ACTOR_LOSS in loss_dict:
                        self.history[CONST_ACTOR_LOSS].append(loss_dict[CONST_ACTOR_LOSS])
                    if CONST_CRITIC_LOSS in loss_dict:
                        self.history[CONST_CRITIC_LOSS].append(loss_dict[CONST_CRITIC_LOSS])
                    # Podrías tener una métrica de recompensa o Q-valor devuelta por update
                    if 'average_reward' in loss_dict:
                        epoch_rewards.append(loss_dict['average_reward'])
                    elif 'q_value' in loss_dict: # O si devuelve Q-valor promedio
                        epoch_rewards.append(loss_dict['q_value'])

            avg_epoch_loss = np.mean(self.history[CONST_CRITIC_LOSS][-num_updates_per_epoch:]) if self.history[CONST_CRITIC_LOSS] else np.nan
            avg_epoch_reward = np.mean(epoch_rewards) if epoch_rewards else np.nan
            
            self.history[CONST_LOSS].append(avg_epoch_loss) # Usar critic loss como la pérdida principal para early stopping
            self.history[CONST_AVERAGE_REWARD].append(avg_epoch_reward)
            
            if verbose > 0:
                log_msg = f"Epoch {epoch+1}/{epochs} - Loss: {avg_epoch_loss:.4f}"
                if not np.isnan(avg_epoch_reward):
                    log_msg += f", Avg Reward/Q-value: {avg_epoch_reward:.4f}"
                print_info(log_msg)

            # Early stopping (si configurado y val_df proporcionado)
            if self.early_stopping_config and val_df is not None and not val_df.is_empty():
                # Aquí necesitarías una forma de evaluar el modelo en val_df.
                # Esto podría ser ejecutar la política en un entorno simulado o calcular una métrica offline.
                # Por simplicidad, vamos a asumir que `evaluate_performance` existe y devuelve una métrica.
                if hasattr(self.model, 'evaluate_performance'):
                    val_performance = self.model.evaluate_performance(val_df) # Esto necesita ser implementado en DDPG/TD3BC
                    self.history.setdefault(CONST_VAL_LOSS, []).append(val_performance) # Asumiendo que devuelve una 'pérdida' o métrica a minimizar
                    
                    if self.early_stopping.step(val_performance):
                        print_info(MSG_EARLY_STOPPING_ACTIVATED.format(epoch + 1))
                        if self.early_stopping_config.get('restore_best_weights', False):
                            print_info(MSG_RESTORE_BEST_WEIGHTS_DRL.format(self.early_stopping.best_score))
                            self.model.load_state_dict(self.early_stopping.best_model_state_dict)
                        break 
                else:
                    if epoch == 0: # Solo advertir una vez
                        print_warning(MSG_NO_EVAL_PERFORMANCE)


        if verbose > 0: print_success(MSG_TRAINING_COMPLETE.format(self.algorithm))
        return self.history

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        self._instantiate_model_if_needed()
        if self.model is None:
            print_error(f"Modelo DRL {self.algorithm} no instanciado. No se puede predecir.")
            return np.array([])
        
        if not (hasattr(self.model, 'select_action') or hasattr(self.model, 'forward')):
            print_error(f"Modelo DRL {self.algorithm} no tiene 'select_action' ni 'forward'.")
            return np.array([])
        
        predictions = []
        if df.is_empty(): return np.array([])

        for i in tqdm(range(len(df)), desc="Prediciendo", leave=False):
            row_dict = df.row(i, named=True)
            
            if self.algorithm == "TD3+BC":
                # Para TD3BC, necesitamos separar CGM y otras características
                cgm_data = np.array([row_dict.get(col, 0.0) for col in self.cgm_cols], dtype=np.float32)
                other_data_list = [row_dict.get(col, 0.0) for col in self.other_cols]
                context_data_list = [row_dict.get(col, 0.0) for col in self.context_cols]
                other_full_data = np.array(other_data_list + context_data_list, dtype=np.float32)

                # Asegurar que las dimensiones sean correctas para el modelo TD3BC
                # cgm_input_dim es (timesteps, features_per_timestep)
                # other_input_dim es (total_other_features_len,)
                if self.cgm_input_dim_tuple != (0,0):
                    x_cgm_tensor = torch.FloatTensor(cgm_data).reshape(1, *self.cgm_input_dim_tuple).to(self.device)
                else:
                    x_cgm_tensor = torch.empty(1, 0).to(self.device) # O manejar de otra forma si no hay CGM

                if self.other_input_dim_tuple != (0,):
                    x_other_tensor = torch.FloatTensor(other_full_data).unsqueeze(0).to(self.device)
                else:
                    x_other_tensor = torch.empty(1, 0).to(self.device) # O manejar de otra forma

                with torch.no_grad():
                    action_tensor = self.model.select_action(x_cgm_tensor, x_other_tensor, add_noise=False)

            else: # Para DDPG y otros modelos que esperan un estado aplanado
                state_np = self._build_state_from_row_dict(row_dict)
                if state_np is None:
                    predictions.append(np.nan) # O alguna otra forma de manejar el error
                    continue
                state_tensor = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    if hasattr(self.model, 'select_action'):
                        action_tensor = self.model.select_action(state_tensor, add_noise=False) # Asumiendo que select_action puede tomar un tensor de estado
                    elif hasattr(self.model, 'forward'):
                        action_tensor = self.model(state_tensor)
                    else:
                        # Esto no debería ocurrir si la verificación al inicio de la función se hizo correctamente
                        predictions.append(np.nan)
                        continue
            
            predictions.append(action_tensor.cpu().numpy().flatten()[0])
        return np.array(predictions)

    def predict_with_context(self, 
                             x_cgm: np.ndarray, 
                             x_other: np.ndarray, 
                             current_glucose: float, 
                             carb_intake: float, 
                             iob: float, 
                             sleep_quality: Optional[float] = None, 
                             work_intensity: Optional[float] = None, 
                             exercise_intensity: Optional[float] = None,
                             target_glucose: Optional[float] = None, # Añadido para consistencia
                             **kwargs: Any) -> float:
        self._instantiate_model_if_needed()
        if self.model is None:
            print_error(MSG_PREDICT_CTX_NOT_IMPLEMENTED_ERROR.format(self.algorithm))
            return 0.0 # O lanzar una excepción

        if hasattr(self.model, 'predict_with_context'):
            # El modelo subyacente tiene su propio método predict_with_context
            # Asegurarse de que los datos estén en el formato correcto (numpy arrays)
            # x_cgm y x_other ya son np.ndarray según la firma
            return self.model.predict_with_context(
                x_cgm, x_other, current_glucose, carb_intake, iob,
                sleep_quality, work_intensity, exercise_intensity, target_glucose=target_glucose, **kwargs
            )
        elif hasattr(self.model, 'select_action'):
            # Construir el estado completo como lo haría el ReplayBuffer
            context_values = [
                current_glucose, carb_intake, iob,
                sleep_quality if sleep_quality is not None else 0.0, # Usar 0 o un valor por defecto si None
                work_intensity if work_intensity is not None else 0.0,
                exercise_intensity if exercise_intensity is not None else 0.0
            ]
            # Asegurar que el orden de context_values coincida con CONTEXT_FEATURE_ORDER
            # Esto es crucial si el modelo fue entrenado con un orden específico.
            # Si CONTEXT_FEATURE_ORDER define el orden, usarlo para construir context_np.
            # Ejemplo:
            # context_map = {
            #     'glucose_last': current_glucose,
            #     'meal_carbs': carb_intake,
            #     'insulin_on_board': iob,
            #     'sleep_quality': sleep_quality if sleep_quality is not None else 0.0,
            #     'work_intensity': work_intensity if work_intensity is not None else 0.0,
            #     'exercise_intensity': exercise_intensity if exercise_intensity is not None else 0.0
            # }
            # context_values_ordered = [context_map[feature_name] for feature_name in CONTEXT_FEATURE_ORDER]
            # context_np = np.array(context_values_ordered, dtype=np.float32)

            # Para simplificar, asumimos que context_values_list está en el orden correcto
            context_np = np.array(context_values, dtype=np.float32)

            # Aplanar x_cgm y x_other si no lo están ya
            x_cgm_flat = x_cgm.flatten() if x_cgm is not None else np.array([])
            x_other_flat = x_other.flatten() if x_other is not None else np.array([])
            
            state_parts = []
            if x_cgm_flat.size > 0:
                state_parts.append(x_cgm_flat)
            if x_other_flat.size > 0:
                state_parts.append(x_other_flat)
            if context_np.size > 0:
                state_parts.append(context_np)
            
            if not state_parts:
                print_error("No se proporcionaron datos de estado para la predicción.")
                return 0.0 # O manejar de otra manera

            state_np = np.concatenate(state_parts)
            
            # Verificar que la dimensión del estado coincida con self.state_dim
            if state_np.shape[0] != self.state_dim:
                print_error(f"Dimensiones de estado inconsistentes. Esperado: {self.state_dim}, Obtenido: {state_np.shape[0]}")
                # Aquí podrías intentar depurar qué parte del estado está mal o faltante
                # print(f"CGM cols: {self.cgm_cols}, Other cols: {self.other_cols}, Context cols: {self.context_cols}")
                # print(f"x_cgm_flat: {x_cgm_flat.shape}, x_other_flat: {x_other_flat.shape}, context_np: {context_np.shape}")
                return 0.0 # O lanzar una excepción

            state_tensor = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action_tensor = self.model.select_action(state_tensor, add_noise=False) # Asumiendo que select_action toma estado y add_noise
            
            return action_tensor.item()
        else:
            print_error(MSG_PREDICT_CTX_NOT_IMPLEMENTED_UNDERLYING.format(self.algorithm))
            return 0.0 # O lanzar una excepción

    def evaluate(self, test_df: pl.DataFrame, metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evalúa el modelo DRL. Para DRL, esto podría implicar ejecutar la política en un entorno
        o usar métricas específicas de RL como el retorno promedio.
        """
        self._instantiate_model_if_needed()
        if self.model is None:
            print_error(f"Modelo {self.algorithm} no instanciado. No se puede evaluar.")
            return {}

        if test_df.is_empty():
            print_warning("DataFrame de prueba vacío. No se puede evaluar.")
            return {}

        # Si el modelo tiene un método `evaluate_performance` específico, úsalo.
        if hasattr(self.model, 'evaluate_performance') and callable(getattr(self.model, 'evaluate_performance')):
            try:
                return self.model.evaluate_performance(test_df, metrics)
            except Exception as e:
                print_error(f"Error durante la evaluación específica del modelo {self.algorithm}: {e}")
                # Fallback a una evaluación genérica si es posible, o simplemente retornar vacío.
        
        # Evaluación genérica: Calcular recompensas promedio si la columna 'reward' está presente
        # Esto es una simplificación. Una evaluación DRL adecuada a menudo requiere simulación.
        if self.reward_col in test_df.columns:
            total_reward = 0
            num_episodes = 0 # O número de pasos, dependiendo de cómo se estructuren los datos
            
            # Esto es una simplificación. En un escenario real, necesitarías simular episodios.
            # Aquí, asumimos que cada fila es un paso y queremos la recompensa promedio por paso.
            if 'reward' in test_df.columns:
                avg_reward = test_df.select(pl.col(self.reward_col).mean()).item()
                print_info(f"Evaluación genérica: Recompensa promedio en datos de prueba = {avg_reward:.4f}")
                return {"average_reward": avg_reward}
            else:
                print_warning(f"Columna de recompensa '{self.reward_col}' no encontrada en test_df. No se puede calcular la recompensa promedio.")
                return {}
        else:
            print_warning(f"No se pudo realizar la evaluación para el modelo {self.algorithm}. "
                          f"Implemente 'evaluate_performance' en el modelo o asegúrese de que '{self.reward_col}' esté en los datos.")
            return {}

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Obtiene los parámetros del modelo."""
        params = {
            'algorithm': self.algorithm,
            'feature_config': self.feature_config, # Guardar la configuración de características
            'model_kwargs': self.underlying_model_constructor_kwargs # Guardar los kwargs originales
        }
        if deep and self.model is not None and hasattr(self.model, 'get_params'):
            params.update(self.model.get_params(deep=deep))
        return params

    def set_params(self, **params: Any) -> 'DRLModelWrapperPyTorch':
        """Establece los parámetros del modelo."""
        if 'algorithm' in params:
            self.algorithm = params.pop('algorithm')
        if 'feature_config' in params:
            self.feature_config = params.pop('feature_config')
            # Re-inicializar columnas basadas en la nueva feature_config
            self.cgm_cols = self.feature_config.get('cgm_features', [])
            self.other_cols = self.feature_config.get('other_features', [])
            self.context_cols = self.feature_config.get('explicit_context_features', CONTEXT_FEATURE_ORDER)
            self.target_col = self.feature_config.get('target_variable', 'bolus_log1p')
            self.reward_col = self.feature_config.get('reward_variable', 'reward')
            self.state_dim = len(self.cgm_cols) + len(self.other_cols) + len(self.context_cols)
            self.cgm_input_dim_tuple = (len(self.cgm_cols), 1) if self.cgm_cols else (0,0)
            self.other_input_dim_tuple = (len(self.other_cols),) if self.other_cols else (0,)
            
            # Actualizar model_kwargs con las nuevas dimensiones si es necesario
            if self.algorithm == "DDPG":
                self.underlying_model_constructor_kwargs['state_dim'] = self.state_dim
            elif self.algorithm == "TD3+BC":
                self.underlying_model_constructor_kwargs['cgm_input_dim'] = self.cgm_input_dim_tuple
                self.underlying_model_constructor_kwargs['other_input_dim'] = (len(self.other_cols) + len(self.context_cols),)

        if 'model_kwargs' in params:
            self.underlying_model_constructor_kwargs.update(params.pop('model_kwargs'))

        # Re-instanciar el modelo si los parámetros relevantes cambiaron
        # Esto es un poco simplista; idealmente, se verificaría si los parámetros que afectan la arquitectura del modelo han cambiado.
        if self.model_cls:
            self.prepared_model_constructor_args = self.underlying_model_constructor_kwargs.copy()
            if self.algorithm == "DDPG":
                self.prepared_model_constructor_args['state_dim'] = self.state_dim
            elif self.algorithm == "TD3+BC":
                self.prepared_model_constructor_args['cgm_input_dim'] = self.cgm_input_dim_tuple
                self.prepared_model_constructor_args['other_input_dim'] = (len(self.other_cols) + len(self.context_cols),)
            
            self.model = self.model_cls(**self.prepared_model_constructor_args).to(self.device)
            # Re-inicializar optimizadores si es necesario
            if hasattr(self.model, 'actor_optimizer') and hasattr(self.model, 'critic_optimizer'):
                self.model.actor_optimizer = optim.Adam(self.model.actor.parameters(), lr=self.model.config.get('actor_lr', 1e-4), weight_decay=self.model.config.get('actor_weight_decay', 0.0))
                if hasattr(self.model, 'critic1') and hasattr(self.model, 'critic2'): # Para TD3
                    self.model.critic_optimizer = optim.Adam(list(self.model.critic1.parameters()) + list(self.model.critic2.parameters()), lr=self.model.config.get('critic_lr', 1e-3), weight_decay=self.model.config.get('critic_weight_decay', 0.0))
                elif hasattr(self.model, 'critic'): # Para DDPG
                    self.model.critic_optimizer = optim.Adam(self.model.critic.parameters(), lr=self.model.config.get('critic_lr', 1e-3), weight_decay=self.model.config.get('critic_weight_decay', 0.0))

        elif self.model and hasattr(self.model, 'set_params'):
            self.model.set_params(**params)
        
        return self

    def get_model_name(self) -> str:
        """Retorna el nombre del algoritmo DRL."""
        return self.algorithm

    def save(self, path: str) -> None:
        self._instantiate_model_if_needed()
        if self.model is None:
            print_error(f"Modelo {self.algorithm} no instanciado, no se puede guardar.")
            return
        try:
            # Guardar el estado del modelo DRL subyacente
            # Si el modelo DRL tiene su propio método save_state o save
            if hasattr(self.model, 'save_state') and callable(getattr(self.model, 'save_state')):
                model_state = self.model.save_state()
            else: # Guardar state_dict como fallback
                model_state = self.model.state_dict()
            
            # Guardar también los kwargs y configuración del wrapper para reconstrucción
            wrapper_state = {
                'model_cls_name': self.model_cls.__name__ if self.model_cls else None,
                'algorithm': self.algorithm,
                'model_kwargs': self.model_kwargs,
                'feature_config': self.feature_config,
                'model_state_dict': model_state # Estado del modelo DRL
            }
            torch.save(wrapper_state, path)
            print_success(MSG_SAVE_DRL_SUCCESS.format(self.algorithm, path))
        except Exception as e:
            print_error(f"Error al guardar el modelo DRL {self.algorithm}: {e}")

    def load(self, path: str) -> None:
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            self.algorithm = checkpoint.get('algorithm', self.algorithm)
            self.model_kwargs = checkpoint.get('model_kwargs', self.model_kwargs)
            self.feature_config = checkpoint.get('feature_config', self.feature_config)
            
            # Re-derivar nombres de columnas y dimensiones desde feature_config
            self.cgm_cols = self.feature_config.get('cgm_features', [])
            self.other_cols = self.feature_config.get('other_features', [])
            self.context_cols = self.feature_config.get('explicit_context_features', [])
            self.state_dim = len(self.cgm_cols) + len(self.other_cols) + len(self.context_cols)
            self.action_dim = self.model_kwargs.get('action_dim', 1)

            # Actualizar kwargs para la instanciación del modelo si es necesario
            self.model_kwargs['cgm_input_dim'] = (len(self.cgm_cols), 1) if self.cgm_cols else (0,0)
            self.model_kwargs['other_input_dim'] = (len(self.other_cols),) if self.other_cols else (0,)
            self.model_kwargs['context_dim'] = len(self.context_cols)
            self.model_kwargs['state_dim'] = self.state_dim


            model_cls_name = checkpoint.get('model_cls_name')
            if model_cls_name:
                # Necesitas una forma de obtener la clase del modelo desde su nombre
                # Esto podría implicar un mapeo o importación dinámica.
                # Ejemplo: self.model_cls = globals().get(model_cls_name)
                # Por ahora, asumimos que la clase correcta ya está en self.model_cls
                # o que se pasó al constructor del wrapper al cargar.
                if self.model_cls is None or self.model_cls.__name__ != model_cls_name:
                    print_warning(f"La clase del modelo {model_cls_name} guardada difiere o no está disponible. Intentando usar la actual.")
                
                self._instantiate_model_if_needed() # Instanciar con los kwargs actualizados
            
            if self.model is None:
                print_error(f"No se pudo instanciar el modelo {self.algorithm} durante la carga.")
                return

            model_state_dict = checkpoint.get('model_state_dict')
            if model_state_dict:
                if hasattr(self.model, 'load_state') and callable(getattr(self.model, 'load_state')):
                    self.model.load_state(model_state_dict)
                else: # Cargar state_dict como fallback
                    self.model.load_state_dict(model_state_dict)
            
            self.model.to(self.device)
            print_success(MSG_LOAD_DRL_SUCCESS.format(self.algorithm, path))
        except Exception as e:
            print_error(f"Error al cargar el modelo DRL {self.algorithm} desde {path}: {e}")
  
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Paso hacia adelante del wrapper. Para DRL, esto usualmente significa obtener una acción del actor/política.
        Este método necesita ser compatible con la firma de nn.Module.forward, pero su uso directo
        puede ser limitado si `predict_with_context` es la interfaz principal.
        
        Parámetros:
        -----------
        *args : Posicionales
            Argumentos posicionales para el modelo DRL subyacente.
        **kwargs : Clave-valor
            Argumentos clave-valor para el modelo DRL subyacente.
            
        Retorna:
        --------
        torch.Tensor
            Acción predicha.
        """
        # Este forward es para cumplir con nn.Module, pero la lógica principal de DRL está en select_action o predict_with_context del modelo DRL subyacente.
        self._instantiate_model_if_needed()
        if self.model and hasattr(self.model, 'forward'):
            return self.model.forward(*args, **kwargs)
        raise NotImplementedError("El modelo DRL subyacente no tiene un método forward o no está instanciado.")
