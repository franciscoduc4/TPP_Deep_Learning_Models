import numpy as np
import torch # No longer strictly needed if only storing numpy, but kept for type hints if any remain
from typing import Optional, Tuple, Any, List
from config.models_config import BUFFER_CONFIG # For default action_dim if not provided
from constants.constants import CONST_DEFAULT_SEED
from custom.printer import print_critical, print_debug, print_error, print_warning

class ReplayBuffer:
    """
    Buffer de experiencia para almacenar y muestrear transiciones (s, a, r, s', done).
    Adaptado para almacenar un vector de estado unificado.
    
    Parámetros:
    -----------
    state_dim : int
        Dimensión del vector de estado unificado.
    action_dim : int
        Dimensión del vector de acción.
    max_size : int
        Capacidad máxima del buffer.
    seed : int, opcional
        Semilla para el generador de números aleatorios (default: CONST_DEFAULT_SEED).
    """
    
    def __init__(self, state_dim: int, action_dim: int, max_size: int, seed: int = CONST_DEFAULT_SEED) -> None:
        self.state_dim: int = state_dim # Dimensión del ESTADO COMPLETO (CGM+Other+Context)
        self.action_dim: int = action_dim
        self.state_dim: int = state_dim 
        self.capacity: int = max_size
        self.rng: np.random.Generator = np.random.default_rng(seed)
        
        # El buffer almacenará tuplas de (estado_completo, acción, recompensa, siguiente_estado_completo, done)
        # donde estado_completo y siguiente_estado_completo son np.ndarray 1D.
        self.buffer: List[Optional[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]]] = [None] * max_size
        self.position: int = 0
        self.current_size: int = 0 # Para saber cuántos elementos reales hay
        
        self.current_episode_step: int = 0 
        self.max_episode_steps: int = 0
    
    def __len__(self) -> int:
        return self.current_size
        
    def _validate_input_types(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        if not isinstance(state, np.ndarray): print_error(f"El estado debe ser np.ndarray, se obtuvo {type(state)}")
        if not isinstance(action, np.ndarray): print_error(f"La acción debe ser np.ndarray, se obtuvo {type(action)}")
        if not isinstance(reward, (float, int)): print_error(f"La recompensa debe ser float, se obtuvo {type(reward)}")
        if not isinstance(next_state, np.ndarray): print_error(f"El siguiente estado debe ser np.ndarray, se obtuvo {type(next_state)}")
        if not isinstance(done, bool): print_error(f"Done debe ser bool, se obtuvo {type(done)}")

        if np.isnan(state).any() or np.isinf(state).any():
            print_error("Estado contiene NaN o Inf en ReplayBuffer.add")
        if np.isnan(action).any() or np.isinf(action).any():
            print_error("Acción contiene NaN o Inf en ReplayBuffer.add")
        if np.isnan(reward) or np.isinf(reward):
            print_error("Recompensa es NaN o Inf en ReplayBuffer.add")
        if np.isnan(next_state).any() or np.isinf(next_state).any():
            print_error("Siguiente estado contiene NaN o Inf en ReplayBuffer.add")

    def _validate_dimensions(self, state_np: np.ndarray, action_np: np.ndarray, next_state_np: np.ndarray) -> None:
        """
        Valida las dimensiones de los datos.
        
        Parámetros:
        -----------
        state_np : np.ndarray
            Estado (vector unificado).
        action_np : np.ndarray
            Acción.
        next_state_np : np.ndarray
            Siguiente estado (vector unificado).
        """
        if state_np.shape != (self.state_dim,):
            print_error(f"Dimensión de estado incorrecta. Esperado: {(self.state_dim,)}, Obtenido: {state_np.shape}")
        if next_state_np.shape != (self.state_dim,):
            print_error(f"Dimensión de siguiente estado incorrecta. Esperado: {(self.state_dim,)}, Obtenido: {next_state_np.shape}")
        
        # Validar acción
        expected_action_shape = (self.action_dim,)
        if self.action_dim == 1 and action_np.shape == (): # Permite acción escalar si action_dim es 1
            pass
        elif action_np.shape != expected_action_shape:
            print_error(f"Dimensión de acción incorrecta. Esperado: {expected_action_shape}, Obtenido: {action_np.shape}")
    
    def add(self, state: np.ndarray, action: np.ndarray, 
            reward: float, next_state: np.ndarray, 
            done: bool) -> None:
        """
        Agrega una transición al buffer.
        'state' y 'next_state' deben ser los vectores de estado completos y aplanados.
        """
        # Asegurar que la acción sea un array numpy antes de la validación de tipos
        action_np_for_validation = np.atleast_1d(action)
        self._validate_input_types(state, action_np_for_validation, reward, next_state, done)
        
        # Asegurar que la acción sea un array 1D si es escalar para el almacenamiento
        action_to_store = np.atleast_1d(action).astype(np.float32)
        self._validate_dimensions(state, action_to_store, next_state)
            
        self.buffer[self.position] = (state.astype(np.float32), 
                                      action_to_store, 
                                      float(reward), 
                                      next_state.astype(np.float32), 
                                      bool(done))
        self.position = (self.position + 1) % self.capacity
        if self.current_size < self.capacity:
            self.current_size += 1
        
    def push(self, state: np.ndarray, action: np.ndarray, 
         reward: float, next_state: np.ndarray, 
         done: bool) -> None:
        """
        Añade una transición al buffer (alias para add).
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado actual (vector unificado).
        action : np.ndarray
            Acción tomada.
        reward : float
            Recompensa recibida.
        next_state : np.ndarray
            Estado siguiente (vector unificado).
        done : bool
            Indicador de fin de episodio.
            
        Retorna:
        --------
        None
        """
        self.add(state, action, reward, next_state, done)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Muestrea un batch de transiciones del buffer.
        
        Parámetros:
        -----------
        batch_size : int
            Tamaño del batch a muestrear.
            
        Retorna:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Batch de transiciones (estados, acciones, recompensas, siguientes_estados, dones).
            Todos son arrays de NumPy.
        """
        if self.current_size == 0:
            raise ValueError("No hay elementos en el buffer para muestrear.")
        
        effective_batch_size = min(self.current_size, batch_size)
        if effective_batch_size < batch_size:
            print_warning(f"Ajustando batch_size de {batch_size} a {effective_batch_size} (elementos disponibles).")
        
        indices = self.rng.choice(self.current_size, effective_batch_size, replace=(self.current_size < batch_size))
        
        # batch_transitions = [self.buffer[i] for i in indices]
        # Esto puede fallar si self.buffer[i] es None y no se ha llenado completamente
        batch_transitions = []
        for i in indices:
            transition = self.buffer[i]
            if transition is None:
                # Esto no debería ocurrir si current_size se gestiona correctamente
                print_error(f"Error: Transición None encontrada en el buffer en el índice {i} durante el muestreo.")
                continue
            batch_transitions.append(transition)

        if not batch_transitions:
             raise ValueError("No se pudieron obtener transiciones válidas del buffer.")

        # print_critical(f"ReplayBuffer SAMPLE: first transition's next_state shape: {batch_transitions[0][3].shape}, dtype: {batch_transitions[0][3].dtype}")
        
        states, actions, rewards, next_states, dones = zip(*batch_transitions)
        
        states_np = np.array(states, dtype=np.float32)
        actions_np = np.array(actions, dtype=np.float32)
        
        # Asegurar que actions_np tenga la forma (batch_size, action_dim)
        if actions_np.ndim == 1 and self.action_dim > 0 : 
             actions_np = actions_np.reshape(-1, self.action_dim)
        elif actions_np.ndim == 0 and self.action_dim == 1: 
             actions_np = actions_np.reshape(-1,1)
        elif actions_np.ndim == 2 and actions_np.shape[1] != self.action_dim:
            # Esto podría ocurrir si las acciones se apilaron incorrectamente
            print_warning(f"Shape de acción inesperado después de apilar: {actions_np.shape}, se esperaba (?, {self.action_dim})")


        rewards_np = np.array(rewards, dtype=np.float32).reshape(-1, 1)
        next_states_np = np.array(next_states, dtype=np.float32)
        dones_np = np.array(dones, dtype=bool).reshape(-1, 1)
        
        return states_np, actions_np, rewards_np, next_states_np, dones_np
