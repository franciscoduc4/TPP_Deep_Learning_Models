import numpy as np
import torch # No longer strictly needed if only storing numpy, but kept for type hints if any remain
from typing import Tuple, Any, List
from config.models_config import BUFFER_CONFIG # For default action_dim if not provided
from constants.constants import CONST_DEFAULT_SEED
from custom.printer import print_warning

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
        self.state_dim: int = state_dim
        self.action_dim: int = action_dim
        self.capacity: int = max_size
        self.rng: np.random.Generator = np.random.default_rng(seed)
        
        self.buffer: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = []
        self.position: int = 0
        # Atributos para compatibilidad con lógica externa si se añaden dinámicamente
        self.current_episode_step: int = 0 
        self.max_episode_steps: int = 0
    
    def __len__(self) -> int:
        """
        Obtiene el número de transiciones en el buffer.
        
        Retorna:
        --------
        int
            Número de transiciones almacenadas.
        """
        return len(self.buffer)
        
    def _validate_input_types(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Valida los tipos de los datos de entrada.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado actual.
        action : np.ndarray
            Acción tomada.
        reward : float
            Recompensa recibida.
        next_state : np.ndarray
            Estado siguiente.
        done : bool
            Indicador de fin de episodio.
        """
        if not isinstance(state, np.ndarray):
            raise ValueError(f"El estado debe ser un np.ndarray, se obtuvo {type(state)}")
        if not isinstance(action, np.ndarray):
            raise ValueError(f"La acción debe ser un np.ndarray, se obtuvo {type(action)}")
        if not isinstance(reward, (int, float)):
            raise ValueError(f"La recompensa debe ser numérica (int o float), se obtuvo {type(reward)}")
        if not isinstance(next_state, np.ndarray):
            raise ValueError(f"El siguiente estado debe ser un np.ndarray, se obtuvo {type(next_state)}")
        if not isinstance(done, bool):
            raise ValueError(f"El indicador 'done' debe ser booleano, se obtuvo {type(done)}")

    def _validate_dimensions(self, state_np: np.ndarray, action_np: np.ndarray) -> None:
        """
        Valida las dimensiones de los datos.
        
        Parámetros:
        -----------
        state_np : np.ndarray
            Estado (vector unificado).
        action_np : np.ndarray
            Acción.
        """
        if state_np.shape != (self.state_dim,):
            raise ValueError(f"La dimensión del estado {state_np.shape} no coincide con la esperada {(self.state_dim,)}")
        if action_np.shape != (self.action_dim,): # Asume que action_dim es un escalar para la forma (action_dim,)
            # Si action_dim es 1, action_np.shape podría ser (1,) o (). Ajustar si es necesario.
            # La implementación actual de DDPG.select_action devuelve np.array([action_value])
            # que tiene forma (1,) si action_dim es 1.
            if self.action_dim == 1 and action_np.shape == (): # Escalar
                 pass # Permitir acción escalar si action_dim es 1
            elif action_np.shape != (self.action_dim,):
                raise ValueError(f"La dimensión de la acción {action_np.shape} no coincide con la esperada {(self.action_dim,)}")
    
    def add(self, state: np.ndarray, action: np.ndarray, 
            reward: float, next_state: np.ndarray, 
            done: bool) -> None:
        """
        Agrega una transición al buffer.
        
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
        self._validate_input_types(state, action, reward, next_state, done)
        self._validate_dimensions(state, action) # Validar directamente las entradas NumPy
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(None) # type: ignore
            
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
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
        buffer_len = len(self.buffer)
        if buffer_len == 0:
            raise ValueError("No hay elementos en el buffer para muestrear.")
        
        effective_batch_size = min(buffer_len, batch_size)
        if effective_batch_size < batch_size:
            print_warning(f"Ajustando batch_size de {batch_size} a {effective_batch_size} (elementos disponibles).")
        
        indices = self.rng.choice(buffer_len, effective_batch_size, replace=(buffer_len < batch_size))
        
        batch_transitions = [self.buffer[i] for i in indices]
        
        states, actions, rewards, next_states, dones = zip(*batch_transitions)
        
        states_np = np.array(states, dtype=np.float32)
        actions_np = np.array(actions, dtype=np.float32)
        # Asegurar que actions_np tenga la forma (batch_size, action_dim)
        if actions_np.ndim == 1 and self.action_dim > 0 : # Si es una lista de escalares o vectores 1D y action_dim > 0
             actions_np = actions_np.reshape(-1, self.action_dim)
        elif actions_np.ndim == 0 and self.action_dim == 1: # Caso de un solo escalar
             actions_np = actions_np.reshape(-1,1)


        rewards_np = np.array(rewards, dtype=np.float32).reshape(-1, 1)
        next_states_np = np.array(next_states, dtype=np.float32)
        dones_np = np.array(dones, dtype=bool).reshape(-1, 1)
        
        return states_np, actions_np, rewards_np, next_states_np, dones_np
