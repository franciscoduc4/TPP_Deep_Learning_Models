import numpy as np
import torch
from typing import Tuple, Any
from config.params import BUFFER_CONFIG
from custom.printer import print_warning

class ReplayBuffer:
    """
    Buffer de experiencia para almacenar y muestrear transiciones.
    
    Parámetros:
    -----------
    capacity : int
        Capacidad máxima del buffer
    cgm_dim : tuple
        Dimensiones de los datos CGM
    other_dim : tuple
        Dimensiones de otras características
    action_dim : int
        Dimensión de la acción
    rng : np.random.Generator
        Generador de números aleatorios
    """
    
    def __init__(self, capacity: int, cgm_dim: tuple, other_dim: tuple, 
                action_dim: int = BUFFER_CONFIG['action_dim'], rng: np.random.Generator = None) -> None:
        self.capacity = capacity
        self.cgm_dim = cgm_dim
        self.other_dim = other_dim
        self.action_dim = action_dim
        self.rng = rng or np.random.default_rng(self.seed)
        
        self.buffer = []
        self.position = 0
    
    def __len__(self) -> int:
        """
        Obtiene el número de transiciones en el buffer.
        
        Retorna:
        --------
        int
            Número de transiciones almacenadas
        """
        return len(self.buffer)
        
    def _log_debug_info(self, state: Tuple[Any, Any], action: Any, next_state: Tuple[Any, Any]) -> None:
        """
        Registra información de depuración sobre los tipos de datos y dispositivos.
        
        Parámetros:
        -----------
        state : Tuple[Any, Any]
            Estado actual
        action : Any
            Acción tomada
        next_state : Tuple[Any, Any]
            Estado siguiente
        """
        print(f"Adding to buffer: state types: {type(state[0])}, {type(state[1])}, "
              f"action type: {type(action)}, next_state types: {type(next_state[0])}, {type(next_state[1])}")
        
        if isinstance(state[0], torch.Tensor):
            print(f"state[0] device: {state[0].device}")
        if isinstance(state[1], torch.Tensor):
            print(f"state[1] device: {state[1].device}")
        if isinstance(action, torch.Tensor):
            print(f"action device: {action.device}")
    
    def _validate_input_types(self, state: Tuple[Any, Any], reward: float, next_state: Tuple[Any, Any], done: bool) -> None:
        """
        Valida los tipos de los datos de entrada.
        
        Parámetros:
        -----------
        state : Tuple[Any, Any]
            Estado actual
        reward : float
            Recompensa recibida
        next_state : Tuple[Any, Any]
            Estado siguiente
        done : bool
            Indicador de fin de episodio
        """
        if not isinstance(state, tuple) or len(state) != 2:
            raise ValueError("State debe ser una tupla de dos elementos")
        if not isinstance(next_state, tuple) or len(next_state) != 2:
            raise ValueError("Next_state debe ser una tupla de dos elementos")
        if not isinstance(reward, (int, float)):
            raise ValueError("Reward debe ser un número (int o float)")
        if not isinstance(done, bool):
            raise ValueError("Done debe ser un booleano")
    
    def _convert_to_numpy(self, state: Tuple[Any, Any], action: Any, next_state: Tuple[Any, Any]) -> Tuple:
        """
        Convierte los datos de entrada a arrays de NumPy.
        
        Parámetros:
        -----------
        state : Tuple[Any, Any]
            Estado actual
        action : Any
            Acción tomada
        next_state : Tuple[Any, Any]
            Estado siguiente
            
        Retorna:
        --------
        Tuple
            Datos convertidos a NumPy
        """
        # Convertir state y next_state
        state_np = (
            # Eliminar dimensión de batch si existe
            state[0].detach().cpu().numpy().squeeze(0) if isinstance(state[0], torch.Tensor) else np.asarray(state[0], dtype=np.float32),
            state[1].detach().cpu().numpy().squeeze(0) if isinstance(state[1], torch.Tensor) else np.asarray(state[1], dtype=np.float32)
        )
        
        next_state_np = (
            # Eliminar dimensión de batch si existe
            next_state[0].detach().cpu().numpy().squeeze(0) if isinstance(next_state[0], torch.Tensor) else np.asarray(next_state[0], dtype=np.float32),
            next_state[1].detach().cpu().numpy().squeeze(0) if isinstance(next_state[1], torch.Tensor) else np.asarray(next_state[1], dtype=np.float32)
        )
        
        # Convertir action
        if isinstance(action, torch.Tensor):
            action_np = action.detach().cpu().numpy()
        elif isinstance(action, np.ndarray):
            action_np = action
        elif np.isscalar(action):
            action_np = np.array([action], dtype=np.float32)
        else:
            action_np = np.asarray(action, dtype=np.float32)
    
        # Ensure action_np is 1D
        action_np = action_np.flatten()
        
        return state_np, action_np, next_state_np
    
    def _validate_dimensions(self, state_np: Tuple[np.ndarray, np.ndarray], action_np: np.ndarray) -> None:
        """
        Valida las dimensiones de los datos convertidos.
        
        Parámetros:
        -----------
        state_np : Tuple[np.ndarray, np.ndarray]
            Estado convertido a NumPy
        action_np : np.ndarray
            Acción convertida a NumPy
        """
        if state_np[0].shape != self.cgm_dim:
            raise ValueError(f"CGM state dimension {state_np[0].shape} does not match expected {self.cgm_dim}")
        if state_np[1].shape != self.other_dim:
            raise ValueError(f"Other state dimension {state_np[1].shape} does not match expected {self.other_dim}")
        if action_np.shape != (self.action_dim,):
            raise ValueError(f"Action dimension {action_np.shape} does not match expected {(self.action_dim,)}")
    
    def add(self, state: Tuple[Any, Any], action: Any, 
        reward: float, next_state: Tuple[Any, Any], 
        done: bool) -> None:
        """
        Agrega una transición al buffer.
        
        Parámetros:
        -----------
        state : Tuple[Any, Any]
            Estado actual (x_cgm, x_other)
        action : Any
            Acción tomada
        reward : float
            Recompensa recibida
        next_state : Tuple[Any, Any]
            Estado siguiente (x_cgm_next, x_other_next)
        done : bool
            Indicador de fin de episodio
            
        Retorna:
        --------
        None
        """
        # self._log_debug_info(state, action, next_state)
        self._validate_input_types(state, reward, next_state, done)
        
        state_np, action_np, next_state_np = self._convert_to_numpy(state, action, next_state)
        self._validate_dimensions(state_np, action_np)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            
        self.buffer[self.position] = (state_np, action_np, reward, next_state_np, done)
        self.position = (self.position + 1) % self.capacity
        
    def push(self, state: Tuple[Any, Any], action: Any, 
         reward: float, next_state: Tuple[Any, Any], 
         done: bool) -> None:
        """
        Añade una transición al buffer (alias para add).
        
        Parámetros:
        -----------
        state : Tuple[Any, Any]
            Estado actual (x_cgm, x_other)
        action : Any
            Acción tomada
        reward : float
            Recompensa recibida
        next_state : Tuple[Any, Any]
            Estado siguiente (x_cgm_next, x_other_next)
        done : bool
            Indicador de fin de episodio
            
        Retorna:
        --------
        None
        """
        self.add(state, action, reward, next_state, done)

    def sample(self, batch_size: int) -> Tuple:
        """
        Muestrea un batch de transiciones del buffer.
        
        Parámetros:
        -----------
        batch_size : int
            Tamaño del batch a muestrear
            
        Retorna:
        --------
        Tuple
            Batch de transiciones (estados, acciones, recompensas, siguientes estados, done flags)
        """
        # Verificar si hay elementos en el buffer
        buffer_size = len(self.buffer)
        if buffer_size == 0:
            raise ValueError("No hay elementos en el buffer para muestrear")
        
        # Ajustar el batch_size al tamaño disponible
        effective_batch_size = min(buffer_size, batch_size)
        if effective_batch_size < batch_size:
            print_warning(f"Ajustando batch_size de {batch_size} a {effective_batch_size} (elementos disponibles)")
        
        # Generar índices aleatorios con reemplazo para garantizar batch completo
        indices = self.rng.choice(buffer_size, effective_batch_size, replace=True)
        
        # Extraer elementos según los índices
        batch = [self.buffer[i] for i in indices]
        
        # Desempaquetar batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Desempaquetar estados y siguientes estados
        x_cgm, x_other = zip(*states)
        x_cgm_next, x_other_next = zip(*next_states)
        
        # Convertir a arrays de numpy
        x_cgm_np = np.array(x_cgm)
        x_other_np = np.array(x_other)
        actions_np = np.array(actions)
        rewards_np = np.array(rewards).reshape(-1, 1)
        x_cgm_next_np = np.array(x_cgm_next)
        x_other_next_np = np.array(x_other_next)
        dones_np = np.array(dones).reshape(-1, 1)
        
        return (
            (x_cgm_np, x_other_np),
            actions_np,
            rewards_np,
            (x_cgm_next_np, x_other_next_np),
            dones_np
        )
        
    def __len__(self) -> int:
        """
        Obtiene el número de transiciones en el buffer.
        
        Retorna:
        --------
        int
            Número de transiciones almacenadas
        """
        return len(self.buffer)
