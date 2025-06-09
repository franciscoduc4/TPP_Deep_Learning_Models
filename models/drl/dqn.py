"""
Implementación de Deep Q-Network (DQN) con discretización para dosificación de insulina.

DQN es un algoritmo de aprendizaje por refuerzo profundo que:
1. Discretiza el espacio continuo de dosis de insulina en valores finitos
2. Utiliza una red neuronal para predecir el valor Q de cada acción posible
3. Emplea un buffer de experiencia para entrenar de manera estable
4. Aplica redes objetivo y actualizaciones periódicas para mayor estabilidad
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from collections import deque

from custom.DeepReinforcementLearning.drl_pt import DRLModelWrapperPyTorch
from custom.printer import print_critical, print_debug, print_warning, print_info
from constants.constants import (
    IDEAL_LOWER_BOUND, IDEAL_UPPER_BOUND, SEVERE_HYPOGLYCEMIA_THRESHOLD, HYPOGLYCEMIA_THRESHOLD, 
    HYPERGLYCEMIA_THRESHOLD, SEVERE_HYPERGLYCEMIA_THRESHOLD, SEVERE_HYPO_PENALTY, HYPO_PENALTY_BASE, 
    HYPER_PENALTY_BASE, SEVERE_HYPER_PENALTY, MAX_REWARD
)
from config.models_config import DQN_CONFIG
from training.utils import compute_reward, calculate_iob
from validation.simulator import GlucoseSimulator
from models.drl.ddpg import ReplayBuffer  # Reutilizamos el buffer de experiencia


class QNetwork(nn.Module):
    """
    Red Q para DQN que evalúa el valor de cada acción discreta posible.
    
    Parámetros:
    -----------
    cgm_input_dim : tuple
        Dimensiones de entrada para datos CGM
    other_input_dim : tuple
        Dimensiones de entrada para otras características
    action_bins : int
        Número de acciones discretas posibles
    hidden_dim : int
        Dimensión de las capas ocultas
    """
    
    def __init__(self, cgm_input_dim: tuple, other_input_dim: tuple, 
                action_bins: int = 20, hidden_dim: int = 256) -> None:
        super().__init__()
        
        # Guardar dimensiones de entrada como atributos de instancia
        self.cgm_input_dim = cgm_input_dim
        self.other_input_dim = other_input_dim
        self.action_bins = action_bins
        self.hidden_dim = hidden_dim
        
        # Encoder para datos CGM
        self.cgm_encoder = nn.Sequential(
            nn.Linear(np.prod(cgm_input_dim), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Encoder para otras características
        self.other_encoder = nn.Sequential(
            nn.Linear(np.prod(other_input_dim), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Capas combinadas
        self.combined_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Capa de salida para valores Q de cada acción
        self.q_head = nn.Linear(hidden_dim // 2, action_bins)
        
    def forward(self, x_cgm: torch.Tensor, x_other: torch.Tensor) -> torch.Tensor:
        """
        Paso hacia adelante de la red Q.
        
        Parámetros:
        -----------
        x_cgm : torch.Tensor
            Datos CGM de entrada
        x_other : torch.Tensor
            Otras características de entrada
            
        Retorna:
        --------
        torch.Tensor
            Valores Q para cada acción discreta posible
        """
        # Aplanar entradas si es necesario
        if len(x_cgm.shape) > 2:
            x_cgm = x_cgm.reshape(x_cgm.shape[0], -1)
        if len(x_other.shape) > 2:
            x_other = x_other.reshape(x_other.shape[0], -1)
        
        # Verificar y manejar entradas vacías o incorrectas
        if x_cgm.shape[1] != np.prod(self.cgm_input_dim):
            print_warning(f"x_cgm shape mismatch: expected {np.prod(self.cgm_input_dim)}, got {x_cgm.shape[1]}")
            if x_cgm.shape[1] < np.prod(self.cgm_input_dim):
                padding = torch.zeros(x_cgm.shape[0], np.prod(self.cgm_input_dim) - x_cgm.shape[1], device=x_cgm.device)
                x_cgm = torch.cat([x_cgm, padding], dim=1)
        
        if x_other.shape[1] != np.prod(self.other_input_dim):
            print_warning(f"x_other shape mismatch: expected {np.prod(self.other_input_dim)}, got {x_other.shape[1]}")
            if x_other.shape[1] < np.prod(self.other_input_dim):
                padding = torch.zeros(x_other.shape[0], np.prod(self.other_input_dim) - x_other.shape[1], device=x_other.device)
                x_other = torch.cat([x_other, padding], dim=1)
        
        # Codificar cada componente
        cgm_features = self.cgm_encoder(x_cgm)
        other_features = self.other_encoder(x_other)
        
        # Combinar características
        combined = torch.cat([cgm_features, other_features], dim=1)
        features = self.combined_layer(combined)
        
        # Calcular valores Q para cada acción
        q_values = self.q_head(features)
        
        return q_values


class DQNModel(nn.Module):
    """
    Implementación de Deep Q-Network (DQN) con discretización para dosificación de insulina.
    
    Parámetros:
    -----------
    cgm_input_dim : tuple
        Dimensiones de entrada para datos CGM
    other_input_dim : tuple
        Dimensiones de entrada para otras características
    config : Dict[str, Any]
        Configuración del modelo DQN
    rewards_function : callable
        Función para calcular recompensas
    """
    
    def __init__(self,
                cgm_input_dim: tuple, 
                other_input_dim: tuple,
                config: Dict[str, Any] = DQN_CONFIG,
                rewards_function = None) -> None:
        """
        Inicializa el modelo DQN para dosificación de insulina.
        
        Parámetros:
        -----------
        cgm_input_dim : tuple
            Dimensiones de entrada para datos CGM
        other_input_dim : tuple
            Dimensiones de entrada para otras características
        config : Dict[str, Any], opcional
            Configuración del modelo DQN (default: DQN_CONFIG)
        rewards_function : callable, opcional
            Función para calcular recompensas (default: None)
        """
        super().__init__()
        
        # Guardar dimensiones de entrada
        self.cgm_input_dim = cgm_input_dim
        self.other_input_dim = other_input_dim
        
        # Inicializar semilla aleatoria para reproducibilidad
        seed = config.get('seed', 42)
        self.seed = seed
        torch.manual_seed(seed)
        self.rng = np.random.Generator(np.random.PCG64(seed))
        
        self.config = config
        
        # Parámetros para discretización del espacio de acción
        self.action_bins = config.get('action_bins', 20)  # Número de acciones discretas
        self.max_action = config.get('max_action', 10.0)  # Dosis máxima de insulina
        self.min_action = config.get('min_action', 0.0)   # Dosis mínima de insulina
        
        # Generar valores discretos de dosis
        self.action_values = torch.linspace(self.min_action, self.max_action, self.action_bins)
        
        # Parámetros del algoritmo
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.lr = config.get('learning_rate', 3e-4)
        self.buffer_size = config.get('buffer_size', 100000)
        self.batch_size = config.get('batch_size', 64)
        self.target_update_freq = config.get('target_update_freq', 10)
        
        # Parámetros para exploración epsilon-greedy
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.update_counter = 0
        
        # Inicializar redes Q
        self.q_network = QNetwork(cgm_input_dim, other_input_dim, self.action_bins)
        self.target_q_network = QNetwork(cgm_input_dim, other_input_dim, self.action_bins)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizador
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr, weight_decay=1e-5)
        
        # Buffer de experiencia
        self.buffer = ReplayBuffer(self.buffer_size, cgm_input_dim, other_input_dim, 1, self.rng)
        
        # Función de recompensa
        self.compute_rewards = rewards_function
        
        # Enviar redes al dispositivo correcto (CPU/GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
        # Inicializar con pesos adecuados
        self._initialize_networks()
        
        print_info("DQN Model iniciado correctamente con espacio de acción discreto")
    
    def _initialize_networks(self) -> None:
        """
        Inicializa las redes con pesos que producen salidas razonables desde el inicio.
        """
        # Inicialización especial para la última capa de la red Q
        if hasattr(self.q_network, 'q_head'):
            nn.init.uniform_(self.q_network.q_head.weight, -0.003, 0.003)
            # No sesgamos hacia valores positivos, dejamos que aprenda libremente
            nn.init.uniform_(self.q_network.q_head.bias, -0.1, 0.1)
            print_debug("Initialized Q-network head with special weights")
        
        # Asegurar que la red target tenga los mismos pesos
        self.target_q_network.load_state_dict(self.q_network.state_dict())
    
    def forward(self, x_cgm: torch.Tensor, x_other: torch.Tensor) -> torch.Tensor:
        """
        Realiza el paso hacia adelante del modelo.
        
        Parámetros:
        -----------
        x_cgm : torch.Tensor
            Datos CGM de entrada
        x_other : torch.Tensor
            Otras características de entrada
            
        Retorna:
        --------
        torch.Tensor
            Acción predicha (dosis de insulina)
        """
        # Forward devuelve la acción elegida, no los valores Q
        q_values = self.q_network(x_cgm, x_other)
        action_indices = torch.argmax(q_values, dim=1)
        return self.action_values[action_indices].reshape(-1, 1)

    def select_action(self, x_cgm: torch.Tensor, x_other: torch.Tensor, 
                  add_noise: bool = True) -> torch.Tensor:
        """
        Selecciona una acción basada en el estado actual usando epsilon-greedy.
        
        Parámetros:
        -----------
        x_cgm : torch.Tensor
            Datos CGM de entrada
        x_other : torch.Tensor
            Otras características de entrada
        add_noise : bool, opcional
            Si aplicar exploración epsilon-greedy (default: True)
        
        Retorna:
        --------
        torch.Tensor
            Acción seleccionada
        """
        # Asegurar modo de evaluación
        self.q_network.eval()
        
        # Aplanar tensores si es necesario
        if len(x_cgm.shape) > 2:
            x_cgm = x_cgm.reshape(x_cgm.shape[0], -1)
            print_debug(f"Reshaping x_cgm to {x_cgm.shape}")
        if len(x_other.shape) > 2:
            x_other = x_other.reshape(x_other.shape[0], -1)
            print_debug(f"Reshaping x_other to {x_other.shape}")
        
        # Epsilon-greedy: exploración o explotación
        batch_size = x_cgm.shape[0]
        actions = torch.zeros((batch_size, 1), device=self.device)
        
        with torch.no_grad():
            # Calcular valores Q para todas las acciones
            q_values = self.q_network(x_cgm, x_other)
            
            # Para cada elemento en el batch
            for i in range(batch_size):
                # Decidir entre exploración o explotación
                if add_noise and self.rng.random() < self.epsilon:
                    # Exploración: elegir acción aleatoria
                    action_idx = self.rng.integers(0, self.action_bins)
                else:
                    # Explotación: elegir mejor acción según valores Q
                    action_idx = torch.argmax(q_values[i]).item()
                
                # Convertir índice a valor de acción
                actions[i, 0] = self.action_values[action_idx]
                
                # Verificar si hay carbohidratos (estrategia híbrida)
                carbs = x_other[i, 0].item()
                if carbs > 20.0:
                    # Si hay comida, asegurar dosis mínima proporcional a carbohidratos
                    min_dose = carbs / 30.0  # Ratio conservador de 1:30
                    actions[i, 0] = max(actions[i, 0], min_dose)
            
            print_debug(f"DQN selected actions: {actions.cpu().numpy()}")
        
        # Actualizar epsilon para la próxima selección
        if add_noise:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            print_debug(f"Updated epsilon to {self.epsilon:.4f}")
        
        # Volver a modo de entrenamiento
        self.q_network.train()
        
        return actions
    
    def add_to_buffer(self, buffer: ReplayBuffer, state: Tuple[np.ndarray, np.ndarray], 
                  action: np.ndarray, reward: float, 
                  next_state: Tuple[np.ndarray, np.ndarray], done: bool) -> None:
        """
        Añade una transición al buffer de experiencia.
        
        Parámetros:
        -----------
        buffer : ReplayBuffer
            Buffer de experiencia
        state : Tuple[np.ndarray, np.ndarray]
            Estado actual (x_cgm, x_other)
        action : np.ndarray
            Acción tomada (dosis de insulina)
        reward : float
            Recompensa recibida
        next_state : Tuple[np.ndarray, np.ndarray]
            Estado siguiente (x_cgm_next, x_other_next)
        done : bool
            Indicador de fin de episodio
        """
        buffer.add(state, action, reward, next_state, done)
    
    def sample_buffer(self, buffer: ReplayBuffer, batch_size: int) -> Tuple:
        """
        Muestrea un batch de transiciones del buffer.
        
        Parámetros:
        -----------
        buffer : ReplayBuffer
            Buffer de experiencia
        batch_size : int
            Tamaño del batch a muestrear
            
        Retorna:
        --------
        Tuple
            Batch de transiciones
        """
        return buffer.sample(batch_size)
    
    def update(self, batch: Tuple) -> Dict[str, float]:
        """
        Actualiza la red Q usando un batch de experiencias.
        
        Parámetros:
        -----------
        batch : Tuple
            Batch de experiencias (states, actions, rewards, next_states, dones)
            
        Retorna:
        --------
        Dict[str, float]
            Diccionario con pérdidas de la red Q
        """
        states, actions, rewards, next_states, dones = batch
        states_cgm, states_other = states
        next_states_cgm, next_states_other = next_states
        
        # Convertir a tensores
        states_cgm_t = torch.FloatTensor(states_cgm).to(self.device)
        states_other_t = torch.FloatTensor(states_other).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_cgm_t = torch.FloatTensor(next_states_cgm).to(self.device)
        next_states_other_t = torch.FloatTensor(next_states_other).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        
        # Asegurar que los tensores no contengan NaN
        states_cgm_t = torch.nan_to_num(states_cgm_t)
        states_other_t = torch.nan_to_num(states_other_t)
        actions_t = torch.nan_to_num(actions_t)
        rewards_t = torch.nan_to_num(rewards_t)
        next_states_cgm_t = torch.nan_to_num(next_states_cgm_t)
        next_states_other_t = torch.nan_to_num(next_states_other_t)
        dones_t = torch.nan_to_num(dones_t)
        
        # Calcular Q-valores para el estado actual
        q_values = self.q_network(states_cgm_t, states_other_t)
        
        # Convertir acciones continuas a índices discretos para seleccionar Q-valores
        action_indices = torch.zeros(actions_t.shape[0], dtype=torch.long, device=self.device)
        for i in range(actions_t.shape[0]):
            # Encontrar el índice más cercano en el espacio discretizado
            action_value = actions_t[i, 0].item()
            action_idx = torch.abs(self.action_values - action_value).argmin().item()
            action_indices[i] = action_idx
        
        # Seleccionar Q-valores para las acciones tomadas
        q_values = q_values.gather(1, action_indices.unsqueeze(1))
        
        # Calcular Q-target usando la red target
        with torch.no_grad():
            # Calcular máximo Q-valor para el siguiente estado
            next_q_values = self.target_q_network(next_states_cgm_t, next_states_other_t)
            next_q_values_max = next_q_values.max(1, keepdim=True)[0]
            
            # Calcular target Q-value usando la ecuación de Bellman
            q_target = rewards_t + (1 - dones_t) * self.gamma * next_q_values_max
        
        # Calcular pérdida MSE
        loss = F.mse_loss(q_values, q_target)
        
        # Optimizar red Q
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Actualizar red target periódicamente
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self._update_target_network()
        
        return {
            'q_loss': loss.item(),
            'total_loss': loss.item(),
            'epsilon': self.epsilon
        }
    
    def _update_target_network(self) -> None:
        """
        Actualiza la red Q target usando actualización suave (soft update).
        """
        # Actualizar red target
        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def _validate_inputs(self, x_cgm: np.ndarray, x_other: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Valida y corrige entradas con valores nulos o NaN.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM para predicción
        x_other : np.ndarray
            Otras características para predicción
            
        Retorna:
        --------
        Tuple[np.ndarray, np.ndarray]
            Datos CGM y otras características validados
        """
        if x_cgm is None or x_other is None or np.isnan(x_cgm).any() or np.isnan(x_other).any():
            print_warning("Se detectaron valores nulos o NaN en las entradas. Aplicando corrección.")
            if x_cgm is None:
                raise ValueError("Los datos CGM no pueden ser nulos")
            if x_other is None:
                raise ValueError("Los datos de otras características no pueden ser nulos")
            
            # Reemplazar NaN con valores seguros
            if np.isnan(x_cgm).any():
                x_cgm = np.nan_to_num(x_cgm, nan=100.0)  # Valor CGM predeterminado seguro
            if np.isnan(x_other).any():
                x_other = np.nan_to_num(x_other, nan=0.0)
                
        return x_cgm, x_other

    def _prepare_tensors(self, x_cgm: np.ndarray, x_other: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convierte arrays numpy a tensores y ajusta sus dimensiones.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM validados
        x_other : np.ndarray
            Otras características validadas
            
        Retorna:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            Tensores preparados para la red
        """
        # Convertir a tensores
        x_cgm_tensor = torch.FloatTensor(x_cgm).to(self.device)
        x_other_tensor = torch.FloatTensor(x_other).to(self.device)
        
        # Asegurar dimensiones correctas
        if len(x_cgm_tensor.shape) == 2:
            x_cgm_tensor = x_cgm_tensor.unsqueeze(0)
        if len(x_other_tensor.shape) == 1:
            x_other_tensor = x_other_tensor.unsqueeze(0)
            
        return x_cgm_tensor, x_other_tensor

    def predict(self, x_cgm: np.ndarray, x_other: np.ndarray) -> np.ndarray:
        """
        Predice dosis de insulina para los estados dados.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM para predicción
        x_other : np.ndarray
            Otras características para predicción
            
        Retorna:
        --------
        np.ndarray
            Dosis de insulina predichas
        """
        # Validar entradas
        x_cgm, x_other = self._validate_inputs(x_cgm, x_other)
        
        # Preparar tensores
        x_cgm_tensor, x_other_tensor = self._prepare_tensors(x_cgm, x_other)
        
        # Depuración
        print_debug(f"predict x_cgm_tensor shape: {x_cgm_tensor.shape}, x_other_tensor shape: {x_other_tensor.shape}")
        
        # Predecir sin exploración (explotación pura)
        with torch.no_grad():
            self.q_network.eval()
            q_values = self.q_network(x_cgm_tensor, x_other_tensor)
            action_indices = torch.argmax(q_values, dim=1)
            actions = self.action_values[action_indices].reshape(-1, 1)
            self.q_network.train()
        
        # Convertir a numpy
        pred_numpy = actions.cpu().numpy()
        
        # Extraer valor de carbohidratos
        carbs = float(x_other[0, 0]) if x_other.shape[1] > 0 else 0.0
        
        # Si hay carbohidratos significativos, asegurar dosis mínima
        if carbs > 20.0:
            min_dose = carbs / 30.0  # Ratio conservador de 1:30
            if pred_numpy[0, 0] < min_dose:
                print_debug(f"Ajustando dosis demasiado baja: {pred_numpy[0, 0]} para {carbs}g de carbohidratos")
                pred_numpy[0, 0] = min_dose
        
        print_debug(f"Dosis final DQN: {pred_numpy}, para carbohidratos: {carbs}")
        return pred_numpy
    
    def _extract_current_glucose(self, x_cgm: np.ndarray) -> float:
        """
        Extrae el valor actual de glucosa de los datos CGM.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM
            
        Retorna:
        --------
        float
            Valor actual de glucosa
        """
        # Extraer último valor de glucosa según la forma del array
        if len(x_cgm.shape) == 3:  # [batch, time_steps, features]
            return float(x_cgm[0, -1, 0])
        elif len(x_cgm.shape) == 2:  # [time_steps, features]
            return float(x_cgm[-1, 0])
        else:  # [features]
            return float(x_cgm[-1])
    
    def _prepare_context_inputs(self, x_other: np.ndarray, carb_intake: float, iob: float,
                              sleep_quality: float, work_intensity: float, 
                              exercise_intensity: float) -> np.ndarray:
        """
        Prepara entradas con contexto para predicción.
        
        Parámetros:
        -----------
        x_other : np.ndarray
            Otras características
        carb_intake : float
            Ingesta de carbohidratos
        iob : float
            Insulina a bordo
        sleep_quality : float
            Calidad del sueño
        work_intensity : float
            Intensidad del trabajo
        exercise_intensity : float
            Intensidad del ejercicio
            
        Retorna:
        --------
        np.ndarray
            Características con contexto incluido
        """
        # Crear copia para no modificar el original
        x_other_copy = x_other.copy()
        
        # Si es un array 3D, extraer la primera muestra
        if len(x_other_copy.shape) == 3:
            x_other_copy = x_other_copy[0]
        
        # Asegurar que es 2D
        if len(x_other_copy.shape) == 1:
            x_other_copy = x_other_copy.reshape(1, -1)
        
        # Actualizar ingesta de carbohidratos (suponemos que está en la primera columna)
        x_other_copy[0, 0] = carb_intake
        
        # Si hay columna para IOB, actualizarla
        if x_other_copy.shape[1] > 1:
            # Asumimos que IOB está en la segunda columna o agregamos información si hay espacio
            x_other_copy[0, min(1, x_other_copy.shape[1]-1)] = iob
        
        # Agregar otras variables de contexto si las columnas existen
        if sleep_quality is not None and x_other_copy.shape[1] > 2:
            x_other_copy[0, 2] = sleep_quality
            
        if work_intensity is not None and x_other_copy.shape[1] > 3:
            x_other_copy[0, 3] = work_intensity
            
        if exercise_intensity is not None and x_other_copy.shape[1] > 4:
            x_other_copy[0, 4] = exercise_intensity
        
        return x_other_copy
    
    def _adjust_for_glucose_level(self, prediction_value: float, current_glucose: float) -> float:
        """
        Ajusta la dosis predicha según el nivel actual de glucosa.
        
        Parámetros:
        -----------
        prediction_value : float
            Dosis predicha
        current_glucose : float
            Nivel actual de glucosa
            
        Retorna:
        --------
        float
            Dosis ajustada
        """
        # Factor de ajuste según nivel de glucosa
        if current_glucose < HYPOGLYCEMIA_THRESHOLD:
            # Reducir o eliminar dosis en hipoglucemia
            return 0.0
        elif current_glucose < IDEAL_LOWER_BOUND:
            # Reducir dosis si está por debajo del rango ideal
            reduction_factor = (current_glucose - HYPOGLYCEMIA_THRESHOLD) / (IDEAL_LOWER_BOUND - HYPOGLYCEMIA_THRESHOLD)
            return prediction_value * reduction_factor
        elif current_glucose > SEVERE_HYPERGLYCEMIA_THRESHOLD:
            # Aumentar dosis en hiperglucemia severa
            increase_factor = 1.0 + min(0.5, (current_glucose - HYPERGLYCEMIA_THRESHOLD) / 100.0)
            return prediction_value * increase_factor
        elif current_glucose > HYPERGLYCEMIA_THRESHOLD:
            # Aumentar ligeramente la dosis en hiperglucemia moderada
            increase_factor = 1.0 + min(0.3, (current_glucose - HYPERGLYCEMIA_THRESHOLD) / 100.0)
            return prediction_value * increase_factor
        else:
            # En rango ideal, mantener dosis
            return prediction_value
    
    def _adjust_for_carbs_and_iob(self, prediction_value: float, carb_intake: float, iob: float) -> float:
        """
        Ajusta la dosis predicha según la ingesta de carbohidratos y la insulina a bordo.
        
        Parámetros:
        -----------
        prediction_value : float
            Dosis predicha
        carb_intake : float
            Ingesta de carbohidratos
        iob : float
            Insulina a bordo
            
        Retorna:
        --------
        float
            Dosis ajustada
        """
        # Sin carbohidratos, reducir o eliminar dosis
        if carb_intake <= 0:
            return max(0.0, prediction_value - iob)
        
        # Calcular dosis basada en carbohidratos (ratio conservador)
        carb_based_dose = carb_intake / 15.0
        
        # Si la dosis predicha es muy baja pero hay carbohidratos, usar al menos 
        # un porcentaje de la dosis basada en carbohidratos
        if prediction_value < carb_based_dose * 0.3:
            prediction_value = max(prediction_value, carb_based_dose * 0.3)
        
        # Considerar la insulina a bordo
        adjusted_dose = max(0.0, prediction_value - max(0.0, iob - 1.0))
        
        return adjusted_dose
    
    def predict_with_context(self, x_cgm: np.ndarray, x_other: np.ndarray, 
                          carb_intake: float,
                          sleep_quality: float = None,
                          work_intensity: float = None,
                          exercise_intensity: float = None,
                          current_glucose: float = None,
                          iob: float = None) -> float:
        """
        Predice dosis de insulina con información contextual adicional.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM para predicción
        x_other : np.ndarray
            Otras características para predicción
        carb_intake : float
            Ingesta de carbohidratos en gramos
        sleep_quality : float, opcional
            Calidad del sueño (escala 0-10)
        work_intensity : float, opcional
            Intensidad del trabajo (escala 0-10)
        exercise_intensity : float, opcional
            Intensidad del ejercicio (escala 0-10)
        current_glucose : float, opcional
            Nivel actual de glucosa en mg/dL
        iob : float, opcional
            Insulina activa en el cuerpo (Insulin On Board)

        Retorna:
        --------
        float
            Dosis de insulina recomendada
        """
        # Añadir un print claro para confirmar que se usa esta implementación
        print_critical("Using DQN's predict_with_context - Direct from nn.Module")
        
        if carb_intake <= 0.0 and (current_glucose is None or current_glucose < HYPERGLYCEMIA_THRESHOLD):
            print_critical("Ingesta de carbohidratos es 0.0 y glucosa normal, no se recomienda dosis de insulina.")
            return 0.0
        
        print_debug(f"Predicción con contexto: ingesta de carbohidratos: {carb_intake}, glucosa actual: {current_glucose}, IOB: {iob}, sleep_quality: {sleep_quality}, work_intensity: {work_intensity}, exercise_intensity: {exercise_intensity}")
        
        # Extraer o utilizar nivel de glucosa proporcionado
        if current_glucose is None:
            current_glucose = self._extract_current_glucose(x_cgm)

        # Manejar IOB no proporcionado
        if iob is None:
            from training.common import calculate_iob
            iob = calculate_iob(x_cgm, carb_intake)

        # Valores por defecto para parámetros opcionales
        sleep_quality = 5.0 if sleep_quality is None else float(sleep_quality)
        work_intensity = 0.0 if work_intensity is None else float(work_intensity)
        exercise_intensity = 0.0 if exercise_intensity is None else float(exercise_intensity)

        # Preparar entrada con contexto
        x_other_with_context = self._prepare_context_inputs(
            x_other, carb_intake, iob, sleep_quality, 
            work_intensity, exercise_intensity
        )

        # Predecir usando la entrada con contexto
        prediction = self.predict(x_cgm, x_other_with_context)
        
        print_debug(f"Predicción inicial (DQN): {prediction}, glucosa actual: {current_glucose}, IOB: {iob}, ingesta de carbohidratos: {carb_intake}")

        # Extraer valor de predicción
        prediction_value = float(prediction[0, 0] if prediction.shape[1] == 1 else prediction[0])
        
        # Aplicar ajustes
        prediction_value = self._adjust_for_glucose_level(prediction_value, current_glucose)
        prediction_value = self._adjust_for_carbs_and_iob(prediction_value, carb_intake, iob)

        # Asegurar límites seguros
        return max(0.0, min(prediction_value, self.max_action))
    
    def to(self, device: torch.device) -> 'DQNModel':
        """
        Mueve el modelo al dispositivo especificado.
        
        Parámetros:
        -----------
        device : torch.device
            Dispositivo de destino
            
        Retorna:
        --------
        DQNModel
            Self para encadenamiento
        """
        self.device = device
        self.q_network = self.q_network.to(device)
        self.target_q_network = self.target_q_network.to(device)
        self.action_values = self.action_values.to(device)
        return self


def create_dqn_model(cgm_input_dim: tuple, other_input_dim: tuple) -> DRLModelWrapperPyTorch:
    """
    Crea un modelo DQN para dosificación de insulina con discretización.
    
    Parámetros:
    -----------
    cgm_input_dim : tuple
        Dimensiones de entrada para datos CGM
    other_input_dim : tuple
        Dimensiones de entrada para otras características
        
    Retorna:
    --------
    DRLModelWrapperPyTorch
        Modelo DQN inicializado envuelto en DRLModelWrapperPyTorch
    """
    from training.utils import compute_reward
    
    model = DQNModel(
        cgm_input_dim=cgm_input_dim,
        other_input_dim=other_input_dim,
        config=DQN_CONFIG,
        rewards_function=compute_reward
    )
    
    print_critical(f"Tipo de modelo creado: {type(model)}")
    
    wrapper = DRLModelWrapperPyTorch(model, algorithm="DQN")
    print_critical(f"Tipo de wrapper: {type(wrapper)}")
    print_critical(f"Tipo de modelo en wrapper: {type(wrapper.model)}")
    
    return wrapper