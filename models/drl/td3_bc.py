"""
Implementación de Twin Delayed DDPG with Behavior Cloning (TD3+BC) para dosificación de insulina.

TD3+BC es un algoritmo de aprendizaje por refuerzo profundo offline que combina:
1. TD3: Usa dos redes críticas para reducir la sobreestimación del valor Q
2. Actualizaciones retrasadas del actor para mayor estabilidad
3. Behavior Cloning para mejorar el aprendizaje a partir de datos históricos
4. Regularización para evitar extrapolaciones fuera de la distribución de datos
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
from config.models_config import TD3_BC_CONFIG
from training.utils import compute_reward, calculate_iob
from validation.simulator import GlucoseSimulator
from models.drl.ddpg import ReplayBuffer


class TD3Actor(nn.Module):
    """
    Red del actor para TD3+BC que determina la acción óptima para un estado dado.
    
    Parámetros:
    -----------
    cgm_input_dim : tuple
        Dimensiones de entrada para datos CGM
    other_input_dim : tuple
        Dimensiones de entrada para otras características
    action_dim : int
        Dimensión de la acción (dosis de insulina)
    hidden_dim : int
        Dimensión de las capas ocultas
    max_action : float
        Valor máximo de acción permitido
    """
    
    def __init__(self, cgm_input_dim: tuple, other_input_dim: tuple, 
             action_dim: int = TD3_BC_CONFIG['action_dim'], 
             hidden_dim: int = TD3_BC_CONFIG['hidden_dim'], 
             max_action: float = TD3_BC_CONFIG['max_action']) -> None:
        super().__init__()
        
        # Guardar dimensiones de entrada como atributos de instancia
        self.cgm_input_dim = cgm_input_dim
        self.other_input_dim = other_input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_action = max_action
        
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
        
        # Capa de salida para acción (dosis de insulina)
        self.action_head = nn.Linear(hidden_dim // 2, action_dim)
        
    def forward(self, x_cgm: torch.Tensor, x_other: torch.Tensor) -> torch.Tensor:
        """
        Paso hacia adelante de la red del actor.
        
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
        
        # Generar acción
        action = torch.sigmoid(self.action_head(features)) * self.max_action
        
        # Ajustar acción para dosis de insulina
        carbs_factor = x_other[:, 0:1] / 20.0  # Simple 1:20 insulin:carb ratio
        carbs_factor = torch.clamp(carbs_factor, 0.0, self.max_action / 2)
        
        # Combinar acción con factor de carbohidratos
        action = 0.7 * action + 0.3 * carbs_factor
        
        return action


class TD3Critic(nn.Module):
    """
    Red del crítico para TD3+BC que evalúa la calidad de pares estado-acción.
    
    Parámetros:
    -----------
    cgm_input_dim : tuple
        Dimensiones de entrada para datos CGM
    other_input_dim : tuple
        Dimensiones de entrada para otras características
    action_dim : int
        Dimensión de la acción (dosis de insulina)
    hidden_dim : int
        Dimensión de las capas ocultas
    """
    
    def __init__(self, cgm_input_dim: tuple, other_input_dim: tuple, 
                action_dim: int = TD3_BC_CONFIG['action_dim'], 
                hidden_dim: int = TD3_BC_CONFIG['hidden_dim']) -> None:
        super().__init__()
        
        # Guardar dimensiones de entrada como atributos de instancia
        self.cgm_input_dim = cgm_input_dim
        self.other_input_dim = other_input_dim
        self.action_dim = action_dim
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
        
        # Encoder para acciones
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 4),
            nn.ReLU()
        )
        
        # Capa combinada
        combined_dim = hidden_dim + hidden_dim // 4
        self.combined_layer = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x_cgm: torch.Tensor, x_other: torch.Tensor, 
           action: torch.Tensor) -> torch.Tensor:
        """
        Paso hacia adelante de la red del crítico.
        
        Parámetros:
        -----------
        x_cgm : torch.Tensor
            Datos CGM de entrada
        x_other : torch.Tensor
            Otras características de entrada
        action : torch.Tensor
            Acción (dosis de insulina)
            
        Retorna:
        --------
        torch.Tensor
            Valor Q estimado
        """
        # Aplanar entradas si es necesario
        if len(x_cgm.shape) > 2:
            x_cgm = x_cgm.reshape(x_cgm.shape[0], -1)
        if len(x_other.shape) > 2:
            x_other = x_other.reshape(x_other.shape[0], -1)
        
        # Verificar y manejar entradas vacías o incorrectas
        if x_cgm.shape[1] != np.prod(self.cgm_input_dim):
            print_warning(f"Critic: x_cgm shape mismatch: expected {np.prod(self.cgm_input_dim)}, got {x_cgm.shape[1]}")
            if x_cgm.shape[1] < np.prod(self.cgm_input_dim):
                padding = torch.zeros(x_cgm.shape[0], np.prod(self.cgm_input_dim) - x_cgm.shape[1], device=x_cgm.device)
                x_cgm = torch.cat([x_cgm, padding], dim=1)
        
        if x_other.shape[1] != np.prod(self.other_input_dim):
            print_warning(f"Critic: x_other shape mismatch: expected {np.prod(self.other_input_dim)}, got {x_other.shape[1]}")
            if x_other.shape[1] < np.prod(self.other_input_dim):
                padding = torch.zeros(x_other.shape[0], np.prod(self.other_input_dim) - x_other.shape[1], device=x_other.device)
                x_other = torch.cat([x_other, padding], dim=1)
        
        # Codificar cada componente
        cgm_features = self.cgm_encoder(x_cgm)
        other_features = self.other_encoder(x_other)
        action_features = self.action_encoder(action)
        
        # Combinar características de estado
        state_features = torch.cat([cgm_features, other_features], dim=1)
        
        # Combinar con características de acción
        combined = torch.cat([state_features, action_features], dim=1)
        
        # Estimar valor Q
        q_value = self.combined_layer(combined)
        
        return q_value


class TD3BCModel(nn.Module):
    """
    Implementación de Twin Delayed DDPG with Behavior Cloning (TD3+BC) para dosificación de insulina.
    
    Parámetros:
    -----------
    cgm_input_dim : tuple
        Dimensiones de entrada para datos CGM
    other_input_dim : tuple
        Dimensiones de entrada para otras características
    config : Dict[str, Any]
        Configuración del modelo TD3+BC
    rewards_function : callable
        Función para calcular recompensas
    """
    
    def __init__(self,
                cgm_input_dim: tuple, 
                other_input_dim: tuple,
                config: Dict[str, Any] = TD3_BC_CONFIG,
                rewards_function = None) -> None:
        """
        Inicializa el modelo TD3+BC para dosificación de insulina.
        
        Parámetros:
        -----------
        cgm_input_dim : tuple
            Dimensiones de entrada para datos CGM
        other_input_dim : tuple
            Dimensiones de entrada para otras características
        config : Dict[str, Any], opcional
            Configuración del modelo TD3+BC (default: TD3_BC_CONFIG)
        rewards_function : callable, opcional
            Función para calcular recompensas (default: None)
        """
        super().__init__()
        
        # Guardar dimensiones de entrada
        self.cgm_input_dim = cgm_input_dim
        self.other_input_dim = other_input_dim
        
        # Inicializar semilla aleatoria para reproducibilidad
        seed = config.get('seed', TD3_BC_CONFIG['seed'])
        self.seed = seed
        torch.manual_seed(seed)
        self.rng = np.random.Generator(np.random.PCG64(seed))
        
        self.config = config
        
        # Parámetros del algoritmo
        self.action_dim = config.get('action_dim', TD3_BC_CONFIG['action_dim'])
        self.hidden_dim = config.get('hidden_dim', TD3_BC_CONFIG['hidden_dim'])
        self.gamma = config.get('gamma', TD3_BC_CONFIG['gamma'])
        self.tau = config.get('tau', TD3_BC_CONFIG['tau'])
        self.policy_noise = config.get('policy_noise', TD3_BC_CONFIG['policy_noise'])
        self.noise_clip = config.get('noise_clip', TD3_BC_CONFIG['noise_clip'])
        self.policy_delay = config.get('policy_delay', TD3_BC_CONFIG['policy_delay'])
        self.alpha = config.get('alpha', TD3_BC_CONFIG['alpha'])  # Factor de peso para la pérdida de BC
        self.max_action = config.get('max_action', TD3_BC_CONFIG['max_action'])
        self.min_action = config.get('min_action', TD3_BC_CONFIG['min_action'])
        self.exploration_noise = config.get('exploration_noise', TD3_BC_CONFIG['exploration_noise'])
        self.actor_lr = config.get('actor_lr', TD3_BC_CONFIG['actor_lr'])
        self.critic_lr = config.get('critic_lr', TD3_BC_CONFIG['critic_lr'])
        self.buffer_size = config.get('buffer_size', TD3_BC_CONFIG['buffer_size'])
        
        # Inicializar redes
        self.actor = TD3Actor(cgm_input_dim, other_input_dim, self.action_dim, self.hidden_dim, self.max_action)
        self.actor_target = TD3Actor(cgm_input_dim, other_input_dim, self.action_dim, self.hidden_dim, self.max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Twin critics (dos críticos para reducir sobreestimación)
        self.critic1 = TD3Critic(cgm_input_dim, other_input_dim, self.action_dim, self.hidden_dim)
        self.critic2 = TD3Critic(cgm_input_dim, other_input_dim, self.action_dim, self.hidden_dim)
        
        self.critic1_target = TD3Critic(cgm_input_dim, other_input_dim, self.action_dim, self.hidden_dim)
        self.critic2_target = TD3Critic(cgm_input_dim, other_input_dim, self.action_dim, self.hidden_dim)
        
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Optimizadores
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr, weight_decay=1e-5)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.critic_lr, weight_decay=1e-5)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.critic_lr, weight_decay=1e-5)
        
        # Buffer de experiencia
        self.buffer = ReplayBuffer(self.buffer_size, cgm_input_dim, other_input_dim, self.action_dim, self.rng)
        
        # Función de recompensa
        self.compute_rewards = rewards_function
        
        # Contador para actualización retrasada del actor
        self.update_counter = 0
        
        # Enviar redes al dispositivo correcto (CPU/GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
        # Inicializar con pesos adecuados
        self._initialize_networks()
        
        print_info("TD3+BC Model iniciado correctamente")
    
    def _initialize_networks(self) -> None:
        """
        Inicializa las redes con pesos que producen salidas razonables desde el inicio.
        """
        # Inicialización especial para la última capa del actor para generar valores no nulos
        if hasattr(self.actor, 'action_head'):
            # Initialize action_head to produce non-zero outputs
            nn.init.uniform_(self.actor.action_head.weight, -0.003, 0.003)
            nn.init.uniform_(self.actor.action_head.bias, 0.1, 0.3)  # Positive bias to start
            print_debug("Initialized actor head with special weights")
        
        # Inicialización especial para la última capa del crítico
        for critic in [self.critic1, self.critic2]:
            if hasattr(critic, 'combined_layer'):
                last_layer = critic.combined_layer[-1]
                if isinstance(last_layer, nn.Linear):
                    nn.init.uniform_(last_layer.weight, -0.003, 0.003)
                    nn.init.uniform_(last_layer.bias, -0.1, 0.1)
                    print_debug("Initialized critic output layer with special weights")
        
        # Asegurar que los críticos target tengan los mismos pesos
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
    
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
        return self.actor(x_cgm, x_other)
    
    def select_action(self, x_cgm: torch.Tensor, x_other: torch.Tensor, 
                  add_noise: bool = True) -> torch.Tensor:
        """
        Selecciona una acción basada en el estado actual.
        
        Parámetros:
        -----------
        x_cgm : torch.Tensor
            Datos CGM de entrada
        x_other : torch.Tensor
            Otras características de entrada
        add_noise : bool, opcional
            Si agregar ruido de exploración (default: True)
        
        Retorna:
        --------
        torch.Tensor
            Acción seleccionada
        """
        # Asegurar modo de evaluación
        self.actor.eval()
        
        # Aplanar tensores si es necesario
        if len(x_cgm.shape) > 2:
            x_cgm = x_cgm.reshape(x_cgm.shape[0], -1)
            print_debug(f"Reshaping x_cgm to {x_cgm.shape}")
        if len(x_other.shape) > 2:
            x_other = x_other.reshape(x_other.shape[0], -1)
            print_debug(f"Reshaping x_other to {x_other.shape}")
        
        # Obtener acción del actor
        with torch.no_grad():
            action = self.actor(x_cgm, x_other)
            
            # Agregar ruido de exploración si está habilitado
            if add_noise:
                noise = torch.randn_like(action) * self.exploration_noise
                action = action + noise
            
            # Recortar a los límites de acción
            action = torch.clamp(action, self.min_action, self.max_action)
            print_debug(f"TD3+BC action: {action.cpu().numpy()}")
        
        # Volver a modo de entrenamiento
        self.actor.train()
        
        return action
    
    def update(self, batch: Tuple) -> Dict[str, float]:
        """
        Actualiza las redes de actor y crítico usando un batch de experiencias.
        
        Parámetros:
        -----------
        batch : Tuple
            Batch de experiencias (states, actions, rewards, next_states, dones)
            
        Retorna:
        --------
        Dict[str, float]
            Diccionario con pérdidas de actor y crítico
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
        
        # ===== Actualización de los críticos =====
        with torch.no_grad():
            # Seleccionar acción del siguiente estado con el actor target
            next_actions = self.actor_target(next_states_cgm_t, next_states_other_t)
            
            # Agregar ruido con recorte para target policy smoothing
            noise = torch.randn_like(next_actions) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_actions = torch.clamp(next_actions + noise, self.min_action, self.max_action)
            
            # Calcular el Q mínimo entre los dos críticos target
            q1_next = self.critic1_target(next_states_cgm_t, next_states_other_t, next_actions)
            q2_next = self.critic2_target(next_states_cgm_t, next_states_other_t, next_actions)
            q_next = torch.min(q1_next, q2_next)
            
            # Calcular el objetivo Q usando la fórmula de Bellman
            q_target = rewards_t + (1 - dones_t) * self.gamma * q_next
        
        # Calcular pérdida del primer crítico
        q1 = self.critic1(states_cgm_t, states_other_t, actions_t)
        critic1_loss = F.mse_loss(q1, q_target)
        
        # Actualizar primer crítico
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optimizer.step()
        
        # Calcular pérdida del segundo crítico
        q2 = self.critic2(states_cgm_t, states_other_t, actions_t)
        critic2_loss = F.mse_loss(q2, q_target)
        
        # Actualizar segundo crítico
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optimizer.step()
        
        # ===== Actualización del actor (retrasada) =====
        actor_loss = torch.tensor(0.0, device=self.device)
        
        # Actualizar el actor solo cada policy_delay pasos
        if self.update_counter % self.policy_delay == 0:
            # Acciones generadas por el actor actual
            pi = self.actor(states_cgm_t, states_other_t)
            
            # Componente de RL: maximizar Q-value
            q_pi = self.critic1(states_cgm_t, states_other_t, pi)
            lmbda = self.alpha / q_pi.abs().mean().detach()
            
            # TD3+BC: combinar pérdida de RL con pérdida de BC
            actor_loss = -lmbda * q_pi.mean() + F.mse_loss(pi, actions_t)
            
            # Actualizar actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            
            # Actualización suave de redes target
            self._update_target_networks()
        
        self.update_counter += 1
        
        return {
            'actor_loss': actor_loss.item(),
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'total_loss': critic1_loss.item() + critic2_loss.item() + actor_loss.item()
        }
    
    def _update_target_networks(self) -> None:
        """
        Actualiza las redes target usando actualización suave (soft update).
        """
        # Actualizar actor target
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        # Actualizar críticos target
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
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

    def _apply_carb_corrections(self, pred_numpy: np.ndarray, carbs: float) -> np.ndarray:
        """
        Aplica correcciones basadas en carbohidratos a las predicciones.
        
        Parámetros:
        -----------
        pred_numpy : np.ndarray
            Predicciones originales
        carbs : float
            Ingesta de carbohidratos
            
        Retorna:
        --------
        np.ndarray
            Predicciones corregidas
        """
        if np.isnan(pred_numpy).any() or np.all(pred_numpy == 0):
            print_warning("Se detectaron NaN o ceros en las predicciones. Aplicando corrección basada en carbohidratos.")
            
            # Aplicar una regla básica: 1U por cada 10g de carbohidratos
            fallback_dose = carbs / 10.0
            
            # Si la predicción es cero, usar fallback
            if np.all(pred_numpy == 0):
                return np.full_like(pred_numpy, fallback_dose)
            else:
                # Si hay algunos NaN, reemplazarlos manteniendo valores válidos
                return np.nan_to_num(pred_numpy, nan=fallback_dose)
        
        # Incluso con predicciones válidas, asegurar dosis mínima para comidas
        if carbs > 20 and pred_numpy[0] < carbs / 30.0:
            print_debug(f"Ajustando dosis demasiado baja: {pred_numpy[0]} para {carbs}g de carbohidratos")
            min_dose = carbs / 30.0  # Ratio conservador de 1:30
            pred_numpy[0] = max(pred_numpy[0], min_dose)
            
        return pred_numpy

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
        
        # Predecir sin ruido
        with torch.no_grad():
            action = self.select_action(x_cgm_tensor, x_other_tensor, add_noise=False)
        
        # Convertir a numpy
        pred_numpy = action.cpu().numpy()
        
        # Extraer valor de carbohidratos
        carbs = float(x_other[0, 0]) if x_other.shape[1] > 0 else 0.0
        
        # Aplicar correcciones basadas en carbohidratos
        pred_numpy = self._apply_carb_corrections(pred_numpy, carbs)
        
        print_debug(f"Dosis final TD3+BC: {pred_numpy}, para carbohidratos: {carbs}")
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
        print_critical("Using TD3+BC's predict_with_context - Direct from nn.Module")
        
        if carb_intake <= 0.0:
            print_critical("Ingesta de carbohidratos es 0.0, no se recomienda dosis de insulina.")
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
        
        print_debug(f"Predicción inicial: {prediction}, glucosa actual: {current_glucose}, IOB: {iob}, ingesta de carbohidratos: {carb_intake}")

        # Extraer valor de predicción
        prediction_value = float(prediction.item() if hasattr(prediction, 'item') else prediction[0])
        
        # Aplicar ajustes
        prediction_value = self._adjust_for_glucose_level(prediction_value, current_glucose)
        prediction_value = self._adjust_for_carbs_and_iob(prediction_value, carb_intake, iob)

        # Asegurar límites seguros
        return max(0.0, min(prediction_value, self.max_action))
    
    def to(self, device: torch.device) -> 'TD3BCModel':
        """
        Mueve el modelo al dispositivo especificado.
        
        Parámetros:
        -----------
        device : torch.device
            Dispositivo de destino
            
        Retorna:
        --------
        TD3BCModel
            Self para encadenamiento
        """
        self.device = device
        self.actor = self.actor.to(device)
        self.actor_target = self.actor_target.to(device)
        self.critic1 = self.critic1.to(device)
        self.critic2 = self.critic2.to(device)
        self.critic1_target = self.critic1_target.to(device)
        self.critic2_target = self.critic2_target.to(device)
        return self


def create_td3_bc_model(cgm_input_dim: tuple, other_input_dim: tuple) -> DRLModelWrapperPyTorch:
    """
    Crea un modelo TD3+BC para dosificación de insulina.
    
    Parámetros:
    -----------
    cgm_input_dim : tuple
        Dimensiones de entrada para datos CGM
    other_input_dim : tuple
        Dimensiones de entrada para otras características
        
    Retorna:
    --------
    DRLModelWrapperPyTorch
        Modelo TD3+BC inicializado envuelto en DRLModelWrapperPyTorch
    """
    from training.utils import compute_reward
    
    model = TD3BCModel(
        cgm_input_dim=cgm_input_dim,
        other_input_dim=other_input_dim,
        config=TD3_BC_CONFIG,
        rewards_function=compute_reward
    )
    
    print_critical(f"Tipo de modelo creado: {type(model)}")
    
    wrapper = DRLModelWrapperPyTorch(model, algorithm="TD3+BC")
    print_critical(f"Tipo de wrapper: {type(wrapper)}")
    print_critical(f"Tipo de modelo en wrapper: {type(wrapper.model)}")
    
    return wrapper