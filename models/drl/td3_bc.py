"""
Implementación de Twin Delayed DDPG with Behavior Cloning (TD3+BC) para dosificación de insulina.

TD3+BC es un algoritmo de aprendizaje por refuerzo profundo offline que combina:
1. TD3: Usa dos redes críticas para reducir la sobreestimación del valor Q
2. Actualizaciones retrasadas del actor para mayor estabilidad
3. Behavior Cloning para mejorar el aprendizaje a partir de datos históricos
4. Regularización para evitar extrapolaciones fuera de la distribución de datos
"""

import copy
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
    CONST_ACTOR_LOSS, CONST_CRITIC_LOSS, CONST_DEFAULT_SEED, CONST_EPSILON, CONTEXT_FEATURE_ORDER, IDEAL_LOWER_BOUND, IDEAL_UPPER_BOUND, SEVERE_HYPOGLYCEMIA_THRESHOLD, HYPOGLYCEMIA_THRESHOLD, 
    HYPERGLYCEMIA_THRESHOLD, SEVERE_HYPERGLYCEMIA_THRESHOLD, SEVERE_HYPO_PENALTY, HYPO_PENALTY_BASE, 
    HYPER_PENALTY_BASE, SEVERE_HYPER_PENALTY, MAX_REWARD
)
from config.models_config import TD3_BC_CONFIG
from training.utils import compute_reward, calculate_iob
from validation.simulator import GlucoseSimulator
from models.utils.replay_buffer import ReplayBuffer


class TD3Actor(nn.Module):
    """
    Red del actor para TD3+BC que determina la acción óptima para un estado dado.
    
    Parámetros:
    -----------
    cgm_input_dim : tuple
        Dimensiones de entrada para datos CGM. Ejemplo: (timesteps, cgm_features) o (flat_cgm_dim,).
    other_input_dim : tuple
        Dimensiones de entrada para otras características. Ejemplo: (other_features_len,).
    action_dim : int
        Dimensión de la acción (dosis de insulina).
    hidden_dim : int
        Dimensión de las capas ocultas.
    max_action : float
        Valor máximo de acción permitido.
    """
    
    def __init__(self, cgm_input_dim: tuple, other_input_dim: tuple, 
                 action_dim: int = TD3_BC_CONFIG.get('action_dim', 1), 
                 hidden_dim: int = TD3_BC_CONFIG.get('hidden_dim', 256), 
                 max_action: float = TD3_BC_CONFIG.get('max_action', 10.0)) -> None:
        super().__init__()
        
        self.flat_cgm_dim = np.prod(cgm_input_dim) if cgm_input_dim and np.prod(cgm_input_dim) > 0 else 0
        self.flat_other_dim = np.prod(other_input_dim) if other_input_dim and np.prod(other_input_dim) > 0 else 0
        total_input_dim = self.flat_cgm_dim + self.flat_other_dim

        if total_input_dim == 0:
            print_warning("TD3Actor: total_input_dim es 0. El modelo podría no funcionar correctamente.")
            # Se podría considerar levantar un error si total_input_dim es 0 y no es un caso esperado.
            # Por ahora, se permite para que la inicialización no falle, pero las capas lineales fallarán si se usan.
            # Alternativamente, definir capas dummy o no definir capas si total_input_dim es 0.
            # Para este ejemplo, asumimos que total_input_dim > 0 si el actor se usa.
            
        self.l1 = nn.Linear(total_input_dim, hidden_dim) if total_input_dim > 0 else nn.Identity() # Usar Identity si no hay entrada
        self.l2 = nn.Linear(hidden_dim, hidden_dim) if total_input_dim > 0 else nn.Identity()
        self.l3 = nn.Linear(hidden_dim, action_dim) if total_input_dim > 0 else nn.Identity()
        
        self.max_action = max_action
        
    def forward(self, x_cgm: torch.Tensor, x_other: torch.Tensor) -> torch.Tensor:
        """
        Paso hacia adelante de la red del actor.

        Parámetros:
        -----------
        x_cgm : torch.Tensor
            Tensor de datos CGM. Forma: (batch_size, *cgm_input_dim).
        x_other : torch.Tensor
            Tensor de otras características. Forma: (batch_size, *other_input_dim).

        Retorna:
        --------
        torch.Tensor
            Acción predicha. Forma: (batch_size, action_dim).
        """
        inputs = []
        if self.flat_cgm_dim > 0:
            if x_cgm.shape[0] == 0 : # Si el batch es vacío
                 # Devolver un tensor vacío con la forma correcta de salida si es posible
                return torch.empty((0, self.l3.out_features if hasattr(self.l3, 'out_features') else 1), device=x_cgm.device)

            inputs.append(x_cgm.reshape(x_cgm.shape[0], -1))
        
        if self.flat_other_dim > 0:
            if x_other.shape[0] == 0: # Si el batch es vacío
                return torch.empty((0, self.l3.out_features if hasattr(self.l3, 'out_features') else 1), device=x_other.device)
            inputs.append(x_other.reshape(x_other.shape[0], -1))

        if not inputs: # Si no hay características de entrada
             # Esto podría suceder si flat_cgm_dim y flat_other_dim son 0.
             # Devolver una acción por defecto o manejar el error.
             # Si l3 es Identity, necesitaríamos saber action_dim para crear un tensor de ceros.
             # Asumiendo que action_dim es conocido (ej. self.l3.out_features si l3 es Linear)
            action_dim_val = self.l3.out_features if hasattr(self.l3, 'out_features') and not isinstance(self.l3, nn.Identity) else TD3_BC_CONFIG.get('action_dim', 1)
            # Necesitamos una forma de obtener el device si los inputs son vacíos.
            # Por ahora, si no hay inputs, es difícil determinar el device o batch_size.
            # Este caso debería ser prevenido por una validación anterior de dimensiones de entrada.
            print_warning("TD3Actor.forward: No hay datos de entrada (CGM u Otros).")
            # Devolver un tensor vacío con la forma correcta si es posible.
            # Esto es problemático si no hay forma de determinar batch_size.
            # Si se llega aquí, es probable que haya un problema en la preparación de datos.
            # Para evitar un error inmediato, si l3 es Identity, intentamos devolver algo,
            # pero esto es una curita.
            if isinstance(self.l3, nn.Identity): # No se puede determinar out_features
                 # Devolver un tensor vacío con la dimensión de acción esperada, pero batch_size 0
                return torch.empty((0, action_dim_val)) # device?
            # Si l3 es Linear, podemos usar out_features
            return torch.empty((0, self.l3.out_features))


        x = torch.cat(inputs, dim=1)
        
        a = F.relu(self.l1(x))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class TD3Critic(nn.Module):
    """
    Red del crítico para TD3+BC que evalúa la calidad de pares estado-acción.
    
    Parámetros:
    -----------
    cgm_input_dim : tuple
        Dimensiones de entrada para datos CGM.
    other_input_dim : tuple
        Dimensiones de entrada para otras características.
    action_dim : int
        Dimensión de la acción (dosis de insulina).
    hidden_dim : int
        Dimensión de las capas ocultas.
    """
    
    def __init__(self, cgm_input_dim: tuple, other_input_dim: tuple, 
                 action_dim: int = TD3_BC_CONFIG.get('action_dim', 1), 
                 hidden_dim: int = TD3_BC_CONFIG.get('hidden_dim', 256)) -> None:
        super().__init__()
        
        self.flat_cgm_dim = np.prod(cgm_input_dim) if cgm_input_dim and np.prod(cgm_input_dim) > 0 else 0
        self.flat_other_dim = np.prod(other_input_dim) if other_input_dim and np.prod(other_input_dim) > 0 else 0
        total_input_dim = self.flat_cgm_dim + self.flat_other_dim

        if total_input_dim == 0:
            print_warning("TD3Critic: total_input_dim es 0. El modelo podría no funcionar correctamente.")
            # Similar al Actor, manejar el caso de entrada 0.
            # Usaremos nn.Identity() para las capas si total_input_dim es 0.
            # La capa de salida será un escalar (valor Q).

        # Q1 architecture
        self.l1 = nn.Linear(total_input_dim + action_dim, hidden_dim) if total_input_dim > 0 or action_dim > 0 else nn.Identity()
        self.l2 = nn.Linear(hidden_dim, hidden_dim) if total_input_dim > 0 or action_dim > 0 else nn.Identity()
        self.l3 = nn.Linear(hidden_dim, 1) if total_input_dim > 0 or action_dim > 0 else nn.Identity()

        # Q2 architecture
        self.l4 = nn.Linear(total_input_dim + action_dim, hidden_dim) if total_input_dim > 0 or action_dim > 0 else nn.Identity()
        self.l5 = nn.Linear(hidden_dim, hidden_dim) if total_input_dim > 0 or action_dim > 0 else nn.Identity()
        self.l6 = nn.Linear(hidden_dim, 1) if total_input_dim > 0 or action_dim > 0 else nn.Identity()
        
    def forward(self, x_cgm: torch.Tensor, x_other: torch.Tensor, 
                action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Paso hacia adelante de las redes del crítico.

        Parámetros:
        -----------
        x_cgm : torch.Tensor
            Tensor de datos CGM.
        x_other : torch.Tensor
            Tensor de otras características.
        action : torch.Tensor
            Tensor de acciones.

        Retorna:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            Valores Q estimados por los dos críticos (Q1, Q2).
        """
        inputs = []
        # Manejar batch vacío para CGM
        if self.flat_cgm_dim > 0:
            if x_cgm.shape[0] == 0: # Batch vacío
                return torch.empty((0, 1), device=x_cgm.device), torch.empty((0, 1), device=x_cgm.device)
            inputs.append(x_cgm.reshape(x_cgm.shape[0], -1))
        
        # Manejar batch vacío para Other
        if self.flat_other_dim > 0:
            if x_other.shape[0] == 0: # Batch vacío
                 return torch.empty((0, 1), device=x_other.device), torch.empty((0, 1), device=x_other.device)
            inputs.append(x_other.reshape(x_other.shape[0], -1))

        # Manejar batch vacío para Action
        if action.shape[0] == 0 and (self.flat_cgm_dim == 0 and self.flat_other_dim == 0) : # Si no hay estado y la acción es vacía
            return torch.empty((0, 1), device=action.device), torch.empty((0, 1), device=action.device)
        
        # Si no hay características de estado pero sí acción (poco probable pero posible)
        if not inputs and action.shape[0] > 0:
            sa1 = action.reshape(action.shape[0], -1) # Solo acción
        elif not inputs and action.shape[0] == 0: # No hay estado ni acción
            print_warning("TD3Critic.forward: No hay datos de entrada (CGM, Otros, o Acción).")
            return torch.empty((0, 1)), torch.empty((0, 1)) # device?
        else: # Hay características de estado
            state_features = torch.cat(inputs, dim=1)
            sa1 = torch.cat([state_features, action.reshape(action.shape[0], -1)], 1)

        q1 = F.relu(self.l1(sa1))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa1)) # Reutilizar sa1 ya que las entradas son las mismas
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2
class TD3BCModel(nn.Module):
    """
    Implementación de Twin Delayed DDPG with Behavior Cloning (TD3+BC) para dosificación de insulina.
    
    Parámetros:
    -----------
    cgm_input_dim : tuple
        Dimensiones de entrada para datos CGM (ej: (timesteps, num_features_per_timestep)).
        Estas son las dimensiones *antes* de cualquier aplanamiento que el actor/crítico interno pueda hacer.
    other_input_dim : tuple
        Dimensiones de entrada para otras características (ej: (total_other_features_len,)).
        Estas son las dimensiones *antes* de cualquier aplanamiento. Incluye características contextuales explícitas.
    config : Dict[str, Any]
        Configuración del modelo TD3+BC.
    rewards_function : Optional[Callable]
        Función para calcular recompensas.
    """
    
    def __init__(self,
                 cgm_input_dim: tuple, 
                 other_input_dim: tuple,
                 config: Dict[str, Any] = TD3_BC_CONFIG,
                 rewards_function: Optional[Callable] = None) -> None:
        super().__init__()
        
        self.cgm_input_dim = cgm_input_dim
        self.other_input_dim = other_input_dim # Ya incluye características de contexto explícitas
        self.config = config
        self.rewards_function = rewards_function if rewards_function is not None else compute_reward
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Calcular dimensiones aplanadas para referencia, si es necesario para alguna lógica interna.
        # Los actores/críticos usarán cgm_input_dim y other_input_dim directamente.
        flat_cgm_dim = np.prod(self.cgm_input_dim) if self.cgm_input_dim and np.prod(self.cgm_input_dim) > 0 else 0
        # other_input_dim ya es la dimensión aplanada de "otras" características (incluyendo contexto)
        flat_other_dim = np.prod(self.other_input_dim) if self.other_input_dim and np.prod(self.other_input_dim) > 0 else 0
        
        # Esta es la variable que causaba el UnboundLocalError.
        # Se usa para una verificación, pero no necesariamente como la state_dim unificada para el actor/crítico
        # si estos manejan las entradas CGM y Otras por separado.
        total_state_dim_for_check = flat_cgm_dim + flat_other_dim 

        if total_state_dim_for_check == 0: # Esta es la línea de la traza de error
            print_critical("TD3BCModel: La dimensión total del estado (CGM aplanado + Otras aplanado) es 0. "
                           "Esto indica que no se proporcionaron características CGM ni Otras. "
                           "El modelo no podrá aprender. Verifique la configuración de características.")
            # Considerar levantar un ValueError aquí, ya que un modelo sin entradas no tiene sentido.
            # raise ValueError("La dimensión total del estado no puede ser cero.")

        self.action_dim = self.config.get('action_dim', 1)
        self.max_action = self.config.get('max_action', 10.0)
        self.min_action = self.config.get('min_action', 0.0) # Asegurar que min_action esté en config

        self._initialize_networks() # Llama al método para crear las redes
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.get('actor_lr', 3e-4))
        self.critic_optimizer = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=self.config.get('critic_lr', 3e-4))

        self.total_it = 0
        self.to(self.device)

    def _initialize_networks(self) -> None:
        """Inicializa las redes del actor y crítico (y sus objetivos)."""
        hidden_dim = self.config.get('hidden_dim', 256)
        
        # El actor y los críticos reciben cgm_input_dim y other_input_dim (que ya incluye contexto)
        self.actor = TD3Actor(self.cgm_input_dim, self.other_input_dim, self.action_dim, hidden_dim, self.max_action).to(self.device)
        self.target_actor = copy.deepcopy(self.actor)
        
        self.critic1 = TD3Critic(self.cgm_input_dim, self.other_input_dim, self.action_dim, hidden_dim).to(self.device)
        self.target_critic1 = copy.deepcopy(self.critic1)
        
        self.critic2 = TD3Critic(self.cgm_input_dim, self.other_input_dim, self.action_dim, hidden_dim).to(self.device)
        self.target_critic2 = copy.deepcopy(self.critic2)
        
        print_info(f"Redes TD3+BC inicializadas en {self.device}.")
        print_debug(f"  Actor CGM dim: {self.cgm_input_dim}, Other dim: {self.other_input_dim}, Action dim: {self.action_dim}")

    def forward(self, x_cgm: torch.Tensor, x_other: torch.Tensor) -> torch.Tensor:
        """
        Paso hacia adelante del modelo TD3BC (usado principalmente para inferencia).
        Retorna la acción determinada por el actor.

        Parámetros:
        -----------
        x_cgm : torch.Tensor
            Tensor de datos CGM.
        x_other : torch.Tensor
            Tensor de otras características (incluyendo contexto).

        Retorna:
        --------
        torch.Tensor
            Acción predicha.
        """
        return self.actor(x_cgm, x_other)
    
    def select_action(self, x_cgm: torch.Tensor, x_other: torch.Tensor, 
                      add_noise: bool = True) -> torch.Tensor:
        """
        Selecciona una acción basada en el estado actual, opcionalmente añadiendo ruido.
        Los tensores de entrada deben estar en el dispositivo correcto.

        Parámetros:
        -----------
        x_cgm : torch.Tensor
            Tensor de datos CGM (ya en self.device).
        x_other : torch.Tensor
            Tensor de otras características (incluyendo contexto, ya en self.device).
        add_noise : bool, opcional
            Si añadir ruido de exploración (default: True).

        Retorna:
        --------
        torch.Tensor
            Acción seleccionada.
        """
        self.actor.eval() # Modo evaluación para selección de acción
        with torch.no_grad():
            action = self.actor(x_cgm, x_other)
        self.actor.train() # Volver a modo entrenamiento

        if add_noise:
            noise_std = self.config.get('exploration_noise', 0.1) * self.max_action
            noise = (torch.randn_like(action) * noise_std).to(self.device)
            action = (action + noise).clamp(self.min_action, self.max_action)
        else:
            action = action.clamp(self.min_action, self.max_action)
            
        return action

    def update(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """
        Realiza una actualización de los parámetros del modelo TD3+BC.

        Parámetros:
        -----------
        batch : Tuple[torch.Tensor, ...]
            Un batch de transiciones:
            (x_cgm, x_other, actions, rewards, next_x_cgm, next_x_other, dones)
            Todos los tensores deben estar en self.device.

        Retorna:
        --------
        Dict[str, float]
            Diccionario con las pérdidas del actor y crítico.
        """
        self.total_it += 1
        
        x_cgm, x_other, actions, rewards, next_x_cgm, next_x_other, dones = batch

        with torch.no_grad():
            # Ruido para regularización de la política objetivo
            policy_noise_std = self.config.get('policy_noise', 0.2) * self.max_action
            noise_clip_val = self.config.get('noise_clip', 0.5) * self.max_action
            
            noise = (torch.randn_like(actions) * policy_noise_std).clamp(-noise_clip_val, noise_clip_val)
            
            next_actions_target = (self.target_actor(next_x_cgm, next_x_other) + noise).clamp(self.min_action, self.max_action)

            # Calcular el valor Q objetivo de los dos críticos objetivo
            q1_target_next, q2_target_next = self.target_critic1(next_x_cgm, next_x_other, next_actions_target), \
                                             self.target_critic2(next_x_cgm, next_x_other, next_actions_target)
            q_target_next = torch.min(q1_target_next, q2_target_next)
            
            # TD target
            y_target = rewards + (self.config['gamma'] * q_target_next * (1.0 - dones.float()))

        # --- Actualización del Crítico ---
        current_q1, current_q2 = self.critic1(x_cgm, x_other, actions), self.critic2(x_cgm, x_other, actions)
        
        critic_loss = F.mse_loss(current_q1, y_target) + F.mse_loss(current_q2, y_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_loss_val = torch.tensor(0.0) # Valor por defecto si el actor no se actualiza

        # --- Actualización Retrasada del Actor y Redes Objetivo ---
        if self.total_it % self.config.get('policy_delay', 2) == 0:
            # Pérdida del actor
            actor_actions = self.actor(x_cgm, x_other)
            q_actor = self.critic1(x_cgm, x_other, actor_actions) # Usar critic1 para la pérdida del actor
            
            # Pérdida de Behavior Cloning (BC)
            # lambda_bc = alpha / (promedio de Q_actor) --> alpha es self.config['alpha']
            # El paper original de TD3+BC usa: alpha / (|mean(Q_values_for_actor_loss)| / N)
            # donde N es el tamaño del batch.
            # Simplificado: lambda_bc = alpha / mean(|Q_actor|)
            lambda_bc = self.config.get('alpha', 2.5) / (q_actor.abs().mean().detach() + 1e-3) # +1e-3 para evitar división por cero
            
            # La pérdida de BC es MSE entre la acción del actor y la acción del batch
            bc_loss = F.mse_loss(actor_actions, actions)
            
            actor_loss = -lambda_bc * q_actor.mean() + bc_loss
            actor_loss_val = actor_loss.item() # Guardar para log
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Actualización suave de las redes objetivo
            self._update_target_networks()
            
        return {
            CONST_CRITIC_LOSS: critic_loss.item(),
            CONST_ACTOR_LOSS: actor_loss_val.item() if isinstance(actor_loss_val, torch.Tensor) else actor_loss_val
        }

    def _update_target_networks(self) -> None:
        """Actualización suave de los parámetros de las redes objetivo."""
        tau = self.config.get('tau', 0.005)
        
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
            
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

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

    def _prepare_predict_tensors(self, 
                                 x_cgm: np.ndarray, 
                                 x_other_base: np.ndarray, # x_other sin contexto explícito
                                 current_glucose: float,
                                 carb_intake: float,
                                 iob: float,
                                 sleep_quality: Optional[float],
                                 work_intensity: Optional[float],
                                 exercise_intensity: Optional[float]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepara los tensores de entrada para predict_with_context, asegurando que x_other
        se construya correctamente con las características de contexto.
        """
        # Construir el array de contexto explícito
        context_values_list = [
            current_glucose,
            carb_intake,
            iob,
            sleep_quality if sleep_quality is not None else 0.0,
            work_intensity if work_intensity is not None else 0.0,
            exercise_intensity if exercise_intensity is not None else 0.0
        ]
        # Asegurar que el orden coincida con CONTEXT_FEATURE_ORDER si se usa directamente
        # Aquí asumimos que other_input_dim ya fue definido para incluir estas características en orden.
        context_np = np.array(context_values_list, dtype=np.float32)

        # Concatenar x_other_base (características no contextuales) con context_np
        # Se asume que x_other_base es (num_other_non_context_features,)
        # y context_np es (num_context_features,)
        # El other_input_dim para el modelo debe ser (num_other_non_context_features + num_context_features,)
        
        # Si x_other_base está vacío, x_other_full es solo context_np
        if x_other_base.size == 0:
            x_other_full_np = context_np
        else:
            x_other_full_np = np.concatenate((x_other_base, context_np))

        # Validar que la dimensión de x_other_full_np coincida con self.other_input_dim[0]
        expected_other_dim = np.prod(self.other_input_dim)
        if x_other_full_np.shape[0] != expected_other_dim:
            print_critical(f"Dimensión de x_other_full_np ({x_other_full_np.shape[0]}) no coincide con la esperada "
                           f"por el modelo other_input_dim ({expected_other_dim}). "
                           f"x_other_base: {x_other_base.shape}, context_np: {context_np.shape}")
            # Esto podría indicar un desajuste en cómo se define/usa other_input_dim
            # o cómo se construye x_other_base.
            # Por ahora, se procederá, pero es probable que falle en el forward del actor/crítico.

        # Añadir dimensión de batch y convertir a tensores
        x_cgm_tensor = torch.tensor(x_cgm[np.newaxis, ...], dtype=torch.float32).to(self.device)
        x_other_tensor = torch.tensor(x_other_full_np[np.newaxis, ...], dtype=torch.float32).to(self.device)
        
        return x_cgm_tensor, x_other_tensor

    def predict_with_context(self, 
                             x_cgm: np.ndarray, # Ventana CGM (ej: (timesteps, cgm_features) o (flat_cgm_dim,))
                             x_other: np.ndarray, # Otras características NO contextuales (ej: (other_non_context_len,))
                             current_glucose: float,
                             carb_intake: float,
                             iob: float,
                             sleep_quality: Optional[float] = None,
                             work_intensity: Optional[float] = None,
                             exercise_intensity: Optional[float] = None,
                             stress_level: Optional[float] = None, # No usado actualmente en CONTEXT_FEATURE_ORDER
                             target_glucose: Optional[float] = None, # No usado por TD3 para acción
                             **kwargs: Any                             
                             ) -> float:
        """
        Realiza una predicción de dosis de insulina basada en el estado actual y contexto.
        x_cgm y x_other deben ser arrays NumPy.
        x_other aquí se refiere a las características que NO son parte del contexto explícito
        pasado como argumentos separados (current_glucose, carb_intake, etc.).
        """
        self.actor.eval() # Asegurar modo de evaluación

        # Preparar tensores de entrada. x_other aquí es la parte "base" de otras características.
        # _prepare_predict_tensors construirá el tensor x_other completo.
        x_cgm_tensor, x_other_tensor = self._prepare_predict_tensors(
            x_cgm, x_other, 
            current_glucose, carb_intake, iob, 
            sleep_quality, work_intensity, exercise_intensity
        )
        
        with torch.no_grad():
            action_tensor = self.actor(x_cgm_tensor, x_other_tensor) # No añadir ruido para predicción determinística
        
        # Aplicar clamp a la acción predicha
        action_value = action_tensor.clamp(self.min_action, self.max_action).item()
        
        return action_value
    
    def predict(self, x_cgm: np.ndarray, x_other: np.ndarray) -> np.ndarray:
        """
        Predicción simplificada. Se recomienda usar `predict_with_context` para predicciones en producción/evaluación.
        Esta función asume que `x_other` ya contiene todas las características necesarias,
        incluyendo las contextuales, en el formato correcto.
        """
        self._validate_inputs(x_cgm, x_other) # x_other aquí debe ser el combinado
        x_cgm_tensor, x_other_tensor = self._prepare_tensors(x_cgm, x_other)
        
        # Permutar CGM si es necesario para Conv1d, similar a _prepare_predict_tensors
        # (Batch, Features, Timesteps)
        if x_cgm_tensor.ndim == 3 and x_cgm_tensor.shape[1] == self.cgm_input_dim[0] and x_cgm_tensor.shape[2] == self.cgm_input_dim[1]:
             x_cgm_tensor = x_cgm_tensor.permute(0, 2, 1)

        self.actor.eval()
        with torch.no_grad():
            predictions_tensor = self.actor(x_cgm_tensor, x_other_tensor)
        self.actor.train()
        
        pred_numpy = predictions_tensor.cpu().numpy()
        return pred_numpy.flatten()
    
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
    
    def _update_target_networks(self) -> None:
        """
        Realiza una actualización suave (soft update) de las redes objetivo.
        """
        # Actualización suave del actor
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Actualización suave del crítico 1
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Actualización suave del crítico 2
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def run_training_step(self, replay_buffer: ReplayBuffer, batch_size: int) -> Dict[str, float]:
        """
        Ejecuta un paso de entrenamiento para el modelo TD3+BC.

        Parámetros:
        -----------
        replay_buffer : ReplayBuffer
            Buffer de repetición del cual muestrear transiciones.
        batch_size : int
            Tamaño del minibatch a muestrear.

        Retorna:
        --------
        Dict[str, float]
            Diccionario con las pérdidas del actor y los críticos.
        """
        if len(replay_buffer) < batch_size:
            # No hay suficientes muestras en el buffer para formar un batch completo
            return {CONST_ACTOR_LOSS: 0.0, "critic1_loss": 0.0, "critic2_loss": 0.0}

        state_np, action_np, next_state_np, reward_np, not_done_np = replay_buffer.sample(batch_size)

        # ---- INICIO: Validación de datos del batch ----
        if np.isnan(state_np).any() or np.isinf(state_np).any():
            print_critical("NaN/Inf detectado en 'state_np' del batch de replay_buffer")
            return {CONST_ACTOR_LOSS: float('nan'), "critic1_loss": float('nan'), "critic2_loss": float('nan')}
        if np.isnan(action_np).any() or np.isinf(action_np).any():
            print_critical("NaN/Inf detectado en 'action_np' del batch de replay_buffer")
            return {CONST_ACTOR_LOSS: float('nan'), "critic1_loss": float('nan'), "critic2_loss": float('nan')}
        if np.isnan(next_state_np).any() or np.isinf(next_state_np).any():
            print_critical("NaN/Inf detectado en 'next_state_np' del batch de replay_buffer")
            return {CONST_ACTOR_LOSS: float('nan'), "critic1_loss": float('nan'), "critic2_loss": float('nan')}
        if np.isnan(reward_np).any() or np.isinf(reward_np).any():
            print_critical("NaN/Inf detectado en 'reward_np' del batch de replay_buffer")
            return {CONST_ACTOR_LOSS: float('nan'), "critic1_loss": float('nan'), "critic2_loss": float('nan')}
        # ---- FIN: Validación de datos del batch ----

        state = torch.FloatTensor(state_np).to(self.device)
        action = torch.FloatTensor(action_np).to(self.device) # Shape: (batch_size, action_dim)
        next_state = torch.FloatTensor(next_state_np).to(self.device)
        reward = torch.FloatTensor(reward_np).unsqueeze(1).to(self.device) # Shape: (batch_size, 1)
        not_done = torch.FloatTensor(not_done_np).unsqueeze(1).to(self.device) # Shape: (batch_size, 1)
        
        # Descomponer el estado en datos CGM y otros datos
        # flat_cgm_dim = int(np.prod(self.cgm_input_dim)) if self.cgm_input_dim and np.prod(self.cgm_input_dim) > 0 else 0
        # flat_other_dim = int(np.prod(self.other_input_dim)) if self.other_input_dim and np.prod(self.other_input_dim) > 0 else 0
        # total_flat_dim = flat_cgm_dim + flat_other_dim
        # if state.shape[1] != total_flat_dim:
        #     print_critical(f"La dimensión del estado en el batch ({state.shape[1]}) no coincide con la esperada ({total_flat_dim})")
        #     return {CONST_ACTOR_LOSS: float('nan'), "critic1_loss": float('nan'), "critic2_loss": float('nan')}

        # Usar las dimensiones de entrada del modelo para la división
        # Asegurarse que cgm_input_dim y other_input_dim son tuplas (ej. (timesteps, features) o (features,))
        flat_cgm_dim = int(np.prod(self.cgm_input_dim))

        cgm_data = state[:, :flat_cgm_dim].reshape(batch_size, *self.cgm_input_dim) if flat_cgm_dim > 0 else torch.empty(batch_size, 0).to(self.device)
        other_data = state[:, flat_cgm_dim:] if state.shape[1] > flat_cgm_dim else torch.empty(batch_size, 0).to(self.device)

        next_cgm_data = next_state[:, :flat_cgm_dim].reshape(batch_size, *self.cgm_input_dim) if flat_cgm_dim > 0 else torch.empty(batch_size, 0).to(self.device)
        next_other_data = next_state[:, flat_cgm_dim:] if next_state.shape[1] > flat_cgm_dim else torch.empty(batch_size, 0).to(self.device)
        
        gamma = self.config.get('gamma', 0.99)

        # --- Actualización de los Críticos ---
        with torch.no_grad():
            # Seleccionar acción según la política objetivo (actor_target) y añadir ruido
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            
            next_action = (self.actor_target(next_cgm_data, next_other_data) + noise).clamp(self.min_action, self.max_action)

            # Calcular el valor Q objetivo
            target_Q1 = self.critic1_target(next_cgm_data, next_other_data, next_action)
            target_Q2 = self.critic2_target(next_cgm_data, next_other_data, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * gamma * target_Q

        # Obtener valores Q actuales
        current_Q1 = self.critic1(cgm_data, other_data, action)
        current_Q2 = self.critic2(cgm_data, other_data, action)

        # Calcular pérdida de los críticos
        critic1_loss = F.mse_loss(current_Q1, target_Q)
        critic2_loss = F.mse_loss(current_Q2, target_Q)
        
        # Optimizar crítico 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1.0) # Opcional: gradient clipping
        self.critic1_optimizer.step()

        # Optimizar crítico 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0) # Opcional: gradient clipping
        self.critic2_optimizer.step()
        
        actor_loss_val = torch.tensor(0.0) # Inicializar en caso de que no se actualice el actor

        # Actualización Retrasada de la Política (Actor) y Redes Objetivo
        self.total_it += 1
        if self.total_it % self.policy_delay == 0:
            # Calcular pérdida del actor
            pi_actions = self.actor(cgm_data, other_data)
            
            # Componente RL de la pérdida del actor
            q_pi_for_rl = self.critic1(cgm_data, other_data, pi_actions)
            actor_loss_rl = -q_pi_for_rl.mean()
            
            # Componente de Behavior Cloning (BC)
            # Usar q_pi_for_rl (Q(s, pi(s))) para la normalización de BC, como es común
            lambda_bc_norm_factor = q_pi_for_rl.abs().mean().detach()
            # Asegurar que el factor de normalización no sea demasiado pequeño
            lambda_bc_norm_factor = torch.clamp(lambda_bc_norm_factor, min=CONST_EPSILON) 
            
            actor_loss_bc = F.mse_loss(pi_actions, action) / lambda_bc_norm_factor # Ya se sumó CONST_EPSILON o se clampeó
            
            actor_loss = actor_loss_rl + self.alpha_bc * actor_loss_bc
            actor_loss_val = actor_loss.item()

            # Optimizar actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0) # Opcional: gradient clipping
            self.actor_optimizer.step()

            # Actualizar redes objetivo
            self._update_target_networks()

        return {
            CONST_ACTOR_LOSS: actor_loss_val if isinstance(actor_loss_val, float) else actor_loss_val.item(),
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
        }



def create_td3_bc_model(
    feature_config: Optional[Dict[str, List[str]]] = None
) -> DRLModelWrapperPyTorch:
    """
    Crea un modelo TD3+BC para dosificación de insulina.
    
    Parámetros:
    -----------
    feature_config : Optional[Dict[str, List[str]]], opcional
        Configuración de características para determinar las dimensiones de entrada del modelo
        y para uso del wrapper. Si es None, se usará get_feature_groups().
        
    Retorna:
    --------
    DRLModelWrapperPyTorch
        Modelo TD3+BC inicializado envuelto en DRLModelWrapperPyTorch
    """
    effective_feature_config = feature_config
    
    cgm_feature_names = effective_feature_config.get('cgm_features', [])
    # 'other_features' de get_feature_groups excluye explícitamente CONTEXT_FEATURE_ORDER
    other_non_context_feature_names = effective_feature_config.get('other_features', [])
    context_feature_names = effective_feature_config.get('explicit_context_features', CONTEXT_FEATURE_ORDER)

    # Dimensiones para el modelo TD3BCModel
    cgm_dim_for_model = (len(cgm_feature_names), 1) if cgm_feature_names else (0, 0)
    
    # 'other_input_dim' para TD3BCModel debe incluir tanto 'other_non_context_feature_names' como 'context_feature_names'
    total_other_features_len_for_model = len(other_non_context_feature_names) + len(context_feature_names)
    other_dim_for_model = (total_other_features_len_for_model,) if total_other_features_len_for_model > 0 else (0,)

    # Instanciar el modelo TD3BCModel con las dimensiones derivadas
    # Asegurarse que TD3BCModel internamente use estas dimensiones para sus actor/critic
    model_instance = TD3BCModel(
        cgm_input_dim=cgm_dim_for_model,
        other_input_dim=other_dim_for_model, # Esta es la dimensión combinada de otras y de contexto
        config=TD3_BC_CONFIG,
        rewards_function=compute_reward
    )
    
    print_info(f"TD3BCModel instanciado con cgm_input_dim={cgm_dim_for_model}, other_input_dim={other_dim_for_model}")
    
    # Pasar la instancia del modelo y feature_config al wrapper
    # feature_config es usado por el wrapper para construir el estado completo, tamaño del buffer, etc.
    # El wrapper usará sus propias self.cgm_cols, self.other_cols, self.context_cols para construir el estado aplanado.
    # La consistencia se mantiene porque TD3BCModel ahora espera other_input_dim que coincida con la suma de
    # las longitudes de self.other_cols y self.context_cols del wrapper.
    wrapper = DRLModelWrapperPyTorch(
        model_instance,
        algorithm="TD3+BC",
        feature_config=effective_feature_config # El wrapper usará esto para definir su state_dim
    )
    return wrapper