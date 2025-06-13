import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from typing import Tuple, Dict, Any, Optional, List, Union
from custom.DeepReinforcementLearning.drl_pt import DRLModelWrapperPyTorch
from custom.printer import print_critical, print_warning
from models.utils.replay_buffer import ReplayBuffer
from config.models_config import DDPG_CONFIG
from constants.constants import CONST_ACTOR_LOSS, CONST_CRITIC_LOSS, CONTEXT_FEATURE_ORDER

# Actor y Critic
class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float, hidden_dim: int = 256):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer_1(state))
        x = F.relu(self.layer_2(x))
        action = torch.tanh(self.layer_3(x)) * self.max_action
        return action

class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], 1)
        q_value = F.relu(self.layer_1(sa))
        q_value = F.relu(self.layer_2(q_value))
        q_value = self.layer_3(q_value)
        return q_value

class DDPG(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int = DDPG_CONFIG.get("action_dim",1),
                 max_action: float = DDPG_CONFIG.get("max_action", 20.0),
                 min_action: float = DDPG_CONFIG.get("min_action",0.0),
                 config: Optional[Dict[str, Any]] = None):
        super(DDPG, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.min_action = min_action

        self.config = config if config else DDPG_CONFIG.copy()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hidden_dim = self.config.get("hidden_dim", 256)
        self.actor = Actor(self.state_dim, action_dim, max_action, hidden_dim).to(self.device)
        self.target_actor = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.get("actor_lr", 1e-4), weight_decay=self.config.get("weight_decay", 1e-5))

        self.critic = Critic(self.state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.get("critic_lr", 1e-3), weight_decay=self.config.get("weight_decay", 1e-5))
        
        self.total_it = 0

    def _build_state_tensor_from_components(self,
                                x_cgm_sample_np: np.ndarray,      # (batch, timesteps, cgm_features) o (batch, flat_cgm_dim)
                                x_other_sample_np: np.ndarray,    # (batch, other_features_len)
                                context_values_np: np.ndarray     # (batch, context_dim)
                                ) -> torch.Tensor:
        """
        Construye el tensor de estado aplanado a partir de componentes NumPy y lo pasa al dispositivo.
        """
        # NaN/Inf checks for input numpy arrays
        if np.isnan(x_cgm_sample_np).any() or np.isinf(x_cgm_sample_np).any():
            print_critical("NaN/Inf detectado en 'x_cgm_sample_np' en _build_state_tensor_from_components")
        if np.isnan(x_other_sample_np).any() or np.isinf(x_other_sample_np).any():
            print_critical("NaN/Inf detectado en 'x_other_sample_np' en _build_state_tensor_from_components")
        if np.isnan(context_values_np).any() or np.isinf(context_values_np).any():
            print_critical("NaN/Inf detectado en 'context_values_np' en _build_state_tensor_from_components")

        x_cgm_flat = x_cgm_sample_np.reshape(x_cgm_sample_np.shape[0], -1)
        x_other_flat = x_other_sample_np.reshape(x_other_sample_np.shape[0], -1)
        
        components = []
        if x_cgm_flat.size > 0:
            components.append(x_cgm_flat)
        if x_other_flat.size > 0:
            components.append(x_other_flat)
        if context_values_np.size > 0:
            components.append(context_values_np)

        if not components:
            print_warning("Todos los componentes del estado están vacíos en _build_state_tensor_from_components.")
            # Devolver un tensor vacío con la forma correcta si es posible, o manejar el error
            # Por ahora, esto probablemente llevará a un error más adelante si el estado es fundamental.
            # Considerar devolver un tensor de ceros de la dimensión esperada si es un caso válido.
            # return torch.empty(x_cgm_sample_np.shape[0], 0, dtype=torch.float32).to(self.device)
            # O, si el estado no puede estar vacío:
            raise ValueError("No se pueden construir estados a partir de componentes vacíos.")


        full_state_np = np.concatenate(components, axis=1)
        if np.isnan(full_state_np).any() or np.isinf(full_state_np).any():
            print_critical("NaN/Inf detectado en 'full_state_np' después de la concatenación en _build_state_tensor_from_components")
        return torch.tensor(full_state_np, dtype=torch.float32).to(self.device)

    def select_action(self,
                      x_cgm_np: np.ndarray,      # (1, timesteps, cgm_features) o (1, flat_cgm_dim)
                      x_other_np: np.ndarray,    # (1, other_features_len)
                      context_dict: Dict[str, float], # Diccionario con valores de contexto
                      add_noise: bool = True) -> torch.Tensor:
        # Convertir context_dict a un array NumPy en el orden correcto
        context_values_list = [context_dict.get(col, 0.0) for col in CONTEXT_FEATURE_ORDER]
        context_np = np.array(context_values_list, dtype=np.float32).reshape(1, -1)

        state_tensor = self._build_state_tensor_from_components(x_cgm_np, x_other_np, context_np)
        
        if torch.isnan(state_tensor).any() or torch.isinf(state_tensor).any():
            print_critical(f"NaN/Inf detectado en 'state_tensor' en select_action: {state_tensor}")
            # Devolver una acción por defecto o manejar el error
            return torch.zeros(self.action_dim, device=self.device)


        self.actor.eval() # Modo evaluación para selección de acción
        with torch.no_grad():
            action = self.actor(state_tensor)
        self.actor.train() # Volver a modo entrenamiento

        if torch.isnan(action).any() or torch.isinf(action).any():
            print_critical(f"NaN/Inf detectado en 'action' (salida del actor) en select_action: {action}")
            action = torch.zeros_like(action) # Fallback

        if add_noise:
            noise = torch.randn_like(action) * self.config.get("exploration_noise", 0.1)
            action = (action + noise).clamp(self.min_action, self.max_action)
        
        if torch.isnan(action).any() or torch.isinf(action).any():
            print_critical(f"NaN/Inf detectado en 'action' (después de ruido/clamp) en select_action: {action}")
            # Fallback a una acción segura si es NaN
            action = torch.full_like(action, self.min_action)


        return action

    def update(self, batch_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        self.total_it +=1
        states, actions, rewards, next_states, dones = batch_data

        # Chequeo de NaNs en datos de entrada del batch
        if torch.isnan(states).any() or torch.isinf(states).any():
            print_critical("NaN/Inf detectado en 'states' en DDPG.update")
            return {CONST_CRITIC_LOSS: float('nan'), CONST_ACTOR_LOSS: float('nan')}
        if torch.isnan(actions).any() or torch.isinf(actions).any():
            print_critical("NaN/Inf detectado en 'actions' en DDPG.update")
            return {CONST_CRITIC_LOSS: float('nan'), CONST_ACTOR_LOSS: float('nan')}
        if torch.isnan(rewards).any() or torch.isinf(rewards).any():
            print_critical("NaN/Inf detectado en 'rewards' en DDPG.update")
            return {CONST_CRITIC_LOSS: float('nan'), CONST_ACTOR_LOSS: float('nan')}
        if torch.isnan(next_states).any() or torch.isinf(next_states).any():
            print_critical("NaN/Inf detectado en 'next_states' en DDPG.update")
            return {CONST_CRITIC_LOSS: float('nan'), CONST_ACTOR_LOSS: float('nan')}
        if torch.isnan(dones).any() or torch.isinf(dones).any(): # dones es bool, pero se convierte a float en y_target
            print_critical("NaN/Inf detectado en 'dones' tensor en DDPG.update") # menos probable
            return {CONST_CRITIC_LOSS: float('nan'), CONST_ACTOR_LOSS: float('nan')}

        # --- Actualización del Crítico ---
        with torch.no_grad():
            next_actions_target = self.target_actor(next_states)
            if torch.isnan(next_actions_target).any() or torch.isinf(next_actions_target).any():
                print_critical("NaN/Inf detectado en 'next_actions_target' (target_actor output)")
                return {CONST_CRITIC_LOSS: float('nan'), CONST_ACTOR_LOSS: float('nan')}

            q_next_target = self.target_critic(next_states, next_actions_target)
            if torch.isnan(q_next_target).any() or torch.isinf(q_next_target).any():
                print_critical("NaN/Inf detectado en 'q_next_target' (target_critic output)")
                return {CONST_CRITIC_LOSS: float('nan'), CONST_ACTOR_LOSS: float('nan')}
            
            y_target = rewards + (self.config['gamma'] * q_next_target * (1.0 - dones.float())) # Asegurar que dones sea float
            if torch.isnan(y_target).any() or torch.isinf(y_target).any():
                print_critical(f"NaN/Inf detectado en 'y_target'. rewards: {rewards.mean()}, q_next_target: {q_next_target.mean()}, dones: {dones.float().mean()}")
                return {CONST_CRITIC_LOSS: float('nan'), CONST_ACTOR_LOSS: float('nan')}

        current_q_values = self.critic(states, actions)
        if torch.isnan(current_q_values).any() or torch.isinf(current_q_values).any():
            print_critical("NaN/Inf detectado en 'current_q_values' (critic output)")
            return {CONST_CRITIC_LOSS: float('nan'), CONST_ACTOR_LOSS: float('nan')}

        critic_loss = F.mse_loss(current_q_values, y_target)
        if torch.isnan(critic_loss).any() or torch.isinf(critic_loss).any():
            print_critical(f"Critic loss es NaN/Inf. current_Q: {current_q_values.mean()}, y_target: {y_target.mean()}")
            # Adicionalmente, imprimir algunas muestras de current_q_values y y_target
            print_critical(f"Sample current_Q: {current_q_values[:5]}")
            print_critical(f"Sample y_target: {y_target[:5]}")
            return {CONST_CRITIC_LOSS: float('nan'), CONST_ACTOR_LOSS: float('nan')}

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # --- Actualización del Actor ---
        # DDPG actualiza el actor con menos frecuencia o igual que el crítico.
        # Para simplificar y alinear con el algoritmo base, actualizamos en cada paso.
        # Si se requiere una actualización retrasada como en TD3, se necesitaría un contador `total_it` y `policy_delay`.
        
        actor_actions = self.actor(states)
        if torch.isnan(actor_actions).any() or torch.isinf(actor_actions).any():
            print_critical("NaN/Inf detectado en 'actor_actions' (actor output for loss)")
            return {CONST_CRITIC_LOSS: critic_loss.item(), CONST_ACTOR_LOSS: float('nan')}

        q_values_for_actor_loss = self.critic(states, actor_actions)
        if torch.isnan(q_values_for_actor_loss).any() or torch.isinf(q_values_for_actor_loss).any():
            print_critical("NaN/Inf detectado en 'q_values_for_actor_loss' (critic output for actor loss)")
            return {CONST_CRITIC_LOSS: critic_loss.item(), CONST_ACTOR_LOSS: float('nan')}
            
        actor_loss = -q_values_for_actor_loss.mean()
        if torch.isnan(actor_loss).any() or torch.isinf(actor_loss).any():
            print_critical(f"Actor loss es NaN/Inf. q_values_for_actor_loss: {q_values_for_actor_loss.mean()}")
            return {CONST_CRITIC_LOSS: critic_loss.item(), CONST_ACTOR_LOSS: float('nan')}

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # --- Actualización suave de las redes objetivo ---
        self._soft_update(self.target_critic, self.critic, self.config['tau'])
        self._soft_update(self.target_actor, self.actor, self.config['tau'])

        return {
            CONST_CRITIC_LOSS: critic_loss.item(),
            CONST_ACTOR_LOSS: actor_loss.item()
        }

    def predict_with_context(self, x_cgm: np.ndarray, x_other: np.ndarray,
                             current_glucose: float, carb_intake: float, iob: float,
                             exercise_intensity: Optional[float] = None,
                             work_intensity: Optional[float] = None, 
                             sleep_quality: Optional[float] = None,
                             target_glucose: Optional[float] = None # target_glucose no se usa en DDPG directamente para la acción
                             ) -> float:
        # Construir el diccionario de contexto
        context_dict = {
            'glucose_last': current_glucose,
            'meal_carbs': carb_intake,
            'insulin_on_board': iob,
            'sleep_quality': sleep_quality if sleep_quality is not None else 0.0, # Usar 0 si es None
            'work_intensity': work_intensity if work_intensity is not None else 0.0,
            'exercise_intensity': exercise_intensity if exercise_intensity is not None else 0.0
        }
        
        # x_cgm y x_other deben ser formateados como (1, ...) para select_action
        # Asegurar que x_cgm y x_other tengan una dimensión de batch
        if x_cgm.ndim == 2: # (timesteps, features)
            x_cgm_reshaped = x_cgm[np.newaxis, ...]
        elif x_cgm.ndim == 1: # (flat_cgm_dim)
             x_cgm_reshaped = x_cgm[np.newaxis, :]
        else: # ya tiene batch dim o es incorrecto
            x_cgm_reshaped = x_cgm
            if x_cgm.shape[0] != 1:
                print_warning(f"x_cgm en predict_with_context tiene una forma inesperada: {x_cgm.shape}")
                # Podría intentar tomar la primera muestra si hay varias, o fallar.
                # Por ahora, se asume que si tiene más de 1D, la primera es el batch.

        if x_other.ndim == 1: # (other_features_len)
            x_other_reshaped = x_other[np.newaxis, :]
        else: # ya tiene batch dim o es incorrecto
            x_other_reshaped = x_other
            if x_other.shape[0] != 1:
                 print_warning(f"x_other en predict_with_context tiene una forma inesperada: {x_other.shape}")


        action_tensor = self.select_action(x_cgm_reshaped, x_other_reshaped, context_dict, add_noise=False)
        return action_tensor.item()

    def _soft_update(self, target: nn.Module, source: nn.Module, tau: float) -> None:
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def save_state(self) -> Dict[str, Any]:
         return {
             'actor_state_dict': self.actor.state_dict(),
             'critic_state_dict': self.critic.state_dict(),
             'target_actor_state_dict': self.target_actor.state_dict(),
             'target_critic_state_dict': self.target_critic.state_dict(),
             'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
             'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
             'config': self.config
         }

    def load_state(self, state: Dict[str, Any]) -> None:
         self.actor.load_state_dict(state['actor_state_dict'])
         self.critic.load_state_dict(state['critic_state_dict'])
         self.target_actor.load_state_dict(state['target_actor_state_dict'])
         self.target_critic.load_state_dict(state['target_critic_state_dict'])
         self.actor_optimizer.load_state_dict(state['actor_optimizer_state_dict'])
         self.critic_optimizer.load_state_dict(state['critic_optimizer_state_dict'])
         self.config = state.get('config', self.config)
         # Mover a dispositivo después de cargar
         self.to(self.device)


def create_ddpg_model(
    action_dim: int = 1,
    config: Optional[Dict[str, Any]] = None,
    feature_config: Optional[Dict[str, List[str]]] = None  # Añadido
) -> DRLModelWrapperPyTorch:
    """
    Crea un modelo DDPG envuelto para dosificación de insulina.

    Parámetros:
    -----------
    action_dim : int, opcional
        Dimensión de la acción (default: 1).
    config : Optional[Dict[str, Any]], opcional
        Configuración específica para DDPG (default: DDPG_CONFIG).
    feature_config : Optional[Dict[str, List[str]]], opcional
        Configuración de características para que el wrapper determine las dimensiones del estado.
        Si es None, se usará get_feature_groups().

    Retorna:
    --------
    DRLModelWrapperPyTorch
        Modelo DDPG inicializado envuelto en DRLModelWrapperPyTorch.
    """
    effective_config = config if config is not None else DDPG_CONFIG.copy()
    effective_feature_config = feature_config

    # model_kwargs para la clase DDPG.
    # state_dim será inyectado por DRLModelWrapperPyTorch basado en feature_config.
    model_specific_kwargs = {
        'action_dim': action_dim,
        'max_action': effective_config.get("max_action", 20.0),
        'min_action': effective_config.get("min_action", 0.0),
        'config': effective_config
        # 'state_dim' es manejado por el wrapper
    }

    # DRLModelWrapperPyTorch usará feature_config para determinar state_dim
    # y pasarlo en model_kwargs al instanciar DDPG.
    wrapper = DRLModelWrapperPyTorch(
        DDPG,
        algorithm="DDPG",
        feature_config=effective_feature_config,
        model_kwargs=model_specific_kwargs
    )
    return wrapper