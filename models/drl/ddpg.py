import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from typing import Tuple, Dict, Any, Optional, List, Union

from custom.DeepReinforcementLearning.drl_pt import DRLModelWrapperPyTorch
from models.utils.replay_buffer import ReplayBuffer
from config.models_config import DDPG_CONFIG, BUFFER_CONFIG
from constants.constants import (
    IDEAL_LOWER_BOUND, IDEAL_UPPER_BOUND, CONST_DEFAULT_SEED,
    CONST_ACTOR_LOSS, CONST_CRITIC_LOSS, CONTEXT_FEATURE_ORDER
)
from custom.printer import print_info, print_warning


class Actor(nn.Module):
    """
    Red Actor para DDPG. Mapea estados a acciones.
    """
    def __init__(self, state_dim: int, action_dim: int, max_action: float, hidden_dim: int = 256):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Pase hacia adelante para el Actor.

        Parámetros:
        -----------
        state : torch.Tensor
            Tensor de estado de entrada.

        Retorna:
        --------
        torch.Tensor
            Tensor de acción, escalado por max_action.
        """
        x = torch.relu(self.layer_1(state))
        x = torch.relu(self.layer_2(x))
        # Salida tanh para acotar entre -1 y 1, luego escalar por max_action
        return self.max_action * torch.tanh(self.layer_3(x))

class Critic(nn.Module):
    """
    Red Crítico para DDPG. Mapea pares (estado, acción) a valores Q.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Critic, self).__init__()
        # Q1 architecture
        self.layer_1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Pase hacia adelante para el Crítico.

        Parámetros:
        -----------
        state : torch.Tensor
            Tensor de estado de entrada.
        action : torch.Tensor
            Tensor de acción de entrada.

        Retorna:
        --------
        torch.Tensor
            Valor Q estimado.
        """
        sa = torch.cat([state, action], 1)
        q1 = torch.relu(self.layer_1(sa))
        q1 = torch.relu(self.layer_2(q1))
        q1 = self.layer_3(q1)
        return q1

class DDPG(nn.Module):
    """
    Implementación del algoritmo Deep Deterministic Policy Gradient (DDPG).
    """
    def __init__(self,
                 cgm_input_dim: Tuple[int, int],      # (timesteps, cgm_features)
                 other_input_dim: Tuple[int],       # (other_features_len,)
                 context_dim: int,                  # num_context_features
                 action_dim: int,
                 max_action: float,
                 min_action: float = 0.0,
                 config: Optional[Dict[str, Any]] = None):
        super(DDPG, self).__init__()

        self.cgm_input_dim = cgm_input_dim
        self.other_input_dim = other_input_dim
        self.context_dim = context_dim
        
        # Calcular la dimensión total del estado
        # Estado = CGM aplanado + otras características + características de contexto
        self.state_dim = (cgm_input_dim[0] * cgm_input_dim[1]) + \
                         other_input_dim[0] + \
                         context_dim

        self.action_dim = action_dim
        self.max_action = max_action
        self.min_action = min_action

        self.config = config if config else DDPG_CONFIG
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hidden_dim = self.config.get("hidden_dim", 256)
        self.actor = Actor(self.state_dim, action_dim, max_action, hidden_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.get("actor_lr", 1e-4), weight_decay=self.config.get("weight_decay", 1e-5))

        self.critic = Critic(self.state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.get("critic_lr", 1e-3), weight_decay=self.config.get("weight_decay", 1e-5))
        
        self.default_target_glucose = (IDEAL_LOWER_BOUND + IDEAL_UPPER_BOUND) / 2.0
        self.exploration_noise = self.config.get("exploration_noise", 0.1)
        
        # Semilla para reproducibilidad del ruido de exploración
        seed = self.config.get('seed', CONST_DEFAULT_SEED)
        self.rng = np.random.default_rng(seed)


    def _build_state_representation(self,
                                   x_cgm_sample: np.ndarray,      # (timesteps, cgm_features)
                                   x_other_sample: np.ndarray,    # (other_features_len,)
                                   context_dict: Dict[str, float] # Dict de escalares de contexto
                                   ) -> torch.Tensor:
        """
        Construye el tensor de estado completo a partir de las entradas.

        Parámetros:
        -----------
        x_cgm_sample : np.ndarray
            Muestra de datos CGM (ventana).
        x_other_sample : np.ndarray
            Muestra de otras características.
        context_dict : Dict[str, float]
            Diccionario con valores de contexto.

        Retorna:
        --------
        torch.Tensor
            Tensor de estado concatenado y aplanado.
        """
        # Asegurar que las entradas sean NumPy arrays
        if not isinstance(x_cgm_sample, np.ndarray):
            print_warning(f"_build_state_representation: x_cgm_sample no es ndarray ({type(x_cgm_sample)}). Intentando convertir.")
            x_cgm_sample = np.array(x_cgm_sample)
        if not isinstance(x_other_sample, np.ndarray):
            print_warning(f"_build_state_representation: x_other_sample no es ndarray ({type(x_other_sample)}). Intentando convertir.")
            x_other_sample = np.array(x_other_sample)

        cgm_flat = torch.tensor(x_cgm_sample.flatten(), dtype=torch.float32, device=self.device)
        other_tensor = torch.tensor(x_other_sample, dtype=torch.float32, device=self.device)
        
        context_values = [context_dict.get(k, 0.0) for k in CONTEXT_FEATURE_ORDER]
        context_tensor = torch.tensor(context_values, dtype=torch.float32, device=self.device)
        
        # Asegurarse que todos los tensores sean 1D antes de concatenar
        if cgm_flat.ndim == 0: cgm_flat = cgm_flat.unsqueeze(0)
        if other_tensor.ndim == 0: other_tensor = other_tensor.unsqueeze(0)
        if context_tensor.ndim == 0: context_tensor = context_tensor.unsqueeze(0)

        full_state = torch.cat([cgm_flat, other_tensor, context_tensor], dim=0)
        return full_state.unsqueeze(0) # Añadir dimensión de batch

    def select_action(self,
                      state_tuple: Tuple[np.ndarray, np.ndarray], # (x_cgm_sample, x_other_sample)
                      context_dict: Dict[str, float],
                      add_noise: bool = True) -> np.ndarray:
        """
        Selecciona una acción usando el actor, con ruido opcional para exploración.

        Parámetros:
        -----------
        state_tuple : Tuple[np.ndarray, np.ndarray]
            Tupla con (muestra CGM, muestra otras características).
        context_dict : Dict[str, float]
            Diccionario con valores de contexto.
        add_noise : bool, opcional
            Si se debe añadir ruido Gaussiano para exploración (default: True).

        Retorna:
        --------
        np.ndarray
            Acción seleccionada (dosis de insulina).
        """
        x_cgm_sample, x_other_sample = state_tuple
        state_tensor = self._build_state_representation(x_cgm_sample, x_other_sample, context_dict)
        
        self.actor.eval() # Modo evaluación para selección de acción determinística
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().data.numpy().flatten()
        self.actor.train() # Volver a modo entrenamiento

        if add_noise:
            noise = self.rng.normal(0, self.max_action * self.exploration_noise, size=self.action_dim)
            action = action + noise
            
        return np.clip(action, self.min_action, self.max_action)

    def run_training_step(self, replay_buffer: ReplayBuffer, batch_size: int) -> Dict[str, float]:
        """
        Realiza un paso de actualización de DDPG.

        Parámetros:
        -----------
        replay_buffer : ReplayBuffer
            Buffer de repetición de donde muestrear transiciones.
        batch_size : int
            Tamaño del lote para el muestreo.

        Retorna:
        --------
        Dict[str, float]
            Diccionario con las pérdidas del actor y el crítico.
        """
        if len(replay_buffer) < batch_size:
            return {CONST_ACTOR_LOSS: 0.0, CONST_CRITIC_LOSS: 0.0} # No hay suficientes muestras

        # Muestrear transiciones del buffer
        # Se asume que el buffer almacena (full_state, action, reward, full_next_state, done)
        # donde full_state y full_next_state ya son tensores preprocesados.
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        # Convertir a tensores PyTorch
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).reshape(-1, 1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).reshape(-1, 1).to(self.device)

        # Calcular Q objetivo
        with torch.no_grad():
            target_actions = self.actor_target(next_state)
            target_q = self.critic_target(next_state, target_actions)
            target_q = reward + (1 - done) * self.config.get("gamma", 0.99) * target_q
        
        # Actualizar Crítico
        current_q = self.critic(state, action)
        critic_loss = nn.functional.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actualizar Actor
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Actualizar redes objetivo (soft update)
        tau = self.config.get("tau", 0.001)
        self._soft_update(self.critic_target, self.critic, tau)
        self._soft_update(self.actor_target, self.actor, tau)

        return {
            CONST_ACTOR_LOSS: actor_loss.item(),
            CONST_CRITIC_LOSS: critic_loss.item()
        }

    def predict_with_context(self, x_cgm: np.ndarray, x_other: np.ndarray,
                             current_glucose: float, carb_intake: float, iob: float,
                             exercise_intensity: Optional[float] = None, stress_level: Optional[float] = None,
                             work_intensity: Optional[float] = None, sleep_quality: Optional[float] = None,
                             target_glucose: Optional[float] = None # No usado directamente por DDPG, pero parte de la interfaz
                             ) -> float:
        """
        Predice una dosis de insulina usando el contexto completo.

        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM (espera forma [1, timesteps, cgm_features]).
        x_other : np.ndarray
            Otras características (espera forma [1, other_features_len]).
        current_glucose, carb_intake, iob, ... : float
            Valores escalares de contexto.

        Retorna:
        --------
        float
            Dosis de insulina recomendada.
        """
        if x_cgm.ndim == 3 and x_cgm.shape[0] == 1:
            x_cgm_sample = x_cgm[0]
        elif x_cgm.ndim == 2: # Si se pasa una sola muestra sin la dimensión de batch
            x_cgm_sample = x_cgm
        else:
            raise ValueError(f"Forma de x_cgm inesperada: {x_cgm.shape}. Se esperaba (1, timesteps, features) o (timesteps, features).")

        if x_other.ndim == 2 and x_other.shape[0] == 1:
            x_other_sample = x_other[0]
        elif x_other.ndim == 1: # Si se pasa una sola muestra sin la dimensión de batch
            x_other_sample = x_other
        else:
            raise ValueError(f"Forma de x_other inesperada: {x_other.shape}. Se esperaba (1, features) o (features,).")

        context_dict = {
            'current_glucose': current_glucose,
            'carb_intake': carb_intake,
            'iob': iob,
            'sleep_quality': sleep_quality if sleep_quality is not None else 0.0,
            'work_intensity': work_intensity if work_intensity is not None else 0.0,
            'exercise_intensity': exercise_intensity if exercise_intensity is not None else 0.0 # Mapeo activity_level a exercise_intensity
        }
        
        action = self.select_action((x_cgm_sample, x_other_sample), context_dict, add_noise=False)
        return float(action[0]) # Retorna la dosis como un escalar

    def _soft_update(self, target: nn.Module, source: nn.Module, tau: float) -> None:
        """
        Realiza una actualización suave de los parámetros de la red objetivo.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

    def save_state(self) -> Dict[str, Any]:
        """Guarda el estado del modelo DDPG (redes y optimizadores)."""
        return {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Carga el estado del modelo DDPG."""
        self.actor.load_state_dict(state['actor_state_dict'])
        self.critic.load_state_dict(state['critic_state_dict'])
        self.actor_optimizer.load_state_dict(state['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer_state_dict'])
        self.actor_target.load_state_dict(state['actor_target_state_dict'])
        self.critic_target.load_state_dict(state['critic_target_state_dict'])


def create_ddpg_model(
    cgm_input_dim: Tuple[int, int],      # (timesteps, cgm_features)
    other_input_dim: Tuple[int],       # (other_features_len,)
    action_dim: int = 1,                 # Dosis de insulina
    config: Optional[Dict[str, Any]] = None
) -> DRLModelWrapperPyTorch:
    """
    Crea una instancia de DRLModelWrapperPyTorch con un modelo DDPG.

    Parámetros:
    -----------
    cgm_input_dim : Tuple[int, int]
        Dimensiones de la entrada CGM (pasos_tiempo, características_cgm).
    other_input_dim : Tuple[int]
        Dimensiones de otras características (num_otras_características,).
    action_dim : int, opcional
        Dimensión del espacio de acciones (default: 1 para dosis de insulina).
    config : Optional[Dict[str, Any]], opcional
        Configuración para el modelo DDPG y el wrapper. Si es None, usa DDPG_CONFIG.

    Retorna:
    --------
    DRLModelWrapperPyTorch
        Wrapper del modelo DDPG listo para ser usado en el pipeline de entrenamiento.
    """
    effective_config = config if config is not None else DDPG_CONFIG
    
    context_dim = len(CONTEXT_FEATURE_ORDER) # Determinado por el orden definido
    max_action = effective_config.get("max_action", 20.0) # Dosis máxima de insulina
    min_action = effective_config.get("min_action", 0.0)   # Dosis mínima de insulina

    ddpg_agent = DDPG(
        cgm_input_dim=cgm_input_dim,
        other_input_dim=other_input_dim,
        context_dim=context_dim,
        action_dim=action_dim,
        max_action=max_action,
        min_action=min_action,
        config=effective_config
    )
    
    # Argumentos para el wrapper, incluyendo los necesarios para la inicialización del modelo DRL
    wrapper_kwargs = {
        'algorithm': "DDPG",
        'cgm_input_dim': cgm_input_dim,
        'other_input_dim': other_input_dim,
        'context_dim': context_dim,
        'action_dim': action_dim,
        'max_action': max_action,
        'min_action': min_action,
        **effective_config # Pasar toda la configuración DDPG al wrapper también
    }

    print_info(f"Creando DDPGModelWrapper con: cgm_dims={cgm_input_dim}, other_dims={other_input_dim}, context_dim={context_dim}")
    
    return DRLModelWrapperPyTorch(ddpg_agent, **wrapper_kwargs)