import os, sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Conv1D, Flatten, Concatenate,
    BatchNormalization, Dropout, LayerNormalization
)
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from models.config import DDPG_CONFIG


class replay_buffer:
    """
    Buffer de experiencias para el algoritmo DDPG.
    
    Almacena transiciones (state, action, reward, next_state, done)
    y permite muestrear lotes de manera aleatoria para el entrenamiento.
    
    Parámetros:
    -----------
    capacity : int, opcional
        Capacidad máxima del buffer (default: 100000)
    """
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float, 
           next_state: np.ndarray, done: float) -> None:
        """
        Añade una transición al buffer.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado actual
        action : np.ndarray
            Acción tomada
        reward : float
            Recompensa recibida
        next_state : np.ndarray
            Estado siguiente
        done : float
            Indicador de fin de episodio (1.0 si terminó, 0.0 si no)
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Muestrea un lote aleatorio de transiciones.
        
        Parámetros:
        -----------
        batch_size : int
            Tamaño del lote a muestrear
            
        Retorna:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            (states, actions, rewards, next_states, dones)
        """
        if len(self.buffer) < batch_size:
            # Si no hay suficientes transiciones, devuelve lo que haya
            batch = random.sample(self.buffer, len(self.buffer))
        else:
            batch = random.sample(self.buffer, batch_size)
            
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self) -> int:
        """
        Retorna la cantidad de transiciones almacenadas.
        
        Retorna:
        --------
        int
            Número de transiciones en el buffer
        """
        return len(self.buffer)


class ou_action_noise:
    """
    Implementa el proceso de ruido de Ornstein-Uhlenbeck para exploración.
    
    Este ruido añade correlación temporal a las acciones para una exploración más efectiva
    en espacios continuos.
    
    Parámetros:
    -----------
    mean : np.ndarray
        Valor medio al que tiende el proceso
    std_deviation : np.ndarray
        Desviación estándar del ruido
    theta : float, opcional
        Velocidad de reversión a la media (default: 0.15)
    dt : float, opcional
        Delta de tiempo para la discretización (default: 1e-2)
    x_initial : Optional[np.ndarray], opcional
        Valor inicial del proceso (default: None)
    seed : int, opcional
        Semilla para la generación de números aleatorios (default: 42)
    """
    def __init__(self, mean: np.ndarray, std_deviation: np.ndarray, theta: float = 0.15, 
                dt: float = 1e-2, x_initial: Optional[np.ndarray] = None, seed: int = 42):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.reset()
        
    def __call__(self) -> np.ndarray:
        """
        Genera un nuevo valor de ruido siguiendo el proceso de Ornstein-Uhlenbeck.
        
        Retorna:
        --------
        np.ndarray
            Valor de ruido generado
        """
        # Fórmula para el proceso de Ornstein-Uhlenbeck
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * self.rng.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x
    
    def reset(self) -> None:
        """
        Reinicia el estado del ruido.
        """
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


def activation_function(x: tf.Tensor, name: str) -> tf.Tensor:
    """
    Aplica una función de activación según su nombre.
    
    Parámetros:
    -----------
    x : tf.Tensor
        Tensor de entrada
    name : str
        Nombre de la activación ('relu', 'tanh', 'leaky_relu', etc.)
        
    Retorna:
    --------
    tf.Tensor
        Tensor con activación aplicada
    """
    if name == 'relu':
        return tf.nn.relu(x)
    elif name == 'tanh':
        return tf.nn.tanh(x)
    elif name == 'leaky_relu':
        return tf.nn.leaky_relu(x)
    elif name == 'gelu':
        return tf.nn.gelu(x)
    else:
        # Por defecto usar ReLU
        return tf.nn.relu(x)


class actor_network(Model):
    """
    Red de Actor para DDPG que mapea estados a acciones determinísticas.
    
    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    action_high : np.ndarray
        Límite superior del rango de acciones
    action_low : np.ndarray
        Límite inferior del rango de acciones
    hidden_units : Optional[List[int]], opcional
        Unidades en capas ocultas (default: None, usa configuración por defecto)
    """
    def __init__(self, state_dim: int, action_dim: int, action_high: np.ndarray, 
                action_low: np.ndarray, hidden_units: Optional[List[int]] = None) -> None:
        super().__init__()
        
        # Valores predeterminados para capas ocultas
        if hidden_units is None:
            hidden_units = DDPG_CONFIG['actor_hidden_units']
        
        self.action_high = action_high
        self.action_low = action_low
        self.action_range = action_high - action_low
        
        # Capas para el procesamiento del estado
        self.layers_list = []
        for i, units in enumerate(hidden_units):
            self.layers_list.append(Dense(
                units, 
                name=f'actor_dense_{i}'
            ))
            self.layers_list.append(LayerNormalization(
                epsilon=DDPG_CONFIG['epsilon'],
                name=f'actor_ln_{i}'
            ))
            self.layers_list.append(Dropout(
                DDPG_CONFIG['dropout_rate'],
                name=f'actor_dropout_{i}'
            ))
        
        # Capa de salida: acciones con activación tanh [-1, 1] escalada al rango de acción
        self.output_layer = Dense(
            action_dim, 
            activation='tanh',
            name='actor_output'
        )
    
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Pasa la entrada por la red del actor.
        
        Parámetros:
        -----------
        inputs : tf.Tensor
            Estado de entrada
        training : bool, opcional
            Indica si está en modo entrenamiento (default: False)
            
        Retorna:
        --------
        tf.Tensor
            Acción determinística
        """
        x = inputs
        
        # Procesar a través de capas ocultas
        for layer in self.layers_list:
            # Aplicar activación antes de capas normalization/dropout
            if isinstance(layer, Dense):
                x = layer(x)
                x = activation_function(x, DDPG_CONFIG['actor_activation'])
            # Para capas de Dropout, pasamos el parámetro training
            elif isinstance(layer, Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        
        # Capa de salida con activación tanh y escalado
        raw_actions = self.output_layer(x)
        
        # Escalar desde [-1, 1] al rango de acción [low, high]
        scaled_actions = 0.5 * (raw_actions + 1.0) * self.action_range + self.action_low
        
        return scaled_actions


class critic_network(Model):
    """
    Red de Crítico para DDPG que mapea pares (estado, acción) a valores-Q.
    
    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    hidden_units : Optional[List[int]], opcional
        Unidades en capas ocultas (default: None, usa configuración por defecto)
    """
    def __init__(self, state_dim: int, action_dim: int, 
                hidden_units: Optional[List[int]] = None) -> None:
        super().__init__()
        
        # Valores predeterminados para capas ocultas
        if hidden_units is None:
            hidden_units = DDPG_CONFIG['critic_hidden_units']
        
        # Capas iniciales para procesar el estado
        self.state_layers = []
        for i, units in enumerate(hidden_units[:1]):  # Primera capa solo para estado
            self.state_layers.append(Dense(
                units,
                name=f'critic_state_dense_{i}'
            ))
            self.state_layers.append(LayerNormalization(
                epsilon=DDPG_CONFIG['epsilon'],
                name=f'critic_state_ln_{i}'
            ))
        
        # Capas para procesar la combinación de estado y acción
        self.combined_layers = []
        for i, units in enumerate(hidden_units[1:]):
            self.combined_layers.append(Dense(
                units,
                name=f'critic_combined_dense_{i}'
            ))
            self.combined_layers.append(LayerNormalization(
                epsilon=DDPG_CONFIG['epsilon'],
                name=f'critic_combined_ln_{i}'
            ))
            self.combined_layers.append(Dropout(
                DDPG_CONFIG['dropout_rate'],
                name=f'critic_dropout_{i}'
            ))
        
        # Capa de salida: valor Q (sin activación)
        self.output_layer = Dense(1, name='critic_output')
    
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training: bool = False) -> tf.Tensor:
        """
        Pasa la entrada por la red del crítico.
        
        Parámetros:
        -----------
        inputs : Tuple[tf.Tensor, tf.Tensor]
            Tupla de (estados, acciones)
        training : bool, opcional
            Indica si está en modo entrenamiento (default: False)
            
        Retorna:
        --------
        tf.Tensor
            Valores Q estimados
        """
        states, actions = inputs
        
        # Procesar el estado
        x = states
        for i, layer in enumerate(self.state_layers):
            if isinstance(layer, Dense):
                x = layer(x)
                x = activation_function(x, DDPG_CONFIG['critic_activation'])
            else:
                x = layer(x)
        
        # Combinar estado procesado con acción
        x = Concatenate()([x, actions])
        
        # Procesar a través de capas combinadas
        for layer in self.combined_layers:
            if isinstance(layer, Dense):
                x = layer(x)
                x = activation_function(x, DDPG_CONFIG['critic_activation'])
            elif isinstance(layer, Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        
        # Capa de salida
        q_value = self.output_layer(x)
        
        return q_value


class ddpg:
    """
    Implementación del algoritmo Deep Deterministic Policy Gradient (DDPG).
    
    DDPG combina ideas de DQN y métodos de policy gradient para manejar
    espacios de acción continuos con políticas determinísticas.
    
    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    action_high : np.ndarray
        Límite superior del rango de acciones
    action_low : np.ndarray
        Límite inferior del rango de acciones
    config : Optional[Dict[str, Any]], opcional
        Configuración personalizada (default: None, usa configuración por defecto)
    seed : int, opcional
        Semilla para generación de números aleatorios (default: 42)
    """
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int,
        action_high: np.ndarray,
        action_low: np.ndarray,
        config: Optional[Dict[str, Any]] = None,
        seed: int = 42
    ):
        # Use provided config or default
        self.config = config or DDPG_CONFIG
        
        # Extract hyperparameters from config
        actor_lr = self.config.get('actor_lr', DDPG_CONFIG['actor_lr'])
        critic_lr = self.config.get('critic_lr', DDPG_CONFIG['critic_lr'])
        self.gamma = self.config.get('gamma', DDPG_CONFIG['gamma'])
        self.tau = self.config.get('tau', DDPG_CONFIG['tau'])
        buffer_capacity = self.config.get('buffer_capacity', DDPG_CONFIG['buffer_capacity'])
        self.batch_size = self.config.get('batch_size', DDPG_CONFIG['batch_size'])
        noise_std = self.config.get('noise_std', DDPG_CONFIG['noise_std'])
        actor_hidden_units = self.config.get('actor_hidden_units')
        critic_hidden_units = self.config.get('critic_hidden_units')
        
        # Parámetros del entorno y del modelo
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_high = action_high
        self.action_low = action_low
        
        # Configurar semilla para reproducibilidad
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Valores predeterminados para capas ocultas
        self.actor_hidden_units = actor_hidden_units or DDPG_CONFIG['actor_hidden_units']
        self.critic_hidden_units = critic_hidden_units or DDPG_CONFIG['critic_hidden_units']
        
        # Crear modelos Actor y Crítico
        self.actor = actor_network(
            state_dim=state_dim,
            action_dim=action_dim,
            action_high=action_high,
            action_low=action_low,
            hidden_units=self.actor_hidden_units
        )
        
        self.critic = critic_network(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_units=self.critic_hidden_units
        )
        
        # Crear copias target
        self.target_actor = actor_network(
            state_dim=state_dim,
            action_dim=action_dim,
            action_high=action_high,
            action_low=action_low,
            hidden_units=self.actor_hidden_units
        )
        
        self.target_critic = critic_network(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_units=self.critic_hidden_units
        )
        
        # Asegurar que todos los modelos tienen forma inicializada
        dummy_state = np.zeros((1, state_dim), dtype=np.float32)
        dummy_action = np.zeros((1, action_dim), dtype=np.float32)
        
        self.actor(dummy_state)
        self.critic([dummy_state, dummy_action])
        self.target_actor(dummy_state)
        self.target_critic([dummy_state, dummy_action])
        
        # Sincronizar pesos iniciales
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        
        # Optimizadores
        self.actor_optimizer = Adam(learning_rate=actor_lr)
        self.critic_optimizer = Adam(learning_rate=critic_lr)
        
        # Buffer de experiencias
        self.replay_buffer = replay_buffer(buffer_capacity)
        
        # Ruido para exploración
        self.noise = ou_action_noise(
            mean=np.zeros(action_dim),
            std_deviation=noise_std * np.ones(action_dim),
            seed=seed
        )
        
        # Contador de pasos para actualización y métricas
        self.step_counter = 0
        self.actor_loss_sum = 0
        self.critic_loss_sum = 0
        self.q_value_sum = 0
        self.updates_count = 0
    
    def update_target_networks(self, tau: Optional[float] = None) -> None:
        """
        Actualiza los pesos de las redes target usando soft update.
        
        Parámetros:
        -----------
        tau : Optional[float], opcional
            Factor de actualización suave (si None, usa el valor por defecto)
        """
        if tau is None:
            tau = self.tau
            
        # Actualizar pesos del actor target
        actor_weights = self.actor.get_weights()
        target_actor_weights = self.target_actor.get_weights()
        new_target_actor_weights = []
        
        for i, weight in enumerate(actor_weights):
            new_target_actor_weights.append(tau * weight + (1 - tau) * target_actor_weights[i])
            
        self.target_actor.set_weights(new_target_actor_weights)
        
        # Actualizar pesos del crítico target
        critic_weights = self.critic.get_weights()
        target_critic_weights = self.target_critic.get_weights()
        new_target_critic_weights = []
        
        for i, weight in enumerate(critic_weights):
            new_target_critic_weights.append(tau * weight + (1 - tau) * target_critic_weights[i])
            
        self.target_critic.set_weights(new_target_critic_weights)
    
    def get_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Obtiene una acción determinística para un estado, opcionalmente añadiendo ruido.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado actual
        add_noise : bool, opcional
            Si se debe añadir ruido para exploración
            
        Retorna:
        --------
        np.ndarray
            Acción seleccionada
        """
        state_tensor = tf.convert_to_tensor(state[np.newaxis, :], dtype=tf.float32)
        action = self.actor(state_tensor, training=False)[0].numpy()
        
        if add_noise:
            noise = self.noise()
            action += noise
            
        # Clipear al rango válido de acciones
        action = np.clip(action, self.action_low, self.action_high)
        
        return action
    
    @tf.function
    def _update_networks(self, states: tf.Tensor, actions: tf.Tensor, rewards: tf.Tensor, 
                        next_states: tf.Tensor, dones: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Actualiza las redes del actor y crítico usando el algoritmo DDPG.
        
        Parámetros:
        -----------
        states : tf.Tensor
            Estados actuales
        actions : tf.Tensor
            Acciones tomadas
        rewards : tf.Tensor
            Recompensas recibidas
        next_states : tf.Tensor
            Estados siguientes
        dones : tf.Tensor
            Indicadores de finalización
            
        Retorna:
        --------
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
            (critic_loss, actor_loss, q_values)
        """
        with tf.GradientTape() as tape:
            # Predecir acciones target para los siguiente estados
            target_actions = self.target_actor(next_states, training=False)
            
            # Predecir Q-values target
            target_q_values = self.target_critic([next_states, target_actions], training=False)
            
            # Calcular Q-values objetivo usando la ecuación de Bellman
            targets = rewards + (1 - dones) * self.gamma * target_q_values
            
            # Predecir Q-values actuales
            current_q_values = self.critic([states, actions], training=True)
            
            # Calcular pérdida del crítico (error cuadrático medio)
            critic_loss = tf.reduce_mean(tf.square(targets - current_q_values), axis=0)
            
        # Calcular gradientes y actualizar crítico
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
        
        with tf.GradientTape() as tape:
            # Predecir acciones para los estados actuales
            actor_actions = self.actor(states, training=True)
            
            # Calcular Q-values para estas acciones
            actor_q_values = self.critic([states, actor_actions], training=False)
            
            # Pérdida del actor (negativo del Q-value promedio)
            # Queremos maximizar Q-value, así que minimizamos su negativo
            actor_loss = -tf.reduce_mean(actor_q_values, axis=0)
            
        # Calcular gradientes y actualizar actor
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        
        return critic_loss, actor_loss, tf.reduce_mean(current_q_values, axis=0)
    
    def train_step(self) -> Tuple[float, float, float]:
        """
        Realiza un paso de entrenamiento DDPG con un lote de experiencias.
        
        Retorna:
        --------
        Tuple[float, float, float]
            (critic_loss, actor_loss, q_value)
        """
        # Muestrear un lote del buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convertir a tensores para TensorFlow
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        # Reshape rewards y dones para que coincidan con la dimensión de Q-values
        rewards = tf.reshape(rewards, (-1, 1))
        dones = tf.reshape(dones, (-1, 1))
        
        # Actualizar redes
        critic_loss, actor_loss, q_value = self._update_networks(states, actions, rewards, next_states, dones)
        
        # Actualizar redes target con soft update
        self.update_target_networks()
        
        # Actualizar métricas acumuladas
        self.actor_loss_sum += actor_loss
        self.critic_loss_sum += critic_loss
        self.q_value_sum += q_value
        self.updates_count += 1
        
        return critic_loss.numpy(), actor_loss.numpy(), q_value.numpy()
    
    def _select_action(self, state: np.ndarray, step_counter: int, warmup_steps: int) -> np.ndarray:
        """
        Selecciona una acción basada en el estado actual y fase de entrenamiento.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado actual
        step_counter : int
            Contador de pasos
        warmup_steps : int
            Pasos iniciales de exploración pura
            
        Retorna:
        --------
        np.ndarray
            Acción seleccionada
        """
        if step_counter < warmup_steps:
            # Exploración inicial uniforme - usando el generador np.random.default_rng con semilla
            # para asegurar reproducibilidad
            rng = np.random.default_rng(seed=42)  # Using a default seed
            return rng.uniform(
                low=self.action_low,
                high=self.action_high,
                size=(self.action_dim,)
            )
        else:
            # Política del actor con ruido
            return self.get_action(state, add_noise=True)
    
    def _update_model(self, step_counter: int, update_every: int) -> Tuple[Optional[float], Optional[float]]:
        """
        Actualiza los modelos si es momento de hacerlo.
        
        Parámetros:
        -----------
        step_counter : int
            Contador de pasos totales
        update_every : int
            Frecuencia de actualización
            
        Retorna:
        --------
        Tuple[Optional[float], Optional[float]]
            (actor_loss, critic_loss) o (None, None) si no se actualizó
        """
        if step_counter % update_every == 0:
            if len(self.replay_buffer) > self.batch_size:
                # Realizar actualización
                critic_loss, actor_loss, _ = self.train_step()
                return actor_loss, critic_loss
        return None, None
    
    def _update_history(self, history: Dict, episode_reward: float, 
                      episode_actor_loss: List, episode_critic_loss: List) -> Dict:
        """
        Actualiza el historial de entrenamiento con los resultados del episodio.
        
        Parámetros:
        -----------
        history : Dict
            Historial de entrenamiento a actualizar
        episode_reward : float
            Recompensa del episodio
        episode_actor_loss : List
            Lista de pérdidas del actor durante el episodio
        episode_critic_loss : List
            Lista de pérdidas del crítico durante el episodio
            
        Retorna:
        --------
        Dict
            Historial actualizado
        """
        history['episode_rewards'].append(episode_reward)
        
        # Obtener promedios de métricas
        if self.updates_count > 0:
            actor_loss_avg = self.actor_loss_sum / self.updates_count
            critic_loss_avg = self.critic_loss_sum / self.updates_count
            q_value_avg = self.q_value_sum / self.updates_count
            
            # Resetear métricas
            self.actor_loss_sum = 0
            self.critic_loss_sum = 0
            self.q_value_sum = 0
            self.updates_count = 0
        else:
            actor_loss_avg = float('nan')
            critic_loss_avg = float('nan')
            q_value_avg = float('nan')
        
        history['actor_losses'].append(actor_loss_avg)
        history['critic_losses'].append(critic_loss_avg)
        history['avg_q_values'].append(q_value_avg)
            
        return history
    
    def _log_progress(self, episode: int, episodes: int, episode_reward_history: List, 
                    history: Dict, log_interval: int, best_reward: float) -> float:
        """
        Registra y muestra el progreso del entrenamiento periódicamente.
        
        Parámetros:
        -----------
        episode : int
            Episodio actual
        episodes : int
            Total de episodios
        episode_reward_history : List
            Historial reciente de recompensas
        history : Dict
            Historial completo de entrenamiento
        log_interval : int
            Intervalo para mostrar información
        best_reward : float
            Mejor recompensa hasta el momento
            
        Retorna:
        --------
        float
            Mejor recompensa actualizada
        """
        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(episode_reward_history)
            print(f"Episodio {episode+1}/{episodes} - Recompensa Promedio: {avg_reward:.2f}, "
                  f"Pérdida Actor: {history['actor_losses'][-1]:.4f}, "
                  f"Pérdida Crítico: {history['critic_losses'][-1]:.4f}")
            
            # Verificar si es el mejor modelo
            if avg_reward > best_reward:
                best_reward = avg_reward
                print(f"Nuevo mejor modelo con recompensa: {best_reward:.2f}")
        
        return best_reward
    
    def _run_episode(self, env, max_steps: int, step_counter: int, warmup_steps: int, 
                   update_every: int, render: bool) -> Tuple[float, List[float], List[float], int]:
        """
        Ejecuta un episodio completo de entrenamiento.
        
        Parámetros:
        -----------
        env : gym.Env
            Entorno donde ejecutar el episodio
        max_steps : int
            Pasos máximos por episodio
        step_counter : int
            Contador de pasos totales
        warmup_steps : int
            Pasos iniciales de exploración pura
        update_every : int
            Frecuencia de actualización del modelo
        render : bool
            Si se debe renderizar el entorno
            
        Retorna:
        --------
        Tuple[float, List[float], List[float], int]
            (episode_reward, episode_actor_loss, episode_critic_loss, step_counter)
        """
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        episode_reward = 0
        episode_actor_loss = []
        episode_critic_loss = []
        
        for _ in range(max_steps):
            step_counter += 1
            
            # Seleccionar acción
            action = self._select_action(state, step_counter, warmup_steps)
            
            # Ejecutar acción
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)
            
            # Renderizar si es necesario
            if render:
                env.render()
            
            # Guardar transición en buffer
            self.replay_buffer.add(state, action, reward, next_state, float(done))
            
            # Actualizar estado y recompensa
            state = next_state
            episode_reward += reward
            
            # Entrenar modelo si hay suficientes datos
            if len(self.replay_buffer) > self.batch_size and step_counter >= warmup_steps:
                actor_loss, critic_loss = self._update_model(step_counter, update_every)
                if actor_loss is not None:
                    episode_actor_loss.append(float(actor_loss))
                    episode_critic_loss.append(float(critic_loss))
            
            if done:
                break
                
        return episode_reward, episode_actor_loss, episode_critic_loss, step_counter
    
    def train(self, env, episodes: int = 1000, max_steps: int = 1000, warmup_steps: int = 1000, 
             update_every: int = 1, render: bool = False, log_interval: int = 10) -> Dict:
        """
        Entrena el agente DDPG en un entorno dado.
        
        Parámetros:
        -----------
        env : gym.Env
            Entorno de Gym o compatible
        episodes : int, opcional
            Número máximo de episodios (default: 1000)
        max_steps : int, opcional
            Pasos máximos por episodio (default: 1000)
        warmup_steps : int, opcional
            Pasos iniciales para recolectar experiencias antes de entrenar (default: 1000)
        update_every : int, opcional
            Frecuencia de actualización (default: 1)
        render : bool, opcional
            Mostrar entorno gráficamente (default: False)
        log_interval : int, opcional
            Intervalo para mostrar información (default: 10)
            
        Retorna:
        --------
        Dict
            Historia de entrenamiento
        """
        history = {
            'episode_rewards': [],
            'actor_losses': [],
            'critic_losses': [],
            'avg_q_values': []
        }
        
        # Variables para seguimiento de progreso
        best_reward = -float('inf')
        episode_reward_history = []
        step_counter = 0
        
        for episode in range(episodes):
            # Ejecutar un episodio completo
            episode_reward, episode_actor_loss, episode_critic_loss, step_counter = self._run_episode(
                env, max_steps, step_counter, warmup_steps, update_every, render
            )
            
            # Actualizar historial
            history = self._update_history(history, episode_reward, episode_actor_loss, episode_critic_loss)
            
            # Resetear el ruido para el siguiente episodio
            self.noise.reset()
            
            # Actualizar historial de recompensas
            episode_reward_history.append(episode_reward)
            if len(episode_reward_history) > log_interval:
                episode_reward_history.pop(0)
            
            # Registrar progreso
            best_reward = self._log_progress(
                episode, episodes, episode_reward_history, history, log_interval, best_reward
            )
        
        return history
    
    def evaluate(self, env, episodes: int = 10, render: bool = False) -> float:
        """
        Evalúa el agente DDPG en un entorno dado sin exploración.
        
        Parámetros:
        -----------
        env : gym.Env
            Entorno de Gym o compatible
        episodes : int, opcional
            Número de episodios para evaluar (default: 10)
        render : bool, opcional
            Si se debe renderizar el entorno (default: False)
            
        Retorna:
        --------
        float
            Recompensa promedio obtenida durante la evaluación
        """
        rewards = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            state = np.array(state, dtype=np.float32)
            episode_reward = 0
            done = False
            
            while not done:
                # Seleccionar acción determinística (sin ruido)
                action = self.get_action(state, add_noise=False)
                
                # Ejecutar acción
                next_state, reward, done, _, _ = env.step(action)
                next_state = np.array(next_state, dtype=np.float32)
                
                # Renderizar si es necesario
                if render:
                    env.render()
                
                # Actualizar estado y recompensa
                state = next_state
                episode_reward += reward
            
            rewards.append(episode_reward)
            print(f"Episodio de Evaluación {episode+1}/{episodes} - Recompensa: {episode_reward:.2f}")
        
        avg_reward = np.mean(rewards)
        print(f"Recompensa Promedio de Evaluación: {avg_reward:.2f}")
        
        return avg_reward
    
    def save_models(self, actor_path: str, critic_path: str) -> None:
        """
        Guarda los modelos del actor y crítico en archivos separados.
        
        Parámetros:
        -----------
        actor_path : str
            Ruta para guardar el modelo del actor
        critic_path : str
            Ruta para guardar el modelo del crítico
        """
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)
        print(f"Modelos guardados en {actor_path} y {critic_path}")
    
    def load_models(self, actor_path: str, critic_path: str) -> None:
        """
        Carga los modelos del actor y crítico desde archivos.
        
        Parámetros:
        -----------
        actor_path : str
            Ruta para cargar el modelo del actor
        critic_path : str
            Ruta para cargar el modelo del crítico
        """
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)
        
        # Sincronizar modelos target
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        
        print(f"Modelos cargados desde {actor_path} y {critic_path}")
    
    def visualize_training(self, history: Dict, window_size: int = 10) -> None:
        """
        Visualiza los resultados del entrenamiento.
        
        Parámetros:
        -----------
        history : Dict
            Historia de entrenamiento
        window_size : int, opcional
            Tamaño de ventana para suavizado (default: 10)
        """
        # Función para aplicar suavizado
        def smooth(data, window_size):
            kernel = np.ones(window_size) / window_size
            return np.convolve(data, kernel, mode='valid')
        
        # Crear figura con múltiples subplots
        _, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Graficar recompensas
        rewards = history['episode_rewards']
        axs[0, 0].plot(rewards, alpha=0.3, color='blue', label='Raw')
        if len(rewards) > window_size:
            axs[0, 0].plot(range(window_size-1, len(rewards)), smooth(rewards, window_size), 
                         color='blue', label=f'Smoothed (window={window_size})')
        axs[0, 0].set_title('Episode Rewards')
        axs[0, 0].set_xlabel('Episode')
        axs[0, 0].set_ylabel('Reward')
        axs[0, 0].legend()
        axs[0, 0].grid(alpha=0.3)
        
        # Graficar pérdida del actor
        actor_losses = history['actor_losses']
        valid_losses = [l for l in actor_losses if not np.isnan(l)]
        if valid_losses:
            axs[0, 1].plot(valid_losses, alpha=0.3, color='green', label='Raw')
            if len(valid_losses) > window_size:
                axs[0, 1].plot(range(window_size-1, len(valid_losses)), 
                             smooth(valid_losses, window_size), 
                             color='green', label=f'Smoothed (window={window_size})')
        axs[0, 1].set_title('Actor Loss')
        axs[0, 1].set_xlabel('Episode')
        axs[0, 1].set_ylabel('Loss')
        axs[0, 1].legend()
        axs[0, 1].grid(alpha=0.3)
        
        # Graficar pérdida del crítico
        critic_losses = history['critic_losses']
        valid_losses = [l for l in critic_losses if not np.isnan(l)]
        if valid_losses:
            axs[1, 0].plot(valid_losses, alpha=0.3, color='red', label='Raw')
            if len(valid_losses) > window_size:
                axs[1, 0].plot(range(window_size-1, len(valid_losses)), 
                             smooth(valid_losses, window_size), 
                             color='red', label=f'Smoothed (window={window_size})')
        axs[1, 0].set_title('Critic Loss')
        axs[1, 0].set_xlabel('Episode')
        axs[1, 0].set_ylabel('Loss')
        axs[1, 0].legend()
        axs[1, 0].grid(alpha=0.3)
        
        # Graficar valores Q promedio
        q_values = history['avg_q_values']
        valid_q = [q for q in q_values if not np.isnan(q)]
        if valid_q:
            axs[1, 1].plot(valid_q, alpha=0.3, color='purple', label='Raw')
            if len(valid_q) > window_size:
                axs[1, 1].plot(range(window_size-1, len(valid_q)), 
                             smooth(valid_q, window_size), 
                             color='purple', label=f'Smoothed (window={window_size})')
        axs[1, 1].set_title('Average Q Values')
        axs[1, 1].set_xlabel('Episode')
        axs[1, 1].set_ylabel('Q Value')
        axs[1, 1].legend()
        axs[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()