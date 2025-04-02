import os, sys
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.training import train_state
import optax
from typing import Dict, List, Tuple, Any, Optional, Callable, Union, Sequence
from collections import deque
import random
from functools import partial
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from models.config import DQN_CONFIG


class ReplayBuffer:
    """
    Buffer de experiencias para el algoritmo DQN.
    
    Almacena transiciones (state, action, reward, next_state, done)
    y permite muestrear lotes de manera aleatoria para el entrenamiento.
    
    Parámetros:
    -----------
    capacity : int, opcional
        Capacidad máxima del buffer (default: 10000)
    """
    def __init__(self, capacity: int = 10000) -> None:
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state: np.ndarray, action: int, reward: float, 
           next_state: np.ndarray, done: float) -> None:
        """
        Añade una transición al buffer.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado actual
        action : int
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
            np.array(actions, dtype=np.int32),
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


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Buffer de experiencias con muestreo prioritario.
    
    Prioriza experiencias basadas en el TD error para muestreo eficiente.
    
    Parámetros:
    -----------
    capacity : int
        Capacidad máxima del buffer
    """
    def __init__(self, capacity: int) -> None:
        super().__init__(capacity)
        self.priorities = np.zeros(capacity, dtype=np.float32) + 1e-5
        self.pos = 0
    
    def add(self, state: np.ndarray, action: int, reward: float, 
           next_state: np.ndarray, done: float) -> None:
        """
        Añade una transición al buffer con prioridad máxima.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado actual
        action : int
            Acción tomada
        reward : float
            Recompensa recibida
        next_state : np.ndarray
            Estado siguiente
        done : float
            Indicador de fin de episodio (1.0 si terminó, 0.0 si no)
        """
        max_priority = max(np.max(self.priorities), 1e-5)
        
        if len(self.buffer) < self.buffer.maxlen:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.buffer.maxlen
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[np.ndarray, np.ndarray, 
                                                                np.ndarray, np.ndarray, 
                                                                np.ndarray, List[int], np.ndarray]:
        """
        Muestrea un lote basado en prioridades.
        
        Parámetros:
        -----------
        batch_size : int
            Tamaño del lote
        beta : float, opcional
            Factor para corrección de sesgo (0-1) (default: 0.4)
            
        Retorna:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int], np.ndarray]
            (states, actions, rewards, next_states, dones, índices, pesos de importancia)
        """
        if len(self.buffer) < batch_size:
            idx = list(range(len(self.buffer)))
        else:
            # Calcular probabilidades de muestreo basadas en prioridad
            priorities = self.priorities[:len(self.buffer)]
            probabilities = priorities ** DQN_CONFIG['priority_alpha']
            probabilities /= np.sum(probabilities)
            
            # Muestreo según distribución
            rng = np.random.default_rng(seed=42)
            idx = rng.choice(len(self.buffer), batch_size, p=probabilities, replace=False).tolist()
        
        # Extraer batch
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in idx:
            s, a, r, ns, d = self.buffer[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
        
        # Calcular pesos de importancia (corrección del sesgo)
        weights = np.zeros(batch_size, dtype=np.float32)
        priorities = self.priorities[idx]
        # Evitar división por cero
        probabilities = priorities / np.sum(self.priorities[:len(self.buffer)])
        weights = (len(self.buffer) * probabilities) ** -beta
        weights /= np.max(weights)  # Normalizar
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            idx,
            np.array(weights, dtype=np.float32)
        )
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray) -> None:
        """
        Actualiza las prioridades para los índices dados.
        
        Parámetros:
        -----------
        indices : List[int]
            Índices de las transiciones a actualizar
        priorities : np.ndarray
            Nuevas prioridades
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority


class QNetwork(nn.Module):
    """
    Red Q para DQN que mapea estados a valores Q.
    
    Implementa una arquitectura flexible para estimación de valores Q-state-action.
    
    Parámetros:
    -----------
    action_dim : int
        Dimensión del espacio de acciones
    hidden_units : Sequence[int], opcional
        Unidades en capas ocultas (default: None)
    dueling : bool, opcional
        Si usar arquitectura dueling (default: False)
    activation : str, opcional
        Función de activación a utilizar (default: 'relu')
    dropout_rate : float, opcional
        Tasa de dropout para regularización (default: 0.0)
    """
    action_dim: int
    hidden_units: Optional[Sequence[int]] = None
    dueling: bool = False
    activation: str = 'relu'
    dropout_rate: float = 0.0
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Pasa la entrada por la red Q.
        
        Parámetros:
        -----------
        x : jnp.ndarray
            Tensor de entrada (estados)
        training : bool, opcional
            Indica si está en modo entrenamiento (default: False)
            
        Retorna:
        --------
        jnp.ndarray
            Valores Q para cada acción
        """
        # Valores predeterminados para capas ocultas
        if self.hidden_units is None:
            hidden_units = DQN_CONFIG['hidden_units']
        else:
            hidden_units = self.hidden_units
            
        # Función de activación
        activation_fn = getattr(nn, self.activation)
        
        # Capas para el procesamiento del estado (feature extractor)
        for i, units in enumerate(hidden_units):
            x = nn.Dense(units, name=f'feature_dense_{i}')(x)
            x = nn.LayerNorm(epsilon=DQN_CONFIG['epsilon'], name=f'feature_ln_{i}')(x)
            x = activation_fn(x)
            if self.dropout_rate > 0 and training:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        # Para arquitectura Dueling DQN
        if self.dueling:
            # Ventaja: un valor por acción
            advantage = x
            for i, units in enumerate(hidden_units[-2:]):
                advantage = nn.Dense(units, name=f'advantage_dense_{i}')(advantage)
                advantage = activation_fn(advantage)
            advantage = nn.Dense(self.action_dim, name='advantage')(advantage)
            
            # Valor del estado: un valor único
            value = x
            for i, units in enumerate(hidden_units[-2:]):
                value = nn.Dense(units, name=f'value_dense_{i}')(value)
                value = activation_fn(value)
            value = nn.Dense(1, name='value')(value)
            
            # Combinar ventaja y valor (restando la media de ventajas)
            q_values = value + (advantage - advantage.mean(axis=-1, keepdims=True))
        else:
            # DQN clásica: predecir valor Q para cada acción
            q_values = nn.Dense(self.action_dim, name='q_values')(x)
            
        return q_values


class DQNTrainState(train_state.TrainState):
    """
    Estado de entrenamiento para DQN que extiende TrainState de Flax.
    
    Incluye el modelo target además del modelo principal.
    
    Atributos:
    ----------
    target_params : Any
        Parámetros del modelo target
    """
    target_params: Any
    rng: jax.random.PRNGKey


class DQN:
    """
    Implementación del algoritmo Deep Q-Network (DQN) usando JAX.
    
    Incluye mecanismos de Experience Replay y Target Network para
    mejorar la estabilidad del aprendizaje.
    
    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    config : Optional[Dict[str, Any]], opcional
        Configuración personalizada (default: None)
    hidden_units : Optional[List[int]], opcional
        Unidades en capas ocultas (default: None)
    seed : int, opcional
        Semilla para los generadores de números aleatorios (default: 42)
    """
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int,
        config: Optional[Dict[str, Any]] = None,
        hidden_units: Optional[List[int]] = None,
        seed: int = 42
    ) -> None:
        # Use provided config or default
        self.config = config or DQN_CONFIG
        
        # Extract configuration parameters
        learning_rate = self.config.get('learning_rate', 0.001)
        self.gamma = self.config.get('gamma', 0.99)
        self.epsilon_start = self.config.get('epsilon_start', 1.0)
        self.epsilon_end = self.config.get('epsilon_end', 0.01)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)
        buffer_capacity = self.config.get('buffer_capacity', 10000)
        self.batch_size = self.config.get('batch_size', 64)
        self.target_update_freq = self.config.get('target_update_freq', 100)
        dueling = self.config.get('dueling', False)
        self.double = self.config.get('double', False)
        self.prioritized = self.config.get('prioritized', False)
        dropout_rate = self.config.get('dropout_rate', 0.0)
        
        # Parámetros del entorno y del modelo
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = self.epsilon_start
        
        # Cantidad de actualizaciones realizadas
        self.update_counter = 0
        
        # Configurar semillas para reproducibilidad
        self.seed = seed
        self.rng = jax.random.PRNGKey(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Dividir la clave para inicialización y uso posterior
        self.rng, init_rng = jax.random.split(self.rng)
        
        # Valores predeterminados para capas ocultas
        if hidden_units is None:
            self.hidden_units = DQN_CONFIG['hidden_units']
        else:
            self.hidden_units = hidden_units
        
        # Crear modelo Q
        self.q_network = QNetwork(
            action_dim=action_dim,
            hidden_units=self.hidden_units,
            dueling=dueling,
            activation=DQN_CONFIG['activation'],
            dropout_rate=dropout_rate
        )
        
        # Inicializar parámetros con una muestra de estado
        dummy_state = jnp.zeros((1, state_dim))
        params = self.q_network.init(init_rng, dummy_state)
        
        # Crear optimizador
        tx = optax.adam(learning_rate=learning_rate)
        
        # Crear estado de entrenamiento
        self.state = DQNTrainState(
            step=0,
            apply_fn=self.q_network.apply,
            params=params,
            target_params=params,  # Inicializar target = modelo principal
            tx=tx,
            rng=init_rng
        )
        
        # Buffer de experiencias
        if self.prioritized:
            # Con prioridad de muestreo basada en TD-error
            self.replay_buffer = PrioritizedReplayBuffer(buffer_capacity)
            self.alpha = DQN_CONFIG['priority_alpha']
            self.beta = DQN_CONFIG['priority_beta']
            self.beta_increment = DQN_CONFIG['priority_beta_increment']
        else:
            # Buffer uniforme clásico
            self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Métricas acumuladas
        self.loss_sum = 0.0
        self.q_value_sum = 0.0
        self.updates_count = 0
        
        # Precompilar funciones para entrenamiento
        self._compile_jitted_functions()
        
    def _compile_jitted_functions(self) -> None:
        """
        Compila versiones JIT de las funciones principales para acelerar la ejecución.
        """
        self.update_target_jit = jax.jit(self._update_target)
        self.train_step_jit = jax.jit(self._train_step)
        
    def update_target_network(self) -> None:
        """
        Actualiza los pesos del modelo target con los del modelo principal.
        """
        self.state = self.update_target_jit(self.state)
    
    def _update_target(self, state: DQNTrainState) -> DQNTrainState:
        """
        Actualiza los parámetros target.
        
        Parámetros:
        -----------
        state : dqn_train_state
            Estado actual del entrenamiento
            
        Retorna:
        --------
        dqn_train_state
            Nuevo estado con parámetros target actualizados
        """
        return state.replace(target_params=state.params)
    
    def _train_step(self, state: DQNTrainState, 
                  batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
                  importance_weights: Optional[jnp.ndarray] = None) -> Tuple[DQNTrainState, jnp.ndarray, jnp.ndarray]:
        """
        Realiza un paso de entrenamiento para actualizar la red Q.
        
        Parámetros:
        -----------
        state : dqn_train_state
            Estado actual del entrenamiento
        batch : Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
            (estados, acciones, recompensas, siguientes_estados, terminados)
        importance_weights : Optional[jnp.ndarray], opcional
            Pesos de importancia para muestreo prioritario (default: None)
            
        Retorna:
        --------
        Tuple[dqn_train_state, jnp.ndarray, jnp.ndarray]
            (nuevo_estado, pérdida, td_errors)
        """
        states, actions, rewards, next_states, dones = batch
        
        # Dividir rng para posibles necesidades estocásticas
        new_rng, dropout_rng = jax.random.split(state.rng)
        
        # Función de pérdida y su gradiente
        def loss_fn(params):
            # Q-values para los estados actuales
            q_values = state.apply_fn(params, states, rngs={'dropout': dropout_rng})
            q_values_selected = q_values[jnp.arange(q_values.shape[0]), actions]
            
            # Q-values objetivos para los siguientes estados
            if self.double:
                # Double DQN: seleccionar acción con red primaria
                next_q_values_online = state.apply_fn(params, next_states)
                next_actions = jnp.argmax(next_q_values_online, axis=1)
                next_q_values_target = state.apply_fn(state.target_params, next_states)
                next_q_values = next_q_values_target[jnp.arange(next_q_values_target.shape[0]), next_actions]
            else:
                # DQN estándar: target Q-network para seleccionar y evaluar
                next_q_values = state.apply_fn(state.target_params, next_states)
                next_q_values = jnp.max(next_q_values, axis=1)
                
            # Calcular targets usando ecuación de Bellman
            targets = rewards + (1.0 - dones) * self.gamma * next_q_values
            
            # Calcular TD-error
            td_errors = targets - q_values_selected
            
            # Aplicar pesos de importancia para PER si es necesario
            if importance_weights is not None:
                loss = jnp.mean(importance_weights * jnp.square(td_errors))
            else:
                loss = jnp.mean(optax.huber_loss(q_values_selected, targets))
            
            metrics = {
                'loss': loss,
                'q_values': jnp.mean(q_values_selected),
                'td_errors': td_errors
            }
            
            return loss, metrics
        
        # Calcular gradientes y actualizar parámetros
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, metrics), grads = grad_fn(state.params)
        new_state = state.apply_gradients(grads=grads, rng=new_rng)
        
        return new_state, loss, metrics['td_errors']
    
    def get_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Obtiene una acción usando la política epsilon-greedy.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado actual del entorno
        epsilon : float, opcional
            Probabilidad de exploración (0-1) (default: 0.0)
            
        Retorna:
        --------
        int
            Acción seleccionada según la política
        """
        # Dividir clave para exploración
        self.rng, explore_rng = jax.random.split(self.rng)
        
        # Exploración
        if jax.random.uniform(explore_rng) < epsilon:
            # Explorar: acción aleatoria
            self.rng, action_rng = jax.random.split(self.rng)
            return int(jax.random.randint(action_rng, (), 0, self.action_dim))
        else:
            # Explotar: mejor acción según la red
            state_tensor = jnp.array([state], dtype=jnp.float32)
            q_values = self.q_network.apply(self.state.params, state_tensor)
            return int(jnp.argmax(q_values[0]))
    
    def _sample_batch(self) -> Tuple:
        """
        Muestrea un lote del buffer de experiencias.
        
        Retorna:
        --------
        Tuple
            Datos muestreados, según el tipo de buffer
        """
        if self.prioritized:
            (states, actions, rewards, next_states, dones, 
             indices, importance_weights) = self.replay_buffer.sample(
                 self.batch_size, self.beta)
            self.beta = min(1.0, self.beta + self.beta_increment)
            
            # Convertir a JAX arrays
            batch = (
                jnp.array(states, dtype=jnp.float32),
                jnp.array(actions, dtype=jnp.int32),
                jnp.array(rewards, dtype=jnp.float32),
                jnp.array(next_states, dtype=jnp.float32),
                jnp.array(dones, dtype=jnp.float32)
            )
            importance_weights_array = jnp.array(importance_weights, dtype=jnp.float32)
            return batch, indices, importance_weights_array
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(
                self.batch_size)
            
            # Convertir a JAX arrays
            batch = (
                jnp.array(states, dtype=jnp.float32),
                jnp.array(actions, dtype=jnp.int32),
                jnp.array(rewards, dtype=jnp.float32),
                jnp.array(next_states, dtype=jnp.float32),
                jnp.array(dones, dtype=jnp.float32)
            )
            return batch, None, None
    
    def _update_model(self, episode_loss: List[float], update_every: int, update_after: int) -> None:
        """
        Actualiza el modelo si es necesario.
        
        Parámetros:
        -----------
        episode_loss : List[float]
            Lista para almacenar pérdidas del episodio
        update_every : int
            Frecuencia de actualización
        update_after : int
            Pasos antes de empezar a actualizar la red
        """
        # Entrenar modelo si hay suficientes datos
        if (len(self.replay_buffer) > self.batch_size and 
            self.update_counter >= update_after and 
            self.update_counter % update_every == 0):
            
            # Muestrear batch
            if self.prioritized:
                batch, indices, importance_weights = self._sample_batch()
            else:
                batch, _, _ = self._sample_batch()
                importance_weights = None
            
            # Entrenar red
            self.state, loss, td_errors = self.train_step_jit(self.state, batch, importance_weights)
            episode_loss.append(float(loss))
            
            # Actualizar métricas acumuladas
            self.loss_sum += float(loss)
            # Calcular q_values del batch
            states, actions = batch[0], batch[1]
            q_values = self.q_network.apply(self.state.params, states)
            q_values_selected = q_values[jnp.arange(q_values.shape[0]), actions]
            self.q_value_sum += float(jnp.mean(q_values_selected))
            self.updates_count += 1
            
            # Actualizar prioridades si es PER
            if self.prioritized and indices is not None:
                priorities = np.abs(np.array(td_errors)) + 1e-6
                self.replay_buffer.update_priorities(indices, priorities)
                
        # Actualizar target network periódicamente
        if self.update_counter % self.target_update_freq == 0 and self.update_counter > 0:
            self.update_target_network()
        
        self.update_counter += 1
            
    def _run_episode(self, env: Any, max_steps: int, render: bool, 
                   update_every: int, update_after: int) -> Tuple[float, List[float]]:
        """
        Ejecuta un episodio completo de entrenamiento.
        
        Parámetros:
        -----------
        env : Any
            Entorno donde ejecutar el episodio
        max_steps : int
            Pasos máximos por episodio
        render : bool
            Si se debe renderizar el entorno
        update_every : int
            Frecuencia de actualización del modelo
        update_after : int
            Pasos antes de empezar a actualizar la red
            
        Retorna:
        --------
        Tuple[float, List[float]]
            (episode_reward, episode_loss)
        """
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        episode_reward = 0.0
        episode_loss = []
        
        for _ in range(max_steps):
            # Seleccionar acción
            action = self.get_action(state, self.epsilon)
            
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
            
            # Actualizar modelo
            self._update_model(episode_loss, update_every, update_after)
            
            if done:
                break
                
        return episode_reward, episode_loss
        
    def _update_history(self, history: Dict, episode_reward: float, episode_loss: List[float], 
                      episode_reward_history: List[float], log_interval: int) -> List[float]:
        """
        Actualiza el historial de entrenamiento y métricas.
        
        Parámetros:
        -----------
        history : Dict
            Historial de entrenamiento a actualizar
        episode_reward : float
            Recompensa del episodio
        episode_loss : List[float]
            Lista de pérdidas del episodio
        episode_reward_history : List[float]
            Historial reciente de recompensas
        log_interval : int
            Intervalo para mostrar información
            
        Retorna:
        --------
        List[float]
            Historial de recompensas actualizado
        """
        # Almacenar métricas
        history['episode_rewards'].append(episode_reward)
        history['epsilons'].append(self.epsilon)
        
        if episode_loss:
            history['losses'].append(float(np.mean(episode_loss)))
        else:
            history['losses'].append(0.0)
            
        # Calcular y guardar promedio del valor Q
        if self.updates_count > 0:
            avg_q_value = self.q_value_sum / self.updates_count
            self.q_value_sum = 0.0
            self.updates_count = 0
        else:
            avg_q_value = 0.0
            
        history['avg_q_values'].append(avg_q_value)
        
        # Actualizar epsilon (decaimiento)
        self.epsilon = max(
            self.epsilon_end, 
            self.epsilon * self.epsilon_decay
        )
        
        # Guardar últimas recompensas para promedio
        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > log_interval:
            episode_reward_history.pop(0)
            
        return episode_reward_history
        
    def train(self, env: Any, episodes: int = 1000, max_steps: int = 1000, 
             update_after: int = 1000, update_every: int = 4, 
             render: bool = False, log_interval: int = 10) -> Dict:
        """
        Entrena el agente DQN en un entorno dado.
        
        Parámetros:
        -----------
        env : Any
            Entorno de Gym o compatible
        episodes : int, opcional
            Número máximo de episodios (default: 1000)
        max_steps : int, opcional
            Pasos máximos por episodio (default: 1000)
        update_after : int, opcional
            Pasos antes de empezar a actualizar la red (default: 1000)
        update_every : int, opcional
            Frecuencia de actualización (default: 4)
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
            'losses': [],
            'epsilons': [],
            'avg_q_values': []
        }
        
        # Variables para seguimiento de progreso
        best_reward = -float('inf')
        episode_reward_history = []
        
        for episode in range(episodes):
            # Ejecutar un episodio
            episode_reward, episode_loss = self._run_episode(
                env, max_steps, render, update_every, update_after)
            
            # Actualizar historial
            episode_reward_history = self._update_history(
                history, episode_reward, episode_loss, episode_reward_history, log_interval)
            
            # Mostrar progreso periódicamente
            if (episode + 1) % log_interval == 0:
                avg_reward = np.mean(episode_reward_history)
                print(f"Episodio {episode+1}/{episodes} - Recompensa Promedio: {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.4f}, Pérdida: {history['losses'][-1]:.4f}")
                
                # Guardar mejor modelo
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    print(f"Nuevo mejor modelo con recompensa: {best_reward:.2f}")
        
        return history
    
    def evaluate(self, env: Any, episodes: int = 10, render: bool = False) -> float:
        """
        Evalúa el agente DQN entrenado en un entorno.
        
        Parámetros:
        -----------
        env : Any
            Entorno de Gym o compatible
        episodes : int, opcional
            Número de episodios para evaluar (default: 10)
        render : bool, opcional
            Mostrar entorno gráficamente (default: False)
            
        Retorna:
        --------
        float
            Recompensa promedio durante la evaluación
        """
        rewards = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            state = np.array(state, dtype=np.float32)
            episode_reward = 0
            done = False
            step = 0
            
            while not done:
                # Seleccionar acción determinística (sin exploración)
                action = self.get_action(state)
                
                # Ejecutar acción
                next_state, reward, done, _, _ = env.step(action)
                next_state = np.array(next_state, dtype=np.float32)
                
                # Renderizar si es necesario
                if render:
                    env.render()
                
                # Actualizar estado y recompensa
                state = next_state
                episode_reward += reward
                step += 1
            
            rewards.append(episode_reward)
            print(f"Episodio de evaluación {episode+1}/{episodes} - "
                  f"Recompensa: {episode_reward:.2f}, Pasos: {step}")
        
        avg_reward = np.mean(rewards)
        print(f"Recompensa promedio de evaluación: {avg_reward:.2f}")
        
        return avg_reward
    
    def save_model(self, filepath: str) -> None:
        """
        Guarda los parámetros del modelo en un archivo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta donde guardar el modelo
        """
        import flax.serialization
        with open(filepath, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.state.params))
        print(f"Modelo guardado en {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Carga los parámetros del modelo desde un archivo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta desde donde cargar el modelo
        """
        import flax.serialization
        with open(filepath, 'rb') as f:
            params = flax.serialization.from_bytes(self.state.params, f.read())
        
        # Actualizar parámetros del modelo y del target
        self.state = self.state.replace(params=params, target_params=params)
        print(f"Modelo cargado desde {filepath}")
    
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
                         color='blue', label=f'Suavizado (ventana={window_size})')
        axs[0, 0].set_title('Recompensas por Episodio')
        axs[0, 0].set_xlabel('Episodio')
        axs[0, 0].set_ylabel('Recompensa')
        axs[0, 0].legend()
        axs[0, 0].grid(alpha=0.3)
        
        # Graficar epsilon
        epsilons = history['epsilons']
        axs[0, 1].plot(epsilons, color='green')
        axs[0, 1].set_title('Valor Epsilon')
        axs[0, 1].set_xlabel('Episodio')
        axs[0, 1].set_ylabel('Epsilon')
        axs[0, 1].grid(alpha=0.3)
        
        # Graficar pérdida
        losses = history['losses']
        axs[1, 0].plot(losses, alpha=0.3, color='red', label='Raw')
        if len(losses) > window_size:
            axs[1, 0].plot(range(window_size-1, len(losses)), smooth(losses, window_size), 
                         color='red', label=f'Suavizado (ventana={window_size})')
        axs[1, 0].set_title('Pérdida')
        axs[1, 0].set_xlabel('Episodio')
        axs[1, 0].set_ylabel('Pérdida')
        axs[1, 0].legend()
        axs[1, 0].grid(alpha=0.3)
        
        # Graficar valores Q promedio
        q_values = history['avg_q_values']
        axs[1, 1].plot(q_values, alpha=0.3, color='purple', label='Raw')
        if len(q_values) > window_size:
            axs[1, 1].plot(range(window_size-1, len(q_values)), smooth(q_values, window_size), 
                         color='purple', label=f'Suavizado (ventana={window_size})')
        axs[1, 1].set_title('Valores Q Promedio')
        axs[1, 1].set_xlabel('Episodio')
        axs[1, 1].set_ylabel('Valor Q')
        axs[1, 1].legend()
        axs[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()