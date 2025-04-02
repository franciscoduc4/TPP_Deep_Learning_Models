import os, sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Conv1D, Flatten, Concatenate,
    BatchNormalization, Dropout, LayerNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from collections import deque
import random

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from models.config import DQN_CONFIG


class ReplayBuffer:
    """
    Buffer de experiencias para el algoritmo DQN.
    
    Almacena transiciones (state, action, reward, next_state, done)
    y permite muestrear lotes de manera aleatoria para el entrenamiento.
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Añade una transición al buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Muestrea un lote aleatorio de transiciones."""
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
    
    def __len__(self):
        """Retorna la cantidad de transiciones almacenadas."""
        return len(self.buffer)


class QNetwork(Model):
    """
    Red Q para DQN que mapea estados a valores Q.
    
    Implementa una arquitectura flexible para estimación de valores Q-state-action.
    """
    def __init__(self, state_dim, action_dim, hidden_units=None, dueling=False):
        super(QNetwork, self).__init__()
        
        # Valores predeterminados para capas ocultas
        if hidden_units is None:
            hidden_units = DQN_CONFIG['hidden_units']
        
        self.dueling = dueling
        self.action_dim = action_dim
        
        # Capas para el procesamiento del estado
        self.feature_layers = []
        for i, units in enumerate(hidden_units):
            self.feature_layers.append(Dense(
                units, 
                activation=DQN_CONFIG['activation'],
                name=f'feature_dense_{i}'
            ))
            self.feature_layers.append(LayerNormalization(
                epsilon=DQN_CONFIG['epsilon'],
                name=f'feature_ln_{i}'
            ))
            self.feature_layers.append(Dropout(
                DQN_CONFIG['dropout_rate'],
                name=f'feature_dropout_{i}'
            ))
        
        # Para arquitectura Dueling DQN
        if dueling:
            # Ventaja: un valor por acción
            self.advantage_layers = []
            for i, units in enumerate(hidden_units[-2:]):
                self.advantage_layers.append(Dense(
                    units, 
                    activation=DQN_CONFIG['activation'],
                    name=f'advantage_dense_{i}'
                ))
            self.advantage_output = Dense(action_dim, name='advantage')
            
            # Valor de estado: un valor único
            self.value_layers = []
            for i, units in enumerate(hidden_units[-2:]):
                self.value_layers.append(Dense(
                    units, 
                    activation=DQN_CONFIG['activation'],
                    name=f'value_dense_{i}'
                ))
            self.value_output = Dense(1, name='value')
        else:
            # DQN clásica: predecir valor Q para cada acción
            self.q_output = Dense(action_dim, name='q_values')
    
    def _apply_layers(self, x, layers, training=False):
        """Helper method to apply a sequence of layers to input tensor."""
        for layer in layers:
            if isinstance(layer, Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        return x
    
    def _process_dueling_network(self, x, training=False):
        """Process inputs through dueling network architecture."""
        # Ventaja para cada acción
        advantage = self._apply_layers(x, self.advantage_layers, training)
        advantage = self.advantage_output(advantage)
        
        # Valor del estado
        value = self._apply_layers(x, self.value_layers, training)
        value = self.value_output(value)
        
        # Combinar ventaja y valor (restando la media de ventajas)
        return value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
    
    def call(self, inputs, training=False):
        # Procesar características de estado
        x = self._apply_layers(inputs, self.feature_layers, training)
        
        # Aplicar arquitectura dueling o estándar
        if self.dueling:
            return self._process_dueling_network(x, training)
        else:
            # DQN estándar
            return self.q_output(x)
    
    def get_action(self, state, epsilon=0.0):
        """
        Obtiene una acción usando la política epsilon-greedy.
        
        Args:
            state: Estado actual del entorno
            epsilon: Probabilidad de exploración (0-1)
            
        Returns:
            Acción seleccionada según la política
        """
        rng = np.random.default_rng(seed=42)
        if rng.random() < epsilon:
            # Explorar: acción aleatoria
            return rng.integers(0, self.action_dim)
        else:
            # Explotar: mejor acción según la red
            state = tf.convert_to_tensor([state], dtype=tf.float32)
            q_values = self(state)
            return tf.argmax(q_values[0]).numpy()


class DQN:
    """
    Implementación del algoritmo Deep Q-Network (DQN).
    
    Incluye mecanismos de Experience Replay y Target Network para
    mejorar la estabilidad del aprendizaje.
    """
    def __init__(
        self, 
        state_dim, 
        action_dim,
        config=None,
        hidden_units=None,
    ):
        # Use provided config or default DQN_CONFIG
        self.config = config or DQN_CONFIG
        
        # Extract configuration parameters
        learning_rate = self.config.get('learning_rate', 0.001)
        gamma = self.config.get('gamma', 0.99)
        epsilon_start = self.config.get('epsilon_start', 1.0)
        epsilon_end = self.config.get('epsilon_end', 0.01)
        epsilon_decay = self.config.get('epsilon_decay', 0.995)
        buffer_capacity = self.config.get('buffer_capacity', 10000)
        batch_size = self.config.get('batch_size', 64)
        target_update_freq = self.config.get('target_update_freq', 100)
        dueling = self.config.get('dueling', False)
        double = self.config.get('double', False)
        prioritized = self.config.get('prioritized', False)
        # Parámetros del entorno y del modelo
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.dueling = dueling
        self.double = double
        self.prioritized = prioritized
        
        # Cantidad de actualizaciones realizadas
        self.update_counter = 0
        
        # Valores predeterminados para capas ocultas
        if hidden_units is None:
            self.hidden_units = DQN_CONFIG['hidden_units']
        else:
            self.hidden_units = hidden_units
        
        # Crear modelo Q y modelo Q Target
        self.q_network = QNetwork(state_dim, action_dim, self.hidden_units, dueling)
        self.target_q_network = QNetwork(state_dim, action_dim, self.hidden_units, dueling)
        
        # Actualizar pesos del target para que sean iguales al modelo principal
        self.update_target_network()
        
        # Optimizador y función de pérdida
        self.optimizer = Adam(learning_rate=learning_rate)
        self.loss_fn = Huber()  # Menos sensible a outliers que MSE
        
        # Buffer de experiencias
        if prioritized:
            # Con prioridad de muestreo basada en TD-error
            self.replay_buffer = PrioritizedReplayBuffer(buffer_capacity)
            self.alpha = DQN_CONFIG['priority_alpha']
            self.beta = DQN_CONFIG['priority_beta']
            self.beta_increment = DQN_CONFIG['priority_beta_increment']
        else:
            # Buffer uniforme clásico
            self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Métricas
        self.loss_metric = tf.keras.metrics.Mean('loss')
        self.q_metric = tf.keras.metrics.Mean('q_value')
    
    def update_target_network(self):
        """Actualiza los pesos del modelo target con los del modelo principal."""
        self.target_q_network.set_weights(self.q_network.get_weights())
    
    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones, importance_weights=None):
        """
        Realiza un paso de entrenamiento para actualizar la red Q.
        
        Args:
            states: Estados observados
            actions: Acciones tomadas
            rewards: Recompensas recibidas
            next_states: Estados siguientes
            dones: Indicadores de fin de episodio
            importance_weights: Pesos de importancia para PER
            
        Returns:
            Pérdida del paso de entrenamiento y TD-errors
        """
        # Convertir acciones a índices one-hot para gather
        action_indices = tf.one_hot(actions, self.action_dim)
        
        with tf.GradientTape() as tape:
            # Q-values para los estados actuales
            q_values = self.q_network(states, training=True)
            q_values_for_actions = tf.reduce_sum(q_values * action_indices, axis=1)
            
            # Q-values objetivos para los siguientes estados
            if self.double:
                # Double DQN: seleccionar acción con red primaria
                next_actions = tf.argmax(self.q_network(next_states), axis=1)
                next_action_indices = tf.one_hot(next_actions, self.action_dim)
                next_q_values = self.target_q_network(next_states)
                next_q_values = tf.reduce_sum(next_q_values * next_action_indices, axis=1)
            else:
                # DQN estándar: target Q-network para seleccionar y evaluar
                next_q_values = tf.reduce_max(self.target_q_network(next_states), axis=1)
                
            # Calcular targets usando Bellman equation
            targets = rewards + (1 - dones) * self.gamma * next_q_values
            
            # Calcular TD-error
            td_errors = targets - q_values_for_actions
            
            # Aplicar pesos de importancia para PER si es necesario
            if importance_weights is not None:
                loss = tf.reduce_mean(importance_weights * tf.square(td_errors), axis=0)
            else:
                loss = self.loss_fn(targets, q_values_for_actions)
        
        # Calcular gradientes y actualizar pesos
        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))
        
        # Actualizar métricas
        self.loss_metric.update_state(loss)
        self.q_metric.update_state(q_values_for_actions)
        
        return loss, td_errors
    
    def _sample_batch(self):
        """Helper method to sample a batch from replay buffer."""
        if self.prioritized:
            (states, actions, rewards, next_states, dones, 
             indices, importance_weights) = self.replay_buffer.sample(
                 self.batch_size, self.beta)
            self.beta = min(1.0, self.beta + self.beta_increment)
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(
                self.batch_size)
            importance_weights = None
            indices = None
            
        return states, actions, rewards, next_states, dones, indices, importance_weights
    
    def _update_model(self, episode_loss, update_every):
        """Helper method to update the model if necessary."""
        # Entrenar modelo si hay suficientes datos
        if (len(self.replay_buffer) > self.batch_size and 
            self.update_counter >= self.update_after and 
            self.update_counter % update_every == 0):
            
            # Muestrear batch
            states, actions, rewards, next_states, dones, indices, importance_weights = self._sample_batch()
            
            # Entrenar red
            loss, td_errors = self.train_step(
                states, actions, rewards, next_states, dones, importance_weights)
            episode_loss.append(loss.numpy())
            
            # Actualizar prioridades si es PER
            if self.prioritized:
                priorities = np.abs(td_errors.numpy()) + 1e-6
                self.replay_buffer.update_priorities(indices, priorities)
                
        # Actualizar target network periódicamente
        if self.update_counter % self.target_update_freq == 0 and self.update_counter > 0:
            self.update_target_network()
        
        self.update_counter += 1
            
    def _run_episode(self, env, max_steps, render, update_every):
        """Helper method to run a single episode."""
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        episode_reward = 0
        episode_loss = []
        
        for _ in range(max_steps):
            # Seleccionar acción
            action = self.q_network.get_action(state, self.epsilon)
            
            # Ejecutar acción
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)
            
            # Renderizar si es necesario
            if render:
                env.render()
            
            # Guardar transición en buffer
            self.replay_buffer.add(state, action, reward, next_state, done)
            
            # Actualizar estado y recompensa
            state = next_state
            episode_reward += reward
            
            # Actualizar modelo
            self._update_model(episode_loss, update_every)
            
            if done:
                break
                
        return episode_reward, episode_loss
        
    def _update_history(self, history, episode_reward, episode_loss, episode_reward_history, log_interval):
        """Helper method to update training history and metrics."""
        # Almacenar métricas
        history['episode_rewards'].append(episode_reward)
        history['epsilons'].append(self.epsilon)
        if episode_loss:
            history['losses'].append(np.mean(episode_loss))
        else:
            history['losses'].append(0)
        history['avg_q_values'].append(self.q_metric.result().numpy())
        
        # Actualizar epsilon (decaimiento)
        self.epsilon = max(
            self.epsilon_end, 
            self.epsilon * self.epsilon_decay
        )
        
        # Resetear métricas
        self.loss_metric.reset_states()
        self.q_metric.reset_states()
        
        # Guardar últimas recompensas para promedio
        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > log_interval:
            episode_reward_history.pop(0)
            
        return episode_reward_history
        
    def train(self, env, episodes=1000, max_steps=1000, update_after=1000, 
             update_every=4, render=False, log_interval=10):
        """
        Entrena el agente DQN en un entorno dado.
        
        Args:
            env: Entorno de Gym o compatible
            episodes: Número máximo de episodios
            max_steps: Pasos máximos por episodio
            update_after: Pasos antes de empezar a actualizar la red
            update_every: Frecuencia de actualización
            render: Mostrar entorno gráficamente
            log_interval: Intervalo para mostrar información
            
        Returns:
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
        self.update_after = update_after
        
        for episode in range(episodes):
            # Ejecutar un episodio
            episode_reward, episode_loss = self._run_episode(env, max_steps, render, update_every)
            
            # Actualizar historial
            episode_reward_history = self._update_history(
                history, episode_reward, episode_loss, episode_reward_history, log_interval)
            
            # Mostrar progreso periódicamente
            if (episode + 1) % log_interval == 0:
                avg_reward = np.mean(episode_reward_history)
                print(f"Episode {episode+1}/{episodes} - Average Reward: {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.4f}, Loss: {history['losses'][-1]:.4f}")
                
                # Guardar mejor modelo
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    print(f"Nuevo mejor modelo con recompensa: {best_reward:.2f}")
        
        return history
    
    def save_model(self, filepath):
        """Guarda el modelo en un archivo."""
        self.q_network.save_weights(filepath)
        
    def load_model(self, filepath):
        """Carga el modelo desde un archivo."""
        # Asegurarse de que el modelo está construido primero
        dummy_state = np.zeros((1, self.state_dim))
        self.q_network(dummy_state)
        self.q_network.load_weights(filepath)
        # Actualizar target network
        self.update_target_network()


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Buffer de experiencias con muestreo prioritario.
    
    Prioriza experiencias basadas en el TD error para muestreo eficiente.
    """
    def __init__(self, capacity):
        super().__init__(capacity)
        self.priorities = np.zeros(capacity, dtype=np.float32) + 1e-5
        self.pos = 0
    
    def add(self, state, action, reward, next_state, done):
        """Añade una transición al buffer con prioridad máxima."""
        max_priority = max(np.max(self.priorities), 1e-5)
        
        if len(self.buffer) < self.buffer.maxlen:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.buffer.maxlen
    
    def sample(self, batch_size, beta=0.4):
        """
        Muestrea un lote basado en prioridades.
        
        Args:
            batch_size: Tamaño del lote
            beta: Factor para corrección de sesgo (0-1)
            
        Returns:
            Batch de transiciones, índices y pesos de importancia
        """
        if len(self.buffer) < batch_size:
            idx = range(len(self.buffer))
        else:
            # Calcular probabilidades de muestreo basadas en prioridad
            priorities = self.priorities[:len(self.buffer)]
            probabilities = priorities ** DQN_CONFIG['priority_alpha']
            probabilities /= np.sum(probabilities)
            
            # Muestreo según distribución
            rng = np.random.default_rng(seed=42)
            idx = rng.choice(len(self.buffer), batch_size, p=probabilities)
        
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
    
    def update_priorities(self, indices, priorities):
        """Actualiza las prioridades para los índices dados."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority