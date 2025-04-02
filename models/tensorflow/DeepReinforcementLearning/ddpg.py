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

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from models.config import DDPG_CONFIG


class ReplayBuffer:
    """
    Buffer de experiencias para el algoritmo DDPG.
    
    Almacena transiciones (state, action, reward, next_state, done)
    y permite muestrear lotes de manera aleatoria para el entrenamiento.
    """
    def __init__(self, capacity=100000):
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
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        """Retorna la cantidad de transiciones almacenadas."""
        return len(self.buffer)


class OUActionNoise:
    """
    Implementa el proceso de ruido de Ornstein-Uhlenbeck para exploración.
    
    Este ruido añade correlación temporal a las acciones para una exploración más efectiva
    en espacios continuos.
    """
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None, seed=42):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.reset()
        
    def __call__(self):
        # Fórmula para el proceso de Ornstein-Uhlenbeck
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * self.rng.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x
    
    def reset(self):
        """Reinicia el estado del ruido."""
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class ActorNetwork(Model):
    """
    Red de Actor para DDPG que mapea estados a acciones determinísticas.
    """
    def __init__(self, state_dim, action_dim, action_high, action_low, hidden_units=None):
        super(ActorNetwork, self).__init__()
        
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
                activation=DDPG_CONFIG['actor_activation'],
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
    
    def call(self, inputs, training=False):
        x = inputs
        
        # Procesar a través de capas ocultas
        for layer in self.layers_list:
            # Para capas de Dropout, pasamos el parámetro training
            if isinstance(layer, Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        
        # Capa de salida con activación tanh y escalado
        raw_actions = self.output_layer(x)
        
        # Escalar desde [-1, 1] al rango de acción [low, high]
        scaled_actions = 0.5 * (raw_actions + 1.0) * self.action_range + self.action_low
        
        return scaled_actions


class CriticNetwork(Model):
    """
    Red de Crítico para DDPG que mapea pares (estado, acción) a valores-Q.
    """
    def __init__(self, state_dim, action_dim, hidden_units=None):
        super(CriticNetwork, self).__init__()
        
        # Valores predeterminados para capas ocultas
        if hidden_units is None:
            hidden_units = DDPG_CONFIG['critic_hidden_units']
        
        # Capas iniciales para procesar el estado
        self.state_layers = []
        for i, units in enumerate(hidden_units[:1]):  # Primera capa solo para estado
            self.state_layers.append(Dense(
                units, 
                activation=DDPG_CONFIG['critic_activation'],
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
                activation=DDPG_CONFIG['critic_activation'],
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
    
    def call(self, inputs, training=False):
        states, actions = inputs
        
        # Procesar el estado
        x = states
        for layer in self.state_layers:
            x = layer(x)
        
        # Combinar estado procesado con acción
        x = Concatenate()([x, actions])
        
        # Procesar a través de capas combinadas
        for layer in self.combined_layers:
            if isinstance(layer, Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        
        # Capa de salida
        q_value = self.output_layer(x)
        
        return q_value


class DDPG:
    """
    Implementación del algoritmo Deep Deterministic Policy Gradient (DDPG).
    
    DDPG combina ideas de DQN y métodos de policy gradient para manejar
    espacios de acción continuos con políticas determinísticas.
    """
    def __init__(
        self, 
        state_dim, 
        action_dim,
        action_high,
        action_low,
        actor_lr=DDPG_CONFIG['actor_lr'],
        critic_lr=DDPG_CONFIG['critic_lr'],
        gamma=DDPG_CONFIG['gamma'],
        tau=DDPG_CONFIG['tau'],
        buffer_capacity=DDPG_CONFIG['buffer_capacity'],
        batch_size=DDPG_CONFIG['batch_size'],
        noise_std=DDPG_CONFIG['noise_std'],
        actor_hidden_units=None,
        critic_hidden_units=None
    ):
        # Parámetros del entorno y del modelo
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_high = action_high
        self.action_low = action_low
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        # Valores predeterminados para capas ocultas
        if actor_hidden_units is None:
            self.actor_hidden_units = DDPG_CONFIG['actor_hidden_units']
        else:
            self.actor_hidden_units = actor_hidden_units
            
        if critic_hidden_units is None:
            self.critic_hidden_units = DDPG_CONFIG['critic_hidden_units']
        else:
            self.critic_hidden_units = critic_hidden_units
        
        # Crear redes de Actor y Crítico (principal y target)
        self.actor = ActorNetwork(state_dim, action_dim, action_high, action_low, self.actor_hidden_units)
        self.target_actor = ActorNetwork(state_dim, action_dim, action_high, action_low, self.actor_hidden_units)
        
        self.critic = CriticNetwork(state_dim, action_dim, self.critic_hidden_units)
        self.target_critic = CriticNetwork(state_dim, action_dim, self.critic_hidden_units)
        
        # Copiar pesos iniciales
        self.update_target_networks(tau=1.0)
        
        # Optimizadores
        self.actor_optimizer = Adam(learning_rate=actor_lr)
        self.critic_optimizer = Adam(learning_rate=critic_lr)
        
        # Buffer de experiencias
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Ruido para exploración
        self.noise = OUActionNoise(
            mean=np.zeros(action_dim),
            std_deviation=noise_std * np.ones(action_dim)
        )
        
        # Métricas
        self.actor_loss_metric = tf.keras.metrics.Mean('actor_loss')
        self.critic_loss_metric = tf.keras.metrics.Mean('critic_loss')
        self.q_value_metric = tf.keras.metrics.Mean('q_value')
    
    def update_target_networks(self, tau=None):
        """
        Actualiza los pesos de las redes target usando soft update.
        
        Args:
            tau: Factor de actualización suave (si None, usa el valor por defecto)
        """
        if tau is None:
            tau = self.tau
            
        # Actualizar target actor
        for source, target in zip(self.actor.trainable_variables, self.target_actor.trainable_variables):
            target.assign(tau * source + (1.0 - tau) * target)
        
        # Actualizar target critic
        for source, target in zip(self.critic.trainable_variables, self.target_critic.trainable_variables):
            target.assign(tau * source + (1.0 - tau) * target)
    
    def get_action(self, state, add_noise=True):
        """
        Obtiene una acción determinística para un estado, opcionalmente añadiendo ruido.
        
        Args:
            state: Estado actual
            add_noise: Si se debe añadir ruido para exploración
            
        Returns:
            Acción seleccionada
        """
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        action = self.actor(state)[0].numpy()
        
        if add_noise:
            noise = self.noise()
            action += noise
            
        # Clipear al rango válido de acciones
        action = np.clip(action, self.action_low, self.action_high)
        
        return action
    
    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        """
        Realiza un paso de entrenamiento DDPG con un lote de experiencias.
        
        Args:
            states: Estados actuales
            actions: Acciones tomadas
            rewards: Recompensas recibidas
            next_states: Estados siguientes
            dones: Indicadores de fin de episodio
            
        Returns:
            Pérdidas de actor y crítico
        """
        with tf.GradientTape() as tape:
            # Predecir acciones target para los siguientes estados
            target_actions = self.target_actor(next_states, training=False)
            
            # Predecir Q-values target
            target_q_values = self.target_critic([next_states, target_actions], training=False)
            
            # Calcular Q-values objetivo usando la ecuación de Bellman
            target_q = rewards + (1 - dones) * self.gamma * target_q_values
            
            # Predecir Q-values actuales
            current_q = self.critic([states, actions], training=True)
            
            # Calcular pérdida del crítico (error cuadrático medio)
            critic_loss = tf.reduce_mean(tf.square(target_q - current_q), axis=0)
        
        # Calcular gradientes y actualizar pesos del crítico
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
        
        # Actualizar el actor usando el gradiente del crítico respecto a las acciones
        with tf.GradientTape() as tape:
            # Predecir acciones para los estados actuales
            actor_actions = self.actor(states, training=True)
            
            # Calcular Q-values para estas acciones
            actor_q_values = self.critic([states, actor_actions], training=False)
            
            # Pérdida del actor (negativo del Q-value promedio)
            # Queremos maximizar Q-value, así que minimizamos su negativo
            actor_loss = -tf.reduce_mean(actor_q_values, axis=0)
        
        # Calcular gradientes y actualizar pesos del actor
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        
        # Actualizar métricas
        self.actor_loss_metric.update_state(actor_loss)
        self.critic_loss_metric.update_state(critic_loss)
        self.q_value_metric.update_state(tf.reduce_mean(current_q, axis=0))
        
        return actor_loss, critic_loss
    
    def _select_action(self, state, step_counter, warmup_steps):
        """Selecciona una acción basada en el estado actual y fase de entrenamiento."""
        if step_counter < warmup_steps:
            # Using the same seed as the OUActionNoise for consistency
            rng = np.random.default_rng(seed=42)
            return rng.uniform(self.action_low, self.action_high, self.action_dim)
        else:
            return self.get_action(state, add_noise=True)
    
    def _update_model(self, step_counter, update_every):
        """Actualiza los modelos si es momento de hacerlo."""
        if step_counter % update_every == 0:
            # Muestrear batch
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            
            # Realizar actualización
            actor_loss, critic_loss = self.train_step(
                tf.convert_to_tensor(states, dtype=tf.float32),
                tf.convert_to_tensor(actions, dtype=tf.float32),
                tf.convert_to_tensor(rewards, dtype=tf.float32),
                tf.convert_to_tensor(next_states, dtype=tf.float32),
                tf.convert_to_tensor(dones, dtype=tf.float32)
            )
            
            # Actualizar redes target
            self.update_target_networks()
            
            return actor_loss.numpy(), critic_loss.numpy()
        return None, None
    
    def _update_history(self, history, episode_reward, episode_actor_loss, episode_critic_loss):
        """Actualiza el historial de entrenamiento con los resultados del episodio."""
        history['episode_rewards'].append(episode_reward)
        if episode_actor_loss:
            history['actor_losses'].append(np.mean(episode_actor_loss))
            history['critic_losses'].append(np.mean(episode_critic_loss))
            history['avg_q_values'].append(self.q_value_metric.result().numpy())
        else:
            history['actor_losses'].append(float('nan'))
            history['critic_losses'].append(float('nan'))
            history['avg_q_values'].append(float('nan'))
        return history
    
    def _log_progress(self, episode, episodes, episode_reward_history, history, log_interval, best_reward):
        """Registra y muestra el progreso del entrenamiento periódicamente."""
        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(episode_reward_history)
            print(f"Episode {episode+1}/{episodes} - Average Reward: {avg_reward:.2f}, "
                  f"Actor Loss: {history['actor_losses'][-1]:.4f}, "
                  f"Critic Loss: {history['critic_losses'][-1]:.4f}")
            
            # Verificar si es el mejor modelo
            if avg_reward > best_reward:
                best_reward = avg_reward
                print(f"Nuevo mejor modelo con recompensa: {best_reward:.2f}")
        
        return best_reward
    
    def _run_episode(self, env, max_steps, step_counter, warmup_steps, update_every, render):
        """Ejecuta un episodio completo de entrenamiento."""
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
                    episode_actor_loss.append(actor_loss)
                    episode_critic_loss.append(critic_loss)
            
            if done:
                break
                
        return episode_reward, episode_actor_loss, episode_critic_loss, step_counter
    
    def train(self, env, episodes=1000, max_steps=1000, warmup_steps=1000, 
             update_every=1, render=False, log_interval=10):
        """
        Entrena el agente DDPG en un entorno dado.
        
        Args:
            env: Entorno de Gym o compatible
            episodes: Número máximo de episodios
            max_steps: Pasos máximos por episodio
            warmup_steps: Pasos iniciales para recolectar experiencias antes de entrenar
            update_every: Frecuencia de actualización
            render: Mostrar entorno gráficamente
            log_interval: Intervalo para mostrar información
            
        Returns:
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
            
            # Resetear métricas y ruido
            self.actor_loss_metric.reset_states()
            self.critic_loss_metric.reset_states()
            self.q_value_metric.reset_states()
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
    
    def evaluate(self, env, episodes=10, render=False):
        """
        Evalúa el agente DDPG en un entorno dado sin exploración.
        
        Args:
            env: Entorno de Gym o compatible
            episodes: Número de episodios para evaluar
            render: Si se debe renderizar el entorno
            
        Returns:
            Recompensas promedio obtenidas durante la evaluación
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
            print(f"Evaluation Episode {episode+1}/{episodes} - Reward: {episode_reward:.2f}")
        
        avg_reward = np.mean(rewards)
        print(f"Average Evaluation Reward: {avg_reward:.2f}")
        
        return avg_reward
    
    def save_models(self, actor_path, critic_path):
        """
        Guarda los modelos del actor y crítico en archivos separados.
        
        Args:
            actor_path: Ruta para guardar el modelo del actor
            critic_path: Ruta para guardar el modelo del crítico
        """
        # Asegurar que los modelos estén construidos
        dummy_state = np.zeros((1, self.state_dim))
        dummy_action = np.zeros((1, self.action_dim))
        
        # Construir los modelos antes de guardarlos
        self.actor(dummy_state)
        self.critic([dummy_state, dummy_action])
        
        # Guardar pesos
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)
        print(f"Modelos guardados en {actor_path} y {critic_path}")
    
    def load_models(self, actor_path, critic_path):
        """
        Carga los modelos del actor y crítico desde archivos.
        
        Args:
            actor_path: Ruta para cargar el modelo del actor
            critic_path: Ruta para cargar el modelo del crítico
        """
        # Asegurar que los modelos estén construidos
        dummy_state = np.zeros((1, self.state_dim))
        dummy_action = np.zeros((1, self.action_dim))
        
        # Construir los modelos antes de cargarlos
        self.actor(dummy_state)
        self.critic([dummy_state, dummy_action])
        
        # Cargar pesos
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)
        
        # Actualizar redes target
        self.update_target_networks(tau=1.0)
        print(f"Modelos cargados desde {actor_path} y {critic_path}")
    
    def visualize_training(self, history, window_size=10):
        """
        Visualiza los resultados del entrenamiento.
        
        Args:
            history: Historia de entrenamiento
            window_size: Tamaño de ventana para suavizado
        """
        import matplotlib.pyplot as plt
        
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