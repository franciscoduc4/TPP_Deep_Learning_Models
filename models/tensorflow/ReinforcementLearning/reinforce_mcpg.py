import os, sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LayerNormalization, Dropout, Activation
)
from tensorflow.keras.optimizers import Adam

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from models.config import REINFORCE_CONFIG


class PolicyNetwork(Model):
    """
    Red neuronal de política para REINFORCE.
    
    Esta red mapea estados a distribuciones de probabilidad sobre acciones.
    """
    def __init__(self, state_dim, action_dim, hidden_units=None, continuous=False):
        super(PolicyNetwork, self).__init__()
        
        # Configuración de la red
        if hidden_units is None:
            hidden_units = REINFORCE_CONFIG['hidden_units']
        
        self.continuous = continuous
        self.action_dim = action_dim
        
        # Capas ocultas
        self.hidden_layers = []
        for i, units in enumerate(hidden_units):
            self.hidden_layers.append(Dense(units, name=f'hidden_{i}'))
            self.hidden_layers.append(LayerNormalization(epsilon=REINFORCE_CONFIG['epsilon'], name=f'ln_{i}'))
            self.hidden_layers.append(Activation('relu', name=f'relu_{i}'))
            if REINFORCE_CONFIG['dropout_rate'] > 0:
                self.hidden_layers.append(Dropout(REINFORCE_CONFIG['dropout_rate'], name=f'dropout_{i}'))
        
        # Capa de salida: depende de si el espacio de acciones es continuo o discreto
        if continuous:
            # Para espacios continuos: política gaussiana
            self.mu = Dense(action_dim, activation='tanh', name='mu')
            self.log_sigma = Dense(action_dim, activation='linear', name='log_sigma')
        else:
            # Para espacios discretos: política categórica
            self.logits = Dense(action_dim, activation='linear', name='logits')
    
    def call(self, inputs, training=False):
        x = inputs
        
        # Pasar por capas ocultas
        for layer in self.hidden_layers:
            if isinstance(layer, Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        
        # Generar salida según el tipo de política
        if self.continuous:
            mu = self.mu(x)
            log_sigma = self.log_sigma(x)
            # Limitar el rango de log_sigma para estabilidad
            log_sigma = tf.clip_by_value(log_sigma, -20, 2)
            return mu, log_sigma
        else:
            logits = self.logits(x)
            return logits
    
    def get_action(self, state, deterministic=False):
        """
        Obtiene una acción según la política actual.
        
        Args:
            state: Estado actual
            deterministic: Si se usa la acción determinística (mean) o se muestrea

        Returns:
            La acción seleccionada
        """
        # Convertir a tensor y obtener distribución
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        
        if self.continuous:
            mu, log_sigma = self(state_tensor, training=False)
            
            if deterministic:
                return mu[0].numpy()
            
            # Muestrear de la distribución gaussiana para exploración
            sigma = tf.exp(log_sigma)
            normal_dist = tf.random.normal(shape=mu.shape)
            action = mu + sigma * normal_dist
            return action[0].numpy()
        else:
            logits = self(state_tensor, training=False)
            
            if deterministic:
                return tf.argmax(logits[0]).numpy()
            
            # Muestrear de la distribución categórica para exploración
            probs = tf.nn.softmax(logits)
            action = tf.random.categorical(tf.math.log(probs), 1)
            return action[0, 0].numpy()
    
    def get_log_prob(self, states, actions):
        """
        Calcula el logaritmo de la probabilidad de acciones dadas.
        
        Args:
            states: Estados observados
            actions: Acciones tomadas
            
        Returns:
            Log-probabilidades de las acciones
        """
        if self.continuous:
            mu, log_sigma = self(states)
            sigma = tf.exp(log_sigma)
            
            # Log-prob para distribución gaussiana
            log_probs = -0.5 * (
                tf.reduce_sum(
                    tf.square((actions - mu) / sigma) + 
                    2 * log_sigma + 
                    tf.math.log(2.0 * np.pi),
                    axis=1
                )
            )
            return log_probs
        else:
            logits = self(states)
            
            # Log-prob para distribución categórica
            action_masks = tf.one_hot(tf.cast(actions, tf.int32), self.action_dim)
            log_probs = tf.reduce_sum(
                action_masks * tf.nn.log_softmax(logits),
                axis=1
            )
            return log_probs
    
    def get_entropy(self, states):
        """
        Calcula la entropía de la política para los estados dados.
        
        Args:
            states: Estados para evaluar
            
        Returns:
            Entropía de la política
        """
        if self.continuous:
            _, log_sigma = self(states)
            # Entropía de distribución gaussiana: 0.5 * log(2*pi*e*sigma^2)
            entropy = tf.reduce_sum(
                0.5 * tf.math.log(2.0 * np.pi * np.e * tf.exp(2 * log_sigma)),
                axis=1
            )
        else:
            logits = self(states)
            probs = tf.nn.softmax(logits)
            # Entropía de distribución categórica: -sum(p * log(p))
            entropy = -tf.reduce_sum(
                probs * tf.math.log(probs + 1e-10),
                axis=1
            )
        return entropy


class REINFORCE:
    """
    Implementación del algoritmo REINFORCE (Monte Carlo Policy Gradient).
    
    REINFORCE utiliza retornos Monte Carlo completos para actualizar la política,
    haciéndolo un algoritmo simple pero efectivo para aprendizaje de políticas.
    """
    
    def __init__(
        self, 
        state_dim, 
        action_dim,
        continuous=False,
        learning_rate=REINFORCE_CONFIG['learning_rate'],
        gamma=REINFORCE_CONFIG['gamma'],
        hidden_units=None,
        baseline=REINFORCE_CONFIG['use_baseline'],
        entropy_coef=REINFORCE_CONFIG['entropy_coef']
    ):
        """
        Inicializa el agente REINFORCE.
        
        Args:
            state_dim: Dimensión del espacio de estados
            action_dim: Dimensión del espacio de acciones
            continuous: Si el espacio de acciones es continuo
            learning_rate: Tasa de aprendizaje
            gamma: Factor de descuento
            hidden_units: Lista con unidades en capas ocultas
            baseline: Si usar baseline para reducir varianza
            entropy_coef: Coeficiente para regularización por entropía
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous
        self.gamma = gamma
        self.use_baseline = baseline
        self.entropy_coef = entropy_coef
        
        # Valores predeterminados para capas ocultas
        if hidden_units is None:
            self.hidden_units = REINFORCE_CONFIG['hidden_units']
        else:
            self.hidden_units = hidden_units
        
        # Crear red de política
        self.policy = PolicyNetwork(state_dim, action_dim, self.hidden_units, continuous)
        self.optimizer = Adam(learning_rate=learning_rate)
        
        # Red de valor (baseline) opcional para reducir varianza
        if self.use_baseline:
            self.value_network = self._create_value_network()
            self.value_optimizer = Adam(learning_rate=learning_rate)
        
        # Métricas
        self.policy_loss_metric = tf.keras.metrics.Mean('policy_loss')
        self.entropy_metric = tf.keras.metrics.Mean('entropy')
        self.baseline_loss_metric = tf.keras.metrics.Mean('baseline_loss')
        self.returns_metric = tf.keras.metrics.Mean('returns')
    
    def _create_value_network(self):
        """
        Crea una red neuronal para estimar el valor de estado (baseline).
        
        Returns:
            Modelo de red de valor
        """
        inputs = Input(shape=(self.state_dim,))
        x = inputs
        
        # Capas ocultas
        for i, units in enumerate(self.hidden_units):
            x = Dense(units, activation='relu', name=f'value_hidden_{i}')(x)
            x = LayerNormalization(epsilon=REINFORCE_CONFIG['epsilon'], name=f'value_ln_{i}')(x)
            if REINFORCE_CONFIG['dropout_rate'] > 0:
                x = Dropout(REINFORCE_CONFIG['dropout_rate'], name=f'value_dropout_{i}')(x)
        
        # Capa de salida: un solo valor
        outputs = Dense(1, activation='linear', name='value')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    @tf.function
    def train_policy_step(self, states, actions, returns):
        """
        Realiza un paso de entrenamiento para la red de política.
        
        Args:
            states: Estados visitados
            actions: Acciones tomadas
            returns: Retornos calculados
            
        Returns:
            Pérdida de política, entropía media
        """
        with tf.GradientTape() as tape:
            # Calcular log-probabilidades de acciones tomadas
            log_probs = self.policy.get_log_prob(states, actions)
            
            # Si se usa baseline, restar el valor predicho de los retornos
            if self.use_baseline:
                values = tf.squeeze(self.value_network(states))
                advantages = returns - values
            else:
                advantages = returns
            
            # Calcular pérdida de política (negativa porque queremos maximizar)
            policy_loss = -tf.reduce_mean(log_probs * advantages, axis=0)
            
            # Calcular entropía y añadir regularización
            entropy = tf.reduce_mean(self.policy.get_entropy(states), axis=0)
            loss = policy_loss - self.entropy_coef * entropy
        
        # Calcular gradientes y actualizar política
        grads = tape.gradient(loss, self.policy.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))
        
        # Actualizar métricas
        self.policy_loss_metric.update_state(policy_loss)
        self.entropy_metric.update_state(entropy)
        self.returns_metric.update_state(tf.reduce_mean(returns, axis=0))
        
        return policy_loss, entropy
    
    @tf.function
    def train_baseline_step(self, states, returns):
        """
        Realiza un paso de entrenamiento para la red de valor (baseline).
        
        Args:
            states: Estados visitados
            returns: Retornos calculados
            
        Returns:
            Pérdida de la red de valor
        """
        with tf.GradientTape() as tape:
            values = tf.squeeze(self.value_network(states))
            value_loss = tf.reduce_mean(tf.square(returns - values), axis=0)
        
        # Calcular gradientes y actualizar red de valor
        grads = tape.gradient(value_loss, self.value_network.trainable_variables)
        self.value_optimizer.apply_gradients(zip(grads, self.value_network.trainable_variables))
        
        # Actualizar métrica
        self.baseline_loss_metric.update_state(value_loss)
        
        return value_loss
    
    def compute_returns(self, rewards):
        """
        Calcula los retornos descontados para cada paso de tiempo.
        
        Args:
            rewards: Lista de recompensas recibidas
            
        Returns:
            Array de retornos descontados
        """
        returns = np.zeros_like(rewards, dtype=np.float32)
        future_return = 0.0
        
        # Calcular retornos desde el final del episodio
        for t in reversed(range(len(rewards))):
            future_return = rewards[t] + self.gamma * future_return
            returns[t] = future_return
            
        # Normalizar retornos para estabilidad
        if len(returns) > 1:
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        
        return returns
    
    def _run_episode(self, env, render=False):
        """
        Ejecuta un episodio completo y recolecta la experiencia.
        
        Args:
            env: Entorno de OpenAI Gym o compatible
            render: Si renderizar el entorno durante entrenamiento
            
        Returns:
            Tupla con (estados, acciones, recompensas, recompensa_total, longitud_episodio)
        """
        state, _ = env.reset()
        done = False
        
        # Almacenar datos del episodio
        states = []
        actions = []
        rewards = []
        
        # Interactuar con el entorno hasta finalizar episodio
        while not done:
            if render:
                env.render()
            
            # Seleccionar acción según política actual
            action = self.policy.get_action(state)
            
            # Dar paso en el entorno
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Guardar transición
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            # Avanzar al siguiente estado
            state = next_state
        
        return states, actions, rewards, sum(rewards), len(rewards)
    
    def _update_networks(self, states, actions, rewards):
        """
        Actualiza las redes de política y valor.
        
        Args:
            states: Lista de estados
            actions: Lista de acciones
            rewards: Lista de recompensas
            
        Returns:
            Tupla con (pérdida_política, entropía)
        """
        # Calcular retornos
        returns = self.compute_returns(rewards)
        
        # Convertir a tensores
        states = np.array(states, dtype=np.float32)
        if self.continuous:
            actions = np.array(actions, dtype=np.float32)
        else:
            actions = np.array(actions, dtype=np.int32)
        returns = np.array(returns, dtype=np.float32)
        
        # Actualizar política
        policy_loss, entropy = self.train_policy_step(
            tf.convert_to_tensor(states), 
            tf.convert_to_tensor(actions), 
            tf.convert_to_tensor(returns)
        )
        
        # Actualizar baseline si se usa
        if self.use_baseline:
            self.train_baseline_step(
                tf.convert_to_tensor(states), 
                tf.convert_to_tensor(returns)
            )
        
        return policy_loss, entropy
    
    def _update_history(self, history, episode_reward, episode_length):
        """
        Actualiza la historia de entrenamiento con las métricas actuales.
        
        Args:
            history: Diccionario de historia
            episode_reward: Recompensa total del episodio
            episode_length: Longitud del episodio
            policy_loss: Pérdida de política
            entropy: Entropía
        """
        history['episode_rewards'].append(episode_reward)
        history['episode_lengths'].append(episode_length)
        history['policy_losses'].append(self.policy_loss_metric.result().numpy())
        if self.use_baseline:
            history['baseline_losses'].append(self.baseline_loss_metric.result().numpy())
        history['entropies'].append(self.entropy_metric.result().numpy())
        history['mean_returns'].append(self.returns_metric.result().numpy())
        
        # Resetear métricas
        self.policy_loss_metric.reset_states()
        self.entropy_metric.reset_states()
        self.returns_metric.reset_states()
        if self.use_baseline:
            self.baseline_loss_metric.reset_states()
    
    def train(self, env, episodes=None, render=False):
        """
        Entrena el agente REINFORCE en el entorno dado.
        
        Args:
            env: Entorno de OpenAI Gym o compatible
            episodes: Número de episodios de entrenamiento
            render: Si renderizar el entorno durante entrenamiento
            
        Returns:
            Historia de entrenamiento
        """
        if episodes is None:
            episodes = REINFORCE_CONFIG['episodes']
        
        history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_losses': [],
            'baseline_losses': [] if self.use_baseline else None,
            'entropies': [],
            'mean_returns': []
        }
        
        start_time = time.time()
        
        for episode in range(episodes):
            # Ejecutar episodio
            states, actions, rewards, episode_reward, episode_length = self._run_episode(env, render)
            
            # Actualizar redes
            policy_loss, entropy = self._update_networks(states, actions, rewards)
            
            # Actualizar historia
            self._update_history(history, episode_reward, episode_length)
            
            # Mostrar progreso periódicamente
            if (episode + 1) % REINFORCE_CONFIG['log_interval'] == 0:
                elapsed_time = time.time() - start_time
                avg_reward = np.mean(history['episode_rewards'][-REINFORCE_CONFIG['log_interval']:])
                print(f"Episodio {episode+1}/{episodes} - "
                      f"Recompensa Media: {avg_reward:.2f}, "
                      f"Pérdida Política: {policy_loss:.4f}, "
                      f"Entropía: {entropy:.4f}, "
                      f"Tiempo: {elapsed_time:.2f}s")
        
        return history
    
    def evaluate(self, env, episodes=10, render=False):
        """
        Evalúa el agente REINFORCE con su política actual.
        
        Args:
            env: Entorno de OpenAI Gym o compatible
            episodes: Número de episodios para evaluación
            render: Si renderizar el entorno durante evaluación
            
        Returns:
            Recompensa media obtenida
        """
        rewards = []
        lengths = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            while not done:
                if render:
                    env.render()
                
                # Usar política determinística para evaluación
                action = self.policy.get_action(state, deterministic=True)
                
                # Dar paso en el entorno
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Actualizar estado y contadores
                state = next_state
                episode_reward += reward
                steps += 1
            
            rewards.append(episode_reward)
            lengths.append(steps)
            
            print(f"Episodio Evaluación {episode+1}: Recompensa = {episode_reward:.2f}, Pasos = {steps}")
        
        avg_reward = np.mean(rewards)
        avg_length = np.mean(lengths)
        
        print(f"Evaluación Completada - Recompensa Media: {avg_reward:.2f}, Pasos Medios: {avg_length:.2f}")
        
        return avg_reward
    
    def save(self, policy_path, baseline_path=None):
        """
        Guarda los modelos del agente.
        
        Args:
            policy_path: Ruta para guardar la política
            baseline_path: Ruta para guardar el baseline (opcional)
        """
        # Guardar política
        self.policy.save_weights(policy_path)
        
        # Guardar baseline si existe
        if self.use_baseline and baseline_path:
            self.value_network.save_weights(baseline_path)
        
        print(f"Modelo guardado en {policy_path}")
    
    def load(self, policy_path, baseline_path=None):
        """
        Carga los modelos del agente.
        
        Args:
            policy_path: Ruta para cargar la política
            baseline_path: Ruta para cargar el baseline (opcional)
        """
        # Asegurarse que el modelo está construido antes de cargar
        dummy_state = np.zeros((1, self.state_dim))
        _ = self.policy(dummy_state)
        
        # Cargar política
        self.policy.load_weights(policy_path)
        
        # Cargar baseline si existe
        if self.use_baseline and baseline_path:
            _ = self.value_network(dummy_state)
            self.value_network.load_weights(baseline_path)
        
        print(f"Modelo cargado desde {policy_path}")
    
    def visualize_training(self, history=None, smoothing_window=None):
        """
        Visualiza las métricas de entrenamiento.
        
        Args:
            history: Historia de entrenamiento (opcional)
            smoothing_window: Tamaño de ventana para suavizado
        """
        if history is None:
            return
        
        if smoothing_window is None:
            smoothing_window = REINFORCE_CONFIG['smoothing_window']
        
        # Función para suavizar datos
        def smooth(data, window_size):
            if len(data) < window_size:
                return data
            kernel = np.ones(window_size) / window_size
            return np.convolve(data, kernel, mode='valid')
        
        # Determinar número de subplots
        n_plots = 4 if self.use_baseline else 3
        _, axs = plt.subplots(n_plots, 1, figsize=(10, 3*n_plots))
        
        # 1. Gráfico de recompensas
        axs[0].plot(history['episode_rewards'], alpha=0.3, color='blue', label='Raw')
        if len(history['episode_rewards']) > smoothing_window:
            smoothed_rewards = smooth(history['episode_rewards'], smoothing_window)
            axs[0].plot(range(smoothing_window-1, len(history['episode_rewards'])), 
                      smoothed_rewards, color='blue', 
                      label=f'Smoothed (window={smoothing_window})')
        axs[0].set_title('Recompensa por Episodio')
        axs[0].set_xlabel('Episodio')
        axs[0].set_ylabel('Recompensa')
        axs[0].grid(alpha=0.3)
        axs[0].legend()
        
        # 2. Gráfico de longitud de episodios
        axs[1].plot(history['episode_lengths'], alpha=0.3, color='green', label='Raw')
        if len(history['episode_lengths']) > smoothing_window:
            smoothed_lengths = smooth(history['episode_lengths'], smoothing_window)
            axs[1].plot(range(smoothing_window-1, len(history['episode_lengths'])), 
                      smoothed_lengths, color='green', 
                      label=f'Smoothed (window={smoothing_window})')
        axs[1].set_title('Longitud de Episodios')
        axs[1].set_xlabel('Episodio')
        axs[1].set_ylabel('Pasos')
        axs[1].grid(alpha=0.3)
        axs[1].legend()
        
        # 3. Gráfico de pérdida de política
        axs[2].plot(history['policy_losses'], alpha=0.3, color='red', label='Raw')
        if len(history['policy_losses']) > smoothing_window:
            smoothed_losses = smooth(history['policy_losses'], smoothing_window)
            axs[2].plot(range(smoothing_window-1, len(history['policy_losses'])), 
                      smoothed_losses, color='red', 
                      label=f'Smoothed (window={smoothing_window})')
        axs[2].set_title('Pérdida de Política')
        axs[2].set_xlabel('Episodio')
        axs[2].set_ylabel('Pérdida')
        axs[2].grid(alpha=0.3)
        axs[2].legend()
        
        # 4. Gráfico de pérdida de baseline (si se usa)
        if self.use_baseline:
            axs[3].plot(history['baseline_losses'], alpha=0.3, color='purple', label='Raw')
            if len(history['baseline_losses']) > smoothing_window:
                smoothed_baseline = smooth(history['baseline_losses'], smoothing_window)
                axs[3].plot(range(smoothing_window-1, len(history['baseline_losses'])), 
                          smoothed_baseline, color='purple', 
                          label=f'Smoothed (window={smoothing_window})')
            axs[3].set_title('Pérdida de Baseline')
            axs[3].set_xlabel('Episodio')
            axs[3].set_ylabel('Pérdida')
            axs[3].grid(alpha=0.3)
            axs[3].legend()
        
        plt.tight_layout()
        plt.show()