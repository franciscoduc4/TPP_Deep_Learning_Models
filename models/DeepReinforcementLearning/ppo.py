import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Conv1D, LSTM, Flatten, Concatenate,
    BatchNormalization, Dropout, LayerNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
from ..config import PPO_CONFIG


class ActorCriticModel(Model):
    """
    Modelo Actor-Crítico para PPO que divide la arquitectura en redes para
    política (actor) y valor (crítico).
    """
    def __init__(self, state_dim, action_dim, hidden_units=None):
        super(ActorCriticModel, self).__init__()
        
        # Valores predeterminados para capas ocultas
        if hidden_units is None:
            hidden_units = PPO_CONFIG['hidden_units']
        
        # Capas compartidas para procesamiento de estados
        self.shared_layers = []
        for i, units in enumerate(hidden_units[:2]):
            self.shared_layers.append(Dense(units, activation='tanh', name=f'shared_dense_{i}'))
            self.shared_layers.append(LayerNormalization(epsilon=PPO_CONFIG['epsilon'], name=f'shared_ln_{i}'))
            self.shared_layers.append(Dropout(PPO_CONFIG['dropout_rate'], name=f'shared_dropout_{i}'))
        
        # Red del Actor (política)
        self.actor_layers = []
        for i, units in enumerate(hidden_units[2:]):
            self.actor_layers.append(Dense(units, activation='tanh', name=f'actor_dense_{i}'))
            self.actor_layers.append(LayerNormalization(epsilon=PPO_CONFIG['epsilon'], name=f'actor_ln_{i}'))
        
        # Capa de salida del actor (mu y sigma para política gaussiana)
        self.mu = Dense(action_dim, activation='linear', name='actor_mu')
        self.log_sigma = Dense(action_dim, activation='linear', name='actor_log_sigma')
        
        # Red del Crítico (valor)
        self.critic_layers = []
        for i, units in enumerate(hidden_units[2:]):
            self.critic_layers.append(Dense(units, activation='tanh', name=f'critic_dense_{i}'))
            self.critic_layers.append(LayerNormalization(epsilon=PPO_CONFIG['epsilon'], name=f'critic_ln_{i}'))
        
        # Capa de salida del crítico (valor del estado)
        self.value = Dense(1, activation='linear', name='critic_value')
    
    def call(self, inputs):
        x = inputs
        
        # Capas compartidas
        for layer in self.shared_layers:
            x = layer(x)
        
        # Red del Actor
        actor_x = x
        for layer in self.actor_layers:
            actor_x = layer(actor_x)
        
        mu = self.mu(actor_x)
        log_sigma = self.log_sigma(actor_x)
        sigma = tf.exp(log_sigma)
        
        # Red del Crítico
        critic_x = x
        for layer in self.critic_layers:
            critic_x = layer(critic_x)
        
        value = self.value(critic_x)
        
        return mu, sigma, value
    
    def get_action(self, state, deterministic=False):
        """
        Obtiene una acción basada en el estado actual.
        
        Args:
            state: El estado actual
            deterministic: Si es True, devuelve la acción con máxima probabilidad
        
        Returns:
            Una acción muestreada de la distribución de política
        """
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        mu, sigma, _ = self.call(state)
        
        if deterministic:
            return mu[0]
        
        # Muestrear de la distribución normal
        dist = tf.random.normal(shape=mu.shape)
        action = mu + sigma * dist
        
        return action[0]
    
    def get_value(self, state):
        """
        Obtiene el valor estimado para un estado.
        
        Args:
            state: El estado para evaluar
        
        Returns:
            El valor estimado del estado
        """
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        _, _, value = self.call(state)
        return value[0]


class PPO:
    """
    Implementación de Proximal Policy Optimization (PPO).
    
    Esta implementación utiliza el clipping de PPO para actualizar la política
    y un estimador de ventaja generalizada (GAE) para mejorar el aprendizaje.
    """
    def __init__(
        self, 
        state_dim, 
        action_dim,
        learning_rate=PPO_CONFIG['learning_rate'],
        gamma=PPO_CONFIG['gamma'],
        epsilon=PPO_CONFIG['clip_epsilon'],
        hidden_units=None,
        entropy_coef=PPO_CONFIG['entropy_coef'],
        value_coef=PPO_CONFIG['value_coef'],
        max_grad_norm=PPO_CONFIG['max_grad_norm']
    ):
        # Parámetros del modelo
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        # Valores predeterminados para capas ocultas
        if hidden_units is None:
            self.hidden_units = PPO_CONFIG['hidden_units']
        else:
            self.hidden_units = hidden_units
        
        # Crear modelo y optimizador
        self.model = ActorCriticModel(state_dim, action_dim, self.hidden_units)
        self.optimizer = Adam(learning_rate=learning_rate)
        
        # Métricas
        self.policy_loss_metric = tf.keras.metrics.Mean('policy_loss')
        self.value_loss_metric = tf.keras.metrics.Mean('value_loss')
        self.entropy_metric = tf.keras.metrics.Mean('entropy')
        self.total_loss_metric = tf.keras.metrics.Mean('total_loss')
    
    def log_prob(self, mu, sigma, actions):
        """
        Calcula el logaritmo de la probabilidad de acciones bajo una política gaussiana.
        
        Args:
            mu: Media de la distribución gaussiana
            sigma: Desviación estándar de la distribución gaussiana
            actions: Acciones para calcular su probabilidad
        
        Returns:
            Logaritmo de probabilidad de las acciones
        """
        logp_normal = -0.5 * tf.square((actions - mu) / sigma) - 0.5 * tf.math.log(2.0 * np.pi) - tf.math.log(sigma)
        return tf.reduce_sum(logp_normal, axis=-1, keepdims=True)
    
    @tf.function
    def train_step(self, states, actions, old_log_probs, rewards, advantages, values):
        """
        Realiza un paso de entrenamiento para actualizar el modelo.
        
        Args:
            states: Estados observados en el entorno
            actions: Acciones tomadas para esos estados
            old_log_probs: Log de probabilidades de acciones bajo la política antigua
            rewards: Recompensas recibidas
            advantages: Ventajas estimadas
            values: Valores antiguos estimados por el crítico
        
        Returns:
            Pérdida total, pérdida de política, pérdida de valor, entropía
        """
        with tf.GradientTape() as tape:
            # Pasar estados por el modelo
            mu, sigma, new_values = self.model(states)
            
            # Calcular nueva probabilidad de acciones
            new_log_probs = self.log_prob(mu, sigma, actions)
            
            # Ratio entre nuevas y antiguas probabilidades
            ratio = tf.exp(new_log_probs - old_log_probs)
            
            # Términos del clipping de PPO
            p1 = ratio * advantages
            p2 = tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            
            # Pérdida de política (negativa porque queremos maximizar)
            policy_loss = -tf.reduce_mean(tf.minimum(p1, p2), axis=0)
            
            # Pérdida de valor (predicción vs retorno real)
            # Usamos la recompensa total como objetivo para el crítico
            value_loss = tf.reduce_mean(tf.square(rewards - new_values), axis=0)
            
            # Término de entropía para fomentar la exploración
            entropy = tf.reduce_mean(
                tf.reduce_sum(
                    0.5 * tf.math.log(2.0 * np.pi * tf.square(sigma)) + 0.5,
                    axis=-1
                ),
                axis=0
            )
            
            # Pérdida total combinada
            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
        # Calcular gradientes y actualizar pesos
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        
        # Clipping de gradientes para estabilidad
        if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
            
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        # Actualizar métricas
        self.policy_loss_metric.update_state(policy_loss)
        self.value_loss_metric.update_state(value_loss)
        self.entropy_metric.update_state(entropy)
        self.total_loss_metric.update_state(total_loss)
        
        return total_loss, policy_loss, value_loss, entropy
    
    def compute_gae(self, rewards, values, next_values, dones, gamma=0.99, lam=0.95):
        """
        Calcula el Estimador de Ventaja Generalizada (GAE).
        
        Args:
            rewards: Recompensas recibidas
            values: Valores estimados para los estados actuales
            next_values: Valores estimados para los estados siguientes
            dones: Indicadores de fin de episodio
            gamma: Factor de descuento
            lam: Factor lambda para GAE
        
        Returns:
            Ventajas y retornos calculados
        """
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
            
        returns = advantages + values
        
        # Normalizar ventajas
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)
        
        return advantages, returns
    
    def train(self, env, epochs=100, steps_per_epoch=4000, batch_size=64, update_iters=10):
        """
        Entrena el agente con PPO en un entorno dado.
        
        Args:
            env: Entorno de OpenAI Gym o compatible
            epochs: Número de épocas de entrenamiento
            steps_per_epoch: Pasos por época
            batch_size: Tamaño de lote para actualización
            update_iters: Número de iteraciones de actualización por lote
        
        Returns:
            Historial de entrenamiento
        """
        history = {
            'reward': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': []
        }
        
        for epoch in range(epochs):
            # Contenedores para almacenar experiencias
            states = []
            actions = []
            rewards = []
            values = []
            dones = []
            next_values = []
            log_probs = []
            
            # Recolectar experiencias
            state, _ = env.reset()
            episode_reward = 0
            for _ in range(steps_per_epoch):
                states.append(state)
                
                # Obtener acción y valor
                action = self.model.get_action(state)
                mu, sigma, value = self.model(tf.convert_to_tensor([state], dtype=tf.float32))
                
                # Log prob de la acción
                log_prob = self.log_prob(mu, sigma, tf.convert_to_tensor([action], dtype=tf.float32))[0]
                
                # Dar paso en el entorno
                next_state, reward, done, _, _ = env.step(action.numpy())
                
                # Guardar experiencia
                actions.append(action)
                rewards.append(reward)
                values.append(value[0][0])
                dones.append(done)
                log_probs.append(log_prob)
                
                # Actualizar estado y recompensa acumulada
                state = next_state
                episode_reward += reward
                
                # Si el episodio termina, resetear
                if done:
                    state, _ = env.reset()
                    history['reward'].append(episode_reward)
                    episode_reward = 0
                
                # Obtener valor del siguiente estado
                next_value = self.model.get_value(next_state)
                next_values.append(next_value.numpy()[0])
            
            # Calcular ventajas y retornos usando GAE
            advantages, returns = self.compute_gae(
                np.array(rewards), np.array(values), 
                np.array(next_values), np.array(dones)
            )
            
            # Convertir listas a arrays
            states = np.array(states, dtype=np.float32)
            actions = np.array(actions, dtype=np.float32)
            old_log_probs = np.array(log_probs, dtype=np.float32)
            
            # Crear conjunto de datos para entrenar
            dataset = tf.data.Dataset.from_tensor_slices(
                (states, actions, old_log_probs, returns, advantages, values)
            )
            dataset = dataset.shuffle(buffer_size=steps_per_epoch).batch(batch_size)
            
            # Entrenar durante varias iteraciones
            for _ in range(update_iters):
                for batch in dataset:
                    self.train_step(*batch)
            
            # Registrar métricas
            history['policy_loss'].append(self.policy_loss_metric.result().numpy())
            history['value_loss'].append(self.value_loss_metric.result().numpy())
            history['entropy'].append(self.entropy_metric.result().numpy())
            history['total_loss'].append(self.total_loss_metric.result().numpy())
            
            # Resetear métricas
            self.policy_loss_metric.reset_states()
            self.value_loss_metric.reset_states()
            self.entropy_metric.reset_states()
            self.total_loss_metric.reset_states()
            
            # Mostrar progreso
            if (epoch + 1) % 10 == 0:
                avg_reward = np.mean(history['reward'][-10:])
                print(f"Epoch {epoch+1}/{epochs} - Avg Reward: {avg_reward:.2f}, "
                      f"Policy Loss: {history['policy_loss'][-1]:.4f}, "
                      f"Value Loss: {history['value_loss'][-1]:.4f}")
        
        return history
    
    def save_model(self, filepath):
        """Guarda el modelo en un archivo."""
        self.model.save_weights(filepath)
        
    def load_model(self, filepath):
        """Carga el modelo desde un archivo."""
        # Asegurarse de que el modelo está construido primero
        dummy_state = np.zeros((1, self.state_dim))
        self.model(dummy_state)
        self.model.load_weights(filepath)