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
import os
from ..config import SAC_CONFIG

class ReplayBuffer:
    """
    Buffer de experiencias para el algoritmo SAC.
    
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
            np.array(rewards, dtype=np.float32).reshape(-1, 1),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32).reshape(-1, 1)
        )
    
    def __len__(self):
        """Retorna la cantidad de transiciones almacenadas."""
        return len(self.buffer)


class ActorNetwork(Model):
    """
    Red del Actor para SAC que produce una distribución de política gaussiana.
    
    Esta red mapea estados a distribuciones de probabilidad sobre acciones
    mediante una política estocástica parametrizada por una distribución normal.
    """
    def __init__(self, state_dim, action_dim, action_high, action_low, hidden_units=None):
        super(ActorNetwork, self).__init__()
        
        # Límites de acciones para escalar la salida
        self.action_high = action_high
        self.action_low = action_low
        self.action_dim = action_dim
        self.log_std_min = SAC_CONFIG['log_std_min']  # Límite inferior para log_std
        self.log_std_max = SAC_CONFIG['log_std_max']  # Límite superior para log_std
        
        # Valores predeterminados para capas ocultas
        if hidden_units is None:
            hidden_units = SAC_CONFIG['actor_hidden_units']
        
        # Capas para el procesamiento del estado
        self.hidden_layers = []
        for i, units in enumerate(hidden_units):
            self.hidden_layers.append(Dense(
                units, 
                activation=SAC_CONFIG['actor_activation'],
                name=f'actor_dense_{i}'
            ))
            self.hidden_layers.append(LayerNormalization(
                epsilon=SAC_CONFIG['epsilon'],
                name=f'actor_ln_{i}'
            ))
            self.hidden_layers.append(Dropout(
                SAC_CONFIG['dropout_rate'],
                name=f'actor_dropout_{i}'
            ))
        
        # Capas de salida para media y log-desviación estándar
        self.mean_layer = Dense(action_dim, activation='linear', name='actor_mean')
        self.log_std_layer = Dense(action_dim, activation='linear', name='actor_log_std')
    
    def call(self, inputs, training=False):
        x = inputs
        
        # Procesar a través de capas ocultas
        for layer in self.hidden_layers:
            if isinstance(layer, Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        
        # Calcular media y log_std
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = tf.clip_by_value(log_std, self.log_std_min, self.log_std_max)
        std = tf.exp(log_std)
        
        return mean, std
    
    def sample_action(self, state, deterministic=False):
        """
        Muestrea una acción de la distribución de política.
        
        Args:
            state: El estado actual
            deterministic: Si es True, devuelve la acción media (sin ruido)
            
        Returns:
            Acción muestreada, log-probabilidad de la acción
        """
        # Obtener parámetros de la distribución
        mean, std = self(state, training=False)
        
        if deterministic:
            # Para evaluación o explotación
            actions = mean
            log_probs = None
        else:
            # Muestrear usando el truco de reparametrización para permitir backprop
            noise = tf.random.normal(shape=mean.shape)
            z = mean + std * noise
            
            # Aplicar tanh para acotar las acciones
            actions = tf.tanh(z)
            
            # Escalar acciones al rango deseado
            actions = actions * (self.action_high - self.action_low) / 2 + (self.action_high + self.action_low) / 2
            
            # Calcular log-probabilidad con corrección para tanh
            # log p(a|s) = log p(z|s) - sum(log(1 - tanh(z)^2))
            # Donde z son las acciones antes de aplicar tanh
            log_probs = self._log_prob(z, std, actions)
            
        return actions, log_probs
    
    def _log_prob(self, z, std, actions):
        """
        Calcula el logaritmo de la probabilidad de una acción.
        
        Args:
            z: Valor antes de aplicar tanh
            std: Desviación estándar
            actions: Acción muestreada
            
        Returns:
            Log-probabilidad de la acción
        """
        # Log-prob de distribución normal
        log_prob_gaussian = -0.5 * (tf.square(z) + 2 * tf.math.log(std) + np.log(2.0 * np.pi))
        log_prob_gaussian = tf.reduce_sum(log_prob_gaussian, axis=1, keepdims=True)
        
        # Corrección por transformación tanh
        # Deriva de cambio de variable (ver paper SAC)
        squash_correction = tf.reduce_sum(
            tf.math.log(1.0 - tf.square(tf.tanh(z)) + 1e-6),
            axis=1,
            keepdims=True
        )
        
        return log_prob_gaussian - squash_correction


class CriticNetwork(Model):
    """
    Red de Crítico para SAC que mapea pares (estado, acción) a valores-Q.
    """
    def __init__(self, state_dim, action_dim, hidden_units=None, name='critic'):
        super(CriticNetwork, self).__init__(name=name)
        
        # Valores predeterminados para capas ocultas
        if hidden_units is None:
            hidden_units = SAC_CONFIG['critic_hidden_units']
        
        # Capas iniciales para procesar el estado
        self.state_layers = []
        for i, units in enumerate(hidden_units[:1]):  # Primera capa solo para estado
            self.state_layers.append(Dense(
                units, 
                activation=SAC_CONFIG['critic_activation'],
                name=f'{name}_state_dense_{i}'
            ))
            self.state_layers.append(LayerNormalization(
                epsilon=SAC_CONFIG['epsilon'],
                name=f'{name}_state_ln_{i}'
            ))
        
        # Capas para procesar la combinación de estado y acción
        self.combined_layers = []
        for i, units in enumerate(hidden_units[1:]):
            self.combined_layers.append(Dense(
                units, 
                activation=SAC_CONFIG['critic_activation'],
                name=f'{name}_combined_dense_{i}'
            ))
            self.combined_layers.append(LayerNormalization(
                epsilon=SAC_CONFIG['epsilon'],
                name=f'{name}_combined_ln_{i}'
            ))
            self.combined_layers.append(Dropout(
                SAC_CONFIG['dropout_rate'],
                name=f'{name}_dropout_{i}'
            ))
        
        # Capa de salida: valor Q (sin activación)
        self.output_layer = Dense(1, name=f'{name}_output')
    
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


class SAC:
    """
    Implementación del algoritmo Soft Actor-Critic (SAC).
    
    SAC es un algoritmo de aprendizaje por refuerzo fuera de política (off-policy)
    basado en el marco de máxima entropía, que busca maximizar tanto el retorno
    esperado como la entropía de la política.
    """
    def __init__(
        self, 
        state_dim, 
        action_dim,
        action_high,
        action_low,
        actor_lr=SAC_CONFIG['actor_lr'],
        critic_lr=SAC_CONFIG['critic_lr'],
        alpha_lr=SAC_CONFIG['alpha_lr'],
        gamma=SAC_CONFIG['gamma'],
        tau=SAC_CONFIG['tau'],
        buffer_capacity=SAC_CONFIG['buffer_capacity'],
        batch_size=SAC_CONFIG['batch_size'],
        initial_alpha=SAC_CONFIG['initial_alpha'],
        target_entropy=None,
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
            self.actor_hidden_units = SAC_CONFIG['actor_hidden_units']
        else:
            self.actor_hidden_units = actor_hidden_units
            
        if critic_hidden_units is None:
            self.critic_hidden_units = SAC_CONFIG['critic_hidden_units']
        else:
            self.critic_hidden_units = critic_hidden_units
        
        # Parámetro de temperatura alpha (para regularización de entropía)
        # Logaritmo para garantizar positividad al aplicar exp()
        self.log_alpha = tf.Variable(tf.math.log(initial_alpha), trainable=True)
        
        # Entropía objetivo (heurística: -dim(A))
        if target_entropy is None:
            self.target_entropy = -action_dim  # Valor predeterminado: -dim(A)
        else:
            self.target_entropy = target_entropy
        
        # Crear redes de Actor y Críticos (principal y target)
        self.actor = ActorNetwork(state_dim, action_dim, action_high, action_low, self.actor_hidden_units)
        
        # Dos redes críticas para mitigar el sesgo de optimista
        self.critic_1 = CriticNetwork(state_dim, action_dim, self.critic_hidden_units, name='critic_1')
        self.critic_2 = CriticNetwork(state_dim, action_dim, self.critic_hidden_units, name='critic_2')
        self.target_critic_1 = CriticNetwork(state_dim, action_dim, self.critic_hidden_units, name='target_critic_1')
        self.target_critic_2 = CriticNetwork(state_dim, action_dim, self.critic_hidden_units, name='target_critic_2')
        
        # Copiar pesos iniciales a las redes target
        self.update_target_networks(tau=1.0)
        
        # Optimizadores
        self.actor_optimizer = Adam(learning_rate=actor_lr)
        self.critic_optimizer = Adam(learning_rate=critic_lr)
        self.alpha_optimizer = Adam(learning_rate=alpha_lr)
        
        # Buffer de experiencias
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Métricas
        self.actor_loss_metric = tf.keras.metrics.Mean('actor_loss')
        self.critic_loss_metric = tf.keras.metrics.Mean('critic_loss')
        self.alpha_loss_metric = tf.keras.metrics.Mean('alpha_loss')
        self.entropy_metric = tf.keras.metrics.Mean('entropy')
        self.q_value_metric = tf.keras.metrics.Mean('q_value')
    
    def update_target_networks(self, tau=None):
        """
        Actualiza los pesos de las redes target usando soft update.
        
        Args:
            tau: Factor de interpolación (opcional, usa el del objeto si es None)
        """
        tau = tau if tau is not None else self.tau
        
        # Actualizar Target Critic 1
        for target_var, source_var in zip(self.target_critic_1.variables, self.critic_1.variables):
            target_var.assign((1 - tau) * target_var + tau * source_var)
        
        # Actualizar Target Critic 2
        for target_var, source_var in zip(self.target_critic_2.variables, self.critic_2.variables):
            target_var.assign((1 - tau) * target_var + tau * source_var)
    
    def get_action(self, state, deterministic=False):
        """
        Obtiene una acción basada en el estado actual.
        
        Args:
            state: Estado actual
            deterministic: Si es True, devuelve acción determinística (para evaluación)
            
        Returns:
            Acción seleccionada
        """
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        action, _ = self.actor.sample_action(state, deterministic)
        
        # Convertir a numpy y extraer primera acción
        return action.numpy()[0]
    
    @tf.function
    def update_alpha(self, log_probs):
        """
        Actualiza el parámetro de temperatura alpha.
        
        Args:
            log_probs: Log-probabilidades de las acciones muestreadas
            
        Returns:
            Pérdida de alpha
        """
        with tf.GradientTape() as tape:
            # Objetivo: ajustar alpha para alcanzar la entropía objetivo
            alpha = tf.exp(self.log_alpha)
            alpha_loss = -tf.reduce_mean(
                alpha * (log_probs + self.target_entropy),
                axis=0
            )
            
        # Actualizar alpha
        alpha_gradients = tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_optimizer.apply_gradients(zip(alpha_gradients, [self.log_alpha]))
        
        # Actualizar métrica
        self.alpha_loss_metric.update_state(alpha_loss)
        
        return alpha_loss
    
    @tf.function
    def update_critics(self, states, actions, rewards, next_states, dones):
        """
        Actualiza las redes de crítica.
        
        Args:
            states: Estados actuales
            actions: Acciones tomadas
            rewards: Recompensas recibidas
            next_states: Estados siguientes
            dones: Indicadores de fin de episodio
            
        Returns:
            Pérdida de los críticos
        """
        # Obtener alpha actual
        alpha = tf.exp(self.log_alpha)
        
        # Muestrear acciones para el siguiente estado
        next_actions, next_log_probs = self.actor.sample_action(next_states)
        
        # Calcular valores Q para el siguiente estado
        q1_next = self.target_critic_1([next_states, next_actions])
        q2_next = self.target_critic_2([next_states, next_actions])
        
        # Tomar el mínimo para evitar sobreestimación
        q_next = tf.minimum(q1_next, q2_next)
        
        # Añadir término de entropía al Q-target
        soft_q_next = q_next - alpha * next_log_probs
        
        # Calcular target usando ecuación de Bellman
        # No considerar siguiente estado si el episodio terminó
        q_target = rewards + (1 - dones) * self.gamma * soft_q_next
        
        # Detener gradientes para el target
        q_target = tf.stop_gradient(q_target)
        
        # Calcular pérdida para ambos críticos
        with tf.GradientTape(persistent=True) as tape:
            q1_pred = self.critic_1([states, actions])
            q2_pred = self.critic_2([states, actions])
            
            critic_1_loss = tf.reduce_mean(tf.square(q_target - q1_pred), axis=0)
            critic_2_loss = tf.reduce_mean(tf.square(q_target - q2_pred), axis=0)
            critic_loss = critic_1_loss + critic_2_loss
            
        # Actualizar críticos
        critic_1_gradients = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
        critic_2_gradients = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)
        
        self.critic_optimizer.apply_gradients(
            zip(critic_1_gradients, self.critic_1.trainable_variables))
        self.critic_optimizer.apply_gradients(
            zip(critic_2_gradients, self.critic_2.trainable_variables))
        
        # Actualizar métricas
        self.critic_loss_metric.update_state(critic_loss)
        self.q_value_metric.update_state(tf.reduce_mean(q1_pred, axis=0))
        
        del tape
        
        return critic_loss
    
    @tf.function
    def update_actor(self, states):
        """
        Actualiza la red del actor.
        
        Args:
            states: Estados actuales
            
        Returns:
            Pérdida del actor, entropía
        """
        # Obtener alpha actual
        alpha = tf.exp(self.log_alpha)
        
        with tf.GradientTape() as tape:
            # Muestrear acciones y sus log-probs
            actions, log_probs = self.actor.sample_action(states)
            
            # Calcular valores Q de ambos críticos
            q1 = self.critic_1([states, actions])
            q2 = self.critic_2([states, actions])
            
            # Tomar el mínimo para evitar sobreestimación
            q = tf.minimum(q1, q2)
            
            # Pérdida del actor: minimizar KL divergence entre política y Q suavizada
            # Equivalente a maximizar entropía + Q-value esperado
            actor_loss = tf.reduce_mean(alpha * log_probs - q, axis=0)
            
        # Calcular gradientes y actualizar
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        
        # Actualizar métricas
        self.actor_loss_metric.update_state(actor_loss)
        
        # Calcular entropía media para monitoreo
        entropy = -tf.reduce_mean(log_probs, axis=0)
        self.entropy_metric.update_state(entropy)
        
        return actor_loss, log_probs, entropy
    
    def train_step(self):
        """
        Realiza un paso de entrenamiento completo (actor, crítico y alpha).
        
        Returns:
            Tuple con las pérdidas (actor, crítico, alpha) y entropía
        """
        # Si no hay suficientes datos, no hacer nada
        if len(self.replay_buffer) < self.batch_size:
            return None, None, None, None
        
        # Muestrear batch del buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convertir a tensores
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        # Actualizar críticos
        critic_loss = self.update_critics(states, actions, rewards, next_states, dones)
        
        # Actualizar actor
        actor_loss, log_probs, entropy = self.update_actor(states)
        
        # Actualizar alpha
        alpha_loss = self.update_alpha(log_probs)
        
        # Actualizar redes target
        self.update_target_networks()
        
        return actor_loss, critic_loss, alpha_loss, entropy
    
    def train(self, env, episodes=1000, max_steps=1000, 
              warmup_steps=10000, update_after=1000, update_every=50,
              evaluate_interval=10, render=False):
        """
        Entrena el agente SAC en un entorno dado.
        
        Args:
            env: Entorno de Gym o compatible
            episodes: Número máximo de episodios
            max_steps: Pasos máximos por episodio
            warmup_steps: Pasos iniciales con acciones aleatorias para explorar
            update_after: Pasos antes de empezar a entrenar
            update_every: Frecuencia de actualización
            evaluate_interval: Episodios entre evaluaciones
            render: Mostrar entorno gráficamente
            
        Returns:
            Historia de entrenamiento
        """
        history = {
            'episode_rewards': [],
            'actor_losses': [],
            'critic_losses': [],
            'alpha_losses': [],
            'entropies': [],
            'alphas': [],
            'eval_rewards': []
        }
        
        # Variables para seguimiento de progreso
        best_reward = -float('inf')
        episode_reward_history = []
        total_steps = 0
        
        for episode in range(episodes):
            state, _ = env.reset()
            state = np.array(state, dtype=np.float32)
            episode_reward = 0
            episode_steps = 0
            
            # Variables para almacenar pérdidas del episodio
            episode_actor_loss = []
            episode_critic_loss = []
            episode_alpha_loss = []
            episode_entropy = []
            
            for step in range(max_steps):
                if render:
                    env.render()
                    
                # Durante warmup, usar acciones aleatorias
                if total_steps < warmup_steps:
                    action = np.random.uniform(self.action_low, self.action_high, self.action_dim)
                else:
                    action = self.get_action(state, deterministic=False)
                
                # Ejecutar acción
                next_state, reward, done, _, _ = env.step(action)
                next_state = np.array(next_state, dtype=np.float32)
                
                # Guardar transición en buffer
                self.replay_buffer.add(state, action, reward, next_state, done)
                
                # Actualizar estado y contador
                state = next_state
                episode_reward += reward
                episode_steps += 1
                total_steps += 1
                
                # Entrenar si es momento
                if total_steps >= update_after and total_steps % update_every == 0:
                    # Realizar múltiples actualizaciones
                    for _ in range(update_every):
                        actor_loss, critic_loss, alpha_loss, entropy = self.train_step()
                        
                        # Almacenar pérdidas si hubo actualización
                        if actor_loss is not None:
                            episode_actor_loss.append(actor_loss.numpy())
                            episode_critic_loss.append(critic_loss.numpy())
                            episode_alpha_loss.append(alpha_loss.numpy())
                            episode_entropy.append(entropy.numpy())
                
                if done:
                    break
            
            # Almacenar métricas del episodio
            history['episode_rewards'].append(episode_reward)
            
            if episode_actor_loss:
                history['actor_losses'].append(np.mean(episode_actor_loss))
                history['critic_losses'].append(np.mean(episode_critic_loss))
                history['alpha_losses'].append(np.mean(episode_alpha_loss))
                history['entropies'].append(np.mean(episode_entropy))
                history['alphas'].append(float(tf.exp(self.log_alpha).numpy()))
            else:
                # Si no hubo actualizaciones, registrar NaN
                history['actor_losses'].append(float('nan'))
                history['critic_losses'].append(float('nan'))
                history['alpha_losses'].append(float('nan'))
                history['entropies'].append(float('nan'))
                history['alphas'].append(float(tf.exp(self.log_alpha).numpy()))
            
            # Resetear métricas
            self.actor_loss_metric.reset_states()
            self.critic_loss_metric.reset_states()
            self.alpha_loss_metric.reset_states()
            self.entropy_metric.reset_states()
            self.q_value_metric.reset_states()
            
            # Guardar últimas recompensas para promedio
            episode_reward_history.append(episode_reward)
            if len(episode_reward_history) > evaluate_interval:
                episode_reward_history.pop(0)
            
            # Evaluar y mostrar progreso periódicamente
            if (episode + 1) % evaluate_interval == 0:
                avg_reward = np.mean(episode_reward_history)
                print(f"Episode {episode+1}/{episodes} - Average Reward: {avg_reward:.2f}, "
                      f"Alpha: {float(tf.exp(self.log_alpha).numpy()):.4f}")
                
                # Evaluar rendimiento actual
                eval_reward = self.evaluate(env, episodes=3, render=False)
                history['eval_rewards'].append(eval_reward)
                
                # Guardar mejor modelo
                if eval_reward > best_reward:
                    best_reward = eval_reward
                    print(f"Nuevo mejor modelo con recompensa de evaluación: {best_reward:.2f}")
        
        return history
    
    def evaluate(self, env, episodes=10, render=False):
        """
        Evalúa el agente SAC en un entorno dado sin exploración.
        
        Args:
            env: Entorno de Gym o compatible
            episodes: Número de episodios para evaluar
            render: Si se debe renderizar el entorno
            
        Returns:
            Recompensa promedio obtenida
        """
        rewards = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            state = np.array(state, dtype=np.float32)
            episode_reward = 0
            done = False
            
            while not done:
                # Seleccionar acción determinística
                action = self.get_action(state, deterministic=True)
                
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
        
        avg_reward = np.mean(rewards)
        return avg_reward
    
    def save_models(self, directory):
        """
        Guarda los modelos y parámetros del agente.
        
        Args:
            directory: Directorio donde guardar los modelos
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Guardar redes
        self.actor.save_weights(os.path.join(directory, 'actor'))
        self.critic_1.save_weights(os.path.join(directory, 'critic_1'))
        self.critic_2.save_weights(os.path.join(directory, 'critic_2'))
        
        # Guardar alpha
        alpha = tf.exp(self.log_alpha).numpy()
        np.save(os.path.join(directory, 'alpha.npy'), alpha)
        
        print(f"Modelos guardados en {directory}")
    
    def load_models(self, directory):
        """
        Carga los modelos y parámetros del agente.
        
        Args:
            directory: Directorio de donde cargar los modelos
        """
        # Asegurar que los modelos estén construidos
        dummy_state = np.zeros((1, self.state_dim))
        dummy_action = np.zeros((1, self.action_dim))
        
        # Construir modelos
        self.actor(dummy_state)
        self.critic_1([dummy_state, dummy_action])
        self.critic_2([dummy_state, dummy_action])
        self.target_critic_1([dummy_state, dummy_action])
        self.target_critic_2([dummy_state, dummy_action])
        
        # Cargar redes
        self.actor.load_weights(os.path.join(directory, 'actor'))
        self.critic_1.load_weights(os.path.join(directory, 'critic_1'))
        self.critic_2.load_weights(os.path.join(directory, 'critic_2'))
        
        # Copiar pesos a las redes target
        self.update_target_networks(tau=1.0)
        
        # Cargar alpha
        try:
            alpha = np.load(os.path.join(directory, 'alpha.npy'))
            self.log_alpha.assign(tf.math.log(alpha))
        except:
            print("No se pudo cargar alpha, usando el valor actual")
        
        print(f"Modelos cargados desde {directory}")
    
    def visualize_training(self, history, smoothing_window=10):
        """
        Visualiza los resultados del entrenamiento.
        
        Args:
            history: Historia de entrenamiento
            smoothing_window: Ventana para suavizado de gráficos
        """
        import matplotlib.pyplot as plt
        
        # Función para suavizar datos
        def smooth(data, window_size):
            if len(data) < window_size:
                return data
            kernel = np.ones(window_size) / window_size
            return np.convolve(data, kernel, mode='valid')
        
        # Crear figura con múltiples subplots
        fig, axs = plt.subplots(3, 2, figsize=(16, 12))
        
        # 1. Graficar recompensas de episodio
        rewards = history['episode_rewards']
        axs[0, 0].plot(rewards, alpha=0.3, color='blue')
        if len(rewards) > smoothing_window:
            smoothed_rewards = smooth(rewards, smoothing_window)
            axs[0, 0].plot(range(smoothing_window-1, len(rewards)), 
                          smoothed_rewards, color='blue', label='Suavizado')
        axs[0, 0].set_title('Recompensa por Episodio')
        axs[0, 0].set_xlabel('Episodio')
        axs[0, 0].set_ylabel('Recompensa')
        axs[0, 0].grid(alpha=0.3)
        axs[0, 0].legend()
        
        # 2. Graficar recompensas de evaluación
        eval_rewards = history['eval_rewards']
        if eval_rewards:
            eval_interval = len(rewards) // len(eval_rewards)
            x_eval = [i * eval_interval for i in range(len(eval_rewards))]
            axs[0, 1].plot(x_eval, eval_rewards, color='green', marker='o')
            axs[0, 1].set_title('Recompensa de Evaluación')
            axs[0, 1].set_xlabel('Episodio')
            axs[0, 1].set_ylabel('Recompensa Promedio')
            axs[0, 1].grid(alpha=0.3)
        
        # 3. Graficar pérdidas del actor
        actor_losses = [l for l in history['actor_losses'] if not np.isnan(l)]
        if actor_losses:
            axs[1, 0].plot(actor_losses, alpha=0.3, color='red')
            if len(actor_losses) > smoothing_window:
                smoothed_actor_losses = smooth(actor_losses, smoothing_window)
                axs[1, 0].plot(range(smoothing_window-1, len(actor_losses)), 
                              smoothed_actor_losses, color='red', label='Suavizado')
            axs[1, 0].set_title('Pérdida del Actor')
            axs[1, 0].set_xlabel('Episodio')
            axs[1, 0].set_ylabel('Pérdida')
            axs[1, 0].grid(alpha=0.3)
            axs[1, 0].legend()
        
        # 4. Graficar pérdidas del crítico
        critic_losses = [l for l in history['critic_losses'] if not np.isnan(l)]
        if critic_losses:
            axs[1, 1].plot(critic_losses, alpha=0.3, color='purple')
            if len(critic_losses) > smoothing_window:
                smoothed_critic_losses = smooth(critic_losses, smoothing_window)
                axs[1, 1].plot(range(smoothing_window-1, len(critic_losses)), 
                              smoothed_critic_losses, color='purple', label='Suavizado')
            axs[1, 1].set_title('Pérdida del Crítico')
            axs[1, 1].set_xlabel('Episodio')
            axs[1, 1].set_ylabel('Pérdida')
            axs[1, 1].grid(alpha=0.3)
            axs[1, 1].legend()
        
        # 5. Graficar entropía
        entropies = [e for e in history['entropies'] if not np.isnan(e)]
        if entropies:
            axs[2, 0].plot(entropies, alpha=0.3, color='orange')
            if len(entropies) > smoothing_window:
                smoothed_entropies = smooth(entropies, smoothing_window)
                axs[2, 0].plot(range(smoothing_window-1, len(entropies)), 
                              smoothed_entropies, color='orange', label='Suavizado')
            axs[2, 0].set_title('Entropía')
            axs[2, 0].set_xlabel('Episodio')
            axs[2, 0].set_ylabel('Entropía')
            axs[2, 0].grid(alpha=0.3)
            axs[2, 0].legend()
        
        # 6. Graficar alpha
        alphas = history['alphas']
        axs[2, 1].plot(alphas, color='brown')
        axs[2, 1].set_title('Coeficiente Alpha')
        axs[2, 1].set_xlabel('Episodio')
        axs[2, 1].set_ylabel('Alpha')
        axs[2, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()