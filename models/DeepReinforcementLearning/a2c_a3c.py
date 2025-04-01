import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Conv1D, LSTM, Flatten, Concatenate,
    BatchNormalization, Dropout, LayerNormalization
)
from tensorflow.keras.optimizers import Adam
import threading
from ..config import A2C_A3C_CONFIG


class ActorCriticModel(Model):
    """
    Modelo Actor-Crítico para A2C que divide la arquitectura en redes para
    política (actor) y valor (crítico).
    """
    def __init__(self, state_dim, action_dim, continuous=True, hidden_units=None):
        super(ActorCriticModel, self).__init__()
        
        # Valores predeterminados para capas ocultas
        if hidden_units is None:
            hidden_units = A2C_A3C_CONFIG['hidden_units']
        
        self.continuous = continuous
        self.action_dim = action_dim
        
        # Capas compartidas para procesamiento de estados
        self.shared_layers = []
        for i, units in enumerate(hidden_units[:2]):
            self.shared_layers.append(Dense(units, activation='tanh', name=f'shared_dense_{i}'))
            self.shared_layers.append(LayerNormalization(epsilon=A2C_A3C_CONFIG['epsilon'], name=f'shared_ln_{i}'))
            self.shared_layers.append(Dropout(A2C_A3C_CONFIG['dropout_rate'], name=f'shared_dropout_{i}'))
        
        # Red del Actor (política)
        self.actor_layers = []
        for i, units in enumerate(hidden_units[2:]):
            self.actor_layers.append(Dense(units, activation='tanh', name=f'actor_dense_{i}'))
            self.actor_layers.append(LayerNormalization(epsilon=A2C_A3C_CONFIG['epsilon'], name=f'actor_ln_{i}'))
        
        # Capa de salida del actor (depende de si el espacio de acción es continuo o discreto)
        if continuous:
            # Para acción continua (política gaussiana)
            self.mu = Dense(action_dim, activation='linear', name='actor_mu')
            self.log_sigma = Dense(action_dim, activation='linear', name='actor_log_sigma')
        else:
            # Para acción discreta (política categórica)
            self.logits = Dense(action_dim, activation='linear', name='actor_logits')
        
        # Red del Crítico (valor)
        self.critic_layers = []
        for i, units in enumerate(hidden_units[2:]):
            self.critic_layers.append(Dense(units, activation='tanh', name=f'critic_dense_{i}'))
            self.critic_layers.append(LayerNormalization(epsilon=A2C_A3C_CONFIG['epsilon'], name=f'critic_ln_{i}'))
        
        # Capa de salida del crítico (valor del estado)
        self.value = Dense(1, activation='linear', name='critic_value')
    
    def call(self, inputs, training=False):
        x = inputs
        
        # Capas compartidas
        for layer in self.shared_layers:
            if isinstance(layer, Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        
        # Red del Actor
        actor_x = x
        for layer in self.actor_layers:
            if isinstance(layer, Dropout):
                actor_x = layer(actor_x, training=training)
            else:
                actor_x = layer(actor_x)
        
        # Salida del actor según el tipo de política
        if self.continuous:
            mu = self.mu(actor_x)
            log_sigma = self.log_sigma(actor_x)
            log_sigma = tf.clip_by_value(log_sigma, -20, 2)  # Evitar valores extremos
            sigma = tf.exp(log_sigma)
            policy = (mu, sigma)
        else:
            logits = self.logits(actor_x)
            policy = logits
        
        # Red del Crítico
        critic_x = x
        for layer in self.critic_layers:
            if isinstance(layer, Dropout):
                critic_x = layer(critic_x, training=training)
            else:
                critic_x = layer(critic_x)
        
        value = self.value(critic_x)
        
        return policy, value
    
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
        policy, _ = self.call(state)
        
        if self.continuous:
            mu, sigma = policy
            if deterministic:
                return mu[0]
            # Muestrear de la distribución normal
            dist = tf.random.normal(shape=mu.shape)
            action = mu + sigma * dist
            return action[0]
        else:
            logits = policy
            if deterministic:
                return tf.argmax(logits[0]).numpy()
            # Muestrear de la distribución categórica
            probs = tf.nn.softmax(logits)
            action = tf.random.categorical(tf.math.log(probs), 1)
            return action[0, 0].numpy()
    
    def get_value(self, state):
        """
        Obtiene el valor estimado para un estado.
        
        Args:
            state: El estado para evaluar
        
        Returns:
            El valor estimado del estado
        """
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        _, value = self.call(state)
        return value[0]
    
    def evaluate_actions(self, states, actions):
        """
        Evalúa las acciones tomadas, devolviendo log_probs, valores y entropía.
        
        Args:
            states: Los estados observados
            actions: Las acciones tomadas
        
        Returns:
            Tuple con log_probs, valores y entropía
        """
        policy, values = self.call(states)
        
        if self.continuous:
            mu, sigma = policy
            # Calcular log probabilidad para acciones continuas
            log_probs = -0.5 * tf.reduce_sum(
                tf.square((actions - mu) / sigma) + 
                2 * tf.math.log(sigma) + 
                tf.math.log(2.0 * np.pi), 
                axis=1
            )
            # Entropía de política gaussiana
            entropy = tf.reduce_sum(
                0.5 * tf.math.log(2.0 * np.pi * tf.square(sigma)) + 0.5,
                axis=1
            )
        else:
            logits = policy
            # Calcular log probabilidad para acciones discretas
            action_masks = tf.one_hot(actions, self.action_dim)
            log_probs = tf.reduce_sum(
                action_masks * tf.nn.log_softmax(logits),
                axis=1
            )
            # Entropía de política categórica
            probs = tf.nn.softmax(logits)
            entropy = -tf.reduce_sum(
                probs * tf.math.log(probs + 1e-10),
                axis=1
            )
        
        return log_probs, values, entropy


class A2C:
    """
    Implementación del algoritmo Advantage Actor-Critic (A2C).
    
    Este algoritmo utiliza un estimador de ventaja para actualizar la política
    y una red de valor para estimar los retornos esperados.
    """
    def __init__(
        self, 
        state_dim, 
        action_dim,
        continuous=True,
        learning_rate=A2C_A3C_CONFIG['learning_rate'],
        gamma=A2C_A3C_CONFIG['gamma'],
        entropy_coef=A2C_A3C_CONFIG['entropy_coef'],
        value_coef=A2C_A3C_CONFIG['value_coef'],
        max_grad_norm=A2C_A3C_CONFIG['max_grad_norm'],
        hidden_units=None
    ):
        # Parámetros del modelo
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.continuous = continuous
        
        # Valores predeterminados para capas ocultas
        if hidden_units is None:
            self.hidden_units = A2C_A3C_CONFIG['hidden_units']
        else:
            self.hidden_units = hidden_units
        
        # Crear modelo y optimizador
        self.model = ActorCriticModel(state_dim, action_dim, continuous, self.hidden_units)
        self.optimizer = Adam(learning_rate=learning_rate)
        
        # Métricas
        self.policy_loss_metric = tf.keras.metrics.Mean('policy_loss')
        self.value_loss_metric = tf.keras.metrics.Mean('value_loss')
        self.entropy_metric = tf.keras.metrics.Mean('entropy')
        self.total_loss_metric = tf.keras.metrics.Mean('total_loss')
    
    @tf.function
    def train_step(self, states, actions, returns, advantages):
        """
        Realiza un paso de entrenamiento para actualizar el modelo.
        
        Args:
            states: Estados observados en el entorno
            actions: Acciones tomadas para esos estados
            returns: Retornos estimados (para entrenar el crítico)
            advantages: Ventajas estimadas (para entrenar el actor)
        
        Returns:
            Pérdida total, pérdida de política, pérdida de valor, entropía
        """
        with tf.GradientTape() as tape:
            # Evaluar acciones con el modelo actual
            log_probs, values, entropy = self.model.evaluate_actions(states, actions)
            
            # Ventaja ya está calculada externamente
            advantages = tf.reshape(advantages, [-1])
            
            # Calcular pérdida de política
            policy_loss = -tf.reduce_mean(log_probs * advantages, axis=0)
            
            # Calcular pérdida de valor
            value_pred = tf.reshape(values, [-1])
            returns = tf.reshape(returns, [-1])
            value_loss = tf.reduce_mean(tf.square(returns - value_pred), axis=0)
            
            # Calcular pérdida de entropía (regularización)
            entropy_loss = -tf.reduce_mean(entropy, axis=0)
            
            # Pérdida total combinada
            total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
        # Calcular gradientes y actualizar pesos
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        
        # Clipping de gradientes para estabilidad
        if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
            
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        # Actualizar métricas
        self.policy_loss_metric.update_state(policy_loss)
        self.value_loss_metric.update_state(value_loss)
        self.entropy_metric.update_state(entropy_loss)
        self.total_loss_metric.update_state(total_loss)
        
        return total_loss, policy_loss, value_loss, entropy_loss
    
    def compute_returns_advantages(self, rewards, values, dones, next_value):
        """
        Calcula los retornos y ventajas para los estados visitados.
        
        Args:
            rewards: Recompensas recibidas
            values: Valores estimados para los estados actuales
            dones: Indicadores de fin de episodio
            next_value: Valor estimado para el estado final
            
        Returns:
            returns y ventajas calculados
        """
        # Añadir el valor del último estado
        values = np.append(values, next_value)
        
        returns = np.zeros_like(rewards, dtype=np.float32)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        
        # Calcular retornos y ventajas desde el final
        gae = 0
        for t in reversed(range(len(rewards))):
            # Si es terminal, el valor del siguiente estado es 0
            next_non_terminal = 1.0 - dones[t]
            next_value = values[t + 1]
            
            # Delta temporal para GAE
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            
            # Calcular ventaja con GAE
            gae = delta + self.gamma * A2C_A3C_CONFIG['lambda'] * next_non_terminal * gae
            advantages[t] = gae
            
            # Calcular retornos (para entrenar el crítico)
            returns[t] = advantages[t] + values[t]
        
        # Normalizar ventajas para reducir varianza
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def train(self, env, n_steps=10, epochs=1000, render=False):
        """
        Entrena el modelo A2C en el entorno dado.
        
        Args:
            env: Entorno donde entrenar
            n_steps: Número de pasos por actualización
            epochs: Número de épocas de entrenamiento
            render: Si se debe renderizar el entorno
            
        Returns:
            Historia de entrenamiento
        """
        # Historia de entrenamiento
        history = {
            'episode_rewards': [],
            'policy_losses': [],
            'value_losses': [],
            'entropy_losses': []
        }
        
        episode_reward = 0
        episode_rewards = []
        
        # Estado inicial
        state, _ = env.reset()
        
        for epoch in range(epochs):
            # Almacenar transiciones
            states, actions, rewards, dones, values = [], [], [], [], []
            
            # Recolectar experiencia durante n pasos
            for _ in range(n_steps):
                if render:
                    env.render()
                
                # Guardar estado actual
                states.append(state)
                
                # Obtener acción y valor
                action = self.model.get_action(state)
                actions.append(action)
                
                # Valor del estado actual
                value = self.model.get_value(state).numpy()
                values.append(value)
                
                # Ejecutar acción en el entorno
                next_state, reward, done, _, _ = env.step(action)
                
                # Guardar recompensa y done
                rewards.append(reward)
                dones.append(done)
                
                # Actualizar recompensa acumulada
                episode_reward += reward
                
                # Si el episodio termina, resetear
                if done:
                    state, _ = env.reset()
                    episode_rewards.append(episode_reward)
                    episode_reward = 0
                else:
                    state = next_state
                    
            # Si el episodio no terminó, calcular valor del último estado
            if not done:
                next_value = self.model.get_value(state).numpy()
            else:
                next_value = 0
                
            # Convertir a arrays de numpy
            states = np.array(states, dtype=np.float32)
            actions = np.array(actions, dtype=np.float32 if self.continuous else np.int32)
            rewards = np.array(rewards, dtype=np.float32)
            dones = np.array(dones, dtype=np.float32)
            values = np.array(values, dtype=np.float32)
            
            # Calcular retornos y ventajas
            returns, advantages = self.compute_returns_advantages(rewards, values, dones, next_value)
            
            # Actualizar modelo
            _, policy_loss, value_loss, _ = self.train_step(
                states, actions, returns, advantages
            )
            
            # Guardar estadísticas
            history['policy_losses'].append(self.policy_loss_metric.result().numpy())
            history['value_losses'].append(self.value_loss_metric.result().numpy())
            history['entropy_losses'].append(self.entropy_metric.result().numpy())
            
            # Resetear métricas
            self.policy_loss_metric.reset_states()
            self.value_loss_metric.reset_states()
            self.entropy_metric.reset_states()
            self.total_loss_metric.reset_states()
            
            # Añadir recompensas de episodios completados
            if episode_rewards:
                history['episode_rewards'].extend(episode_rewards)
                avg_reward = np.mean(episode_rewards)
                episode_rewards = []
                
                # Mostrar progreso cada 10 épocas
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Avg Reward: {avg_reward:.2f}, "
                          f"Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}")
        
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


class A3C(A2C):
    """
    Implementación de Asynchronous Advantage Actor-Critic (A3C).
    
    Extiende A2C para permitir entrenamiento asíncrono con múltiples trabajadores.
    """
    def __init__(self, state_dim, action_dim, continuous=True, n_workers=4, **kwargs):
        super(A3C, self).__init__(state_dim, action_dim, continuous, **kwargs)
        self.n_workers = n_workers
        self.workers = []
    
    def create_worker(self, env_fn, worker_id):
        """
        Crea un trabajador para entrenamiento asíncrono.
        
        Args:
            env_fn: Función que devuelve un entorno
            worker_id: ID del trabajador
            
        Returns:
            Un trabajador A3C
        """
        return A3CWorker(
            self.model,
            self.optimizer,
            env_fn,
            worker_id,
            self.state_dim,
            self.action_dim,
            self.gamma,
            self.entropy_coef,
            self.value_coef,
            self.max_grad_norm,
            self.continuous
        )
    
    def train_async(self, env_fn, n_steps=10, total_steps=1000000, render=False):
        """
        Entrena el modelo A3C con múltiples trabajadores asíncronos.
        
        Args:
            env_fn: Función que devuelve un entorno
            n_steps: Pasos por actualización
            total_steps: Total de pasos globales
            render: Si se debe renderizar el entorno
            
        Returns:
            Historia de entrenamiento
        """
        # Crear trabajadores
        workers = []
        for i in range(self.n_workers):
            worker = self.create_worker(env_fn, i)
            workers.append(worker)
        
        # Variables para seguimiento
        global_step = 0
        history = {
            'episode_rewards': [],
            'policy_losses': [],
            'value_losses': [],
            'entropy_losses': []
        }
        
        # Crear e iniciar hilos
        threads = []
        for worker in workers:
            thread = threading.Thread(
                target=worker.train,
                args=(n_steps, total_steps // self.n_workers, history, render)
            )
            threads.append(thread)
            thread.start()
        
        # Esperar a que terminen todos los hilos
        for thread in threads:
            thread.join()
        
        return history


class A3CWorker:
    """
    Trabajador para el algoritmo A3C que entrena de forma asíncrona.
    """
    def __init__(
        self,
        global_model,
        optimizer,
        env_fn,
        worker_id,
        state_dim,
        action_dim,
        gamma=0.99,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        continuous=True
    ):
        # Parámetros del trabajador
        self.worker_id = worker_id
        self.env = env_fn()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.continuous = continuous
        
        # Modelo global compartido
        self.global_model = global_model
        self.optimizer = optimizer
        
        # Modelo local para este trabajador
        self.local_model = ActorCriticModel(
            state_dim, action_dim, continuous, 
            hidden_units=A2C_A3C_CONFIG['hidden_units']
        )
        # Sincronizar pesos locales con globales
        self.update_local_model()
    
    def update_local_model(self):
        """Actualiza los pesos del modelo local desde el modelo global."""
        self.local_model.set_weights(self.global_model.get_weights())
    
    def compute_returns_advantages(self, rewards, values, dones, next_value):
        """
        Calcula los retornos y ventajas para los estados visitados.
        
        Args:
            rewards: Recompensas recibidas
            values: Valores estimados para los estados actuales
            dones: Indicadores de fin de episodio
            next_value: Valor estimado para el estado final
            
        Returns:
            returns y ventajas calculados
        """
        # Añadir el valor del último estado
        values = np.append(values, next_value)
        
        returns = np.zeros_like(rewards, dtype=np.float32)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        
        # Calcular retornos y ventajas desde el final
        gae = 0
        for t in reversed(range(len(rewards))):
            # Si es terminal, el valor del siguiente estado es 0
            next_non_terminal = 1.0 - dones[t]
            next_value = values[t + 1]
            
            # Delta temporal para GAE
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            
            # Calcular ventaja con GAE
            gae = delta + self.gamma * A2C_A3C_CONFIG['lambda'] * next_non_terminal * gae
            advantages[t] = gae
            
            # Calcular retornos (para entrenar el crítico)
            returns[t] = advantages[t] + values[t]
        
        # Normalizar ventajas para reducir varianza
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def train_step(self, states, actions, returns, advantages):
        """
        Realiza un paso de entrenamiento asíncrono.
        
        Args:
            states: Estados observados
            actions: Acciones tomadas
            returns: Retornos calculados
            advantages: Ventajas calculadas
            
        Returns:
            Pérdidas calculadas
        """
        with tf.GradientTape() as tape:
            # Evaluar acciones con el modelo local
            log_probs, values, entropy = self.local_model.evaluate_actions(states, actions)
            
            # Calcular pérdida de política
            advantages = tf.reshape(advantages, [-1])
            policy_loss = -tf.reduce_mean(log_probs * advantages, axis=0)
            
            # Calcular pérdida de valor
            value_pred = tf.reshape(values, [-1])
            returns = tf.reshape(returns, [-1])
            value_loss = tf.reduce_mean(tf.square(returns - value_pred), axis=0)
            
            # Calcular pérdida de entropía (regularización)
            entropy_loss = -tf.reduce_mean(entropy, axis=0)
            
            # Pérdida total combinada
            total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
        # Calcular gradientes usando el modelo local
        grads = tape.gradient(total_loss, self.local_model.trainable_variables)
        
        # Clipping de gradientes para estabilidad
        if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
            
        # Aplicar gradientes al modelo global de manera asíncrona
        self.optimizer.apply_gradients(zip(grads, self.global_model.trainable_variables))
        
        # Actualizar modelo local
        self.update_local_model()
        
        return total_loss, policy_loss, value_loss, entropy_loss
    
    def train(self, n_steps, max_steps, shared_history, render=False):
        """
        Entrenamiento asíncrono del trabajador.
        
        Args:
            n_steps: Pasos por actualización
            max_steps: Pasos máximos para este trabajador
            shared_history: Diccionario compartido para seguimiento
            render: Si se debe renderizar el entorno
        """
        # Estado inicial
        state, _ = self.env.reset()
        episode_reward = 0
        episode_step = 0
        
        steps_done = 0
        
        while steps_done < max_steps:
            # Almacenar transiciones
            states, actions, rewards, dones, values = [], [], [], [], []
            
            # Recolectar experiencia durante n pasos
            for _ in range(n_steps):
                if render and self.worker_id == 0:  # Solo renderizar el primer trabajador
                    self.env.render()
                
                # Guardar estado actual
                states.append(state)
                
                # Obtener acción y valor con el modelo local
                action = self.local_model.get_action(state)
                actions.append(action)
                
                # Valor del estado actual
                value = self.local_model.get_value(state).numpy()
                values.append(value)
                
                # Ejecutar acción en el entorno
                next_state, reward, done, _, _ = self.env.step(action)
                
                # Guardar recompensa y done
                rewards.append(reward)
                dones.append(done)
                
                # Actualizar recompensa acumulada
                episode_reward += reward
                episode_step += 1
                steps_done += 1
                
                # Si el episodio termina, resetear
                if done:
                    state, _ = self.env.reset()
                    
                    # Guardar recompensa de episodio completado
                    with threading.Lock():  # Proteger acceso compartido
                        shared_history['episode_rewards'].append(episode_reward)
                    
                    # Mostrar progreso del trabajador
                    if self.worker_id == 0 and len(shared_history['episode_rewards']) % 10 == 0:
                        avg_reward = np.mean(shared_history['episode_rewards'][-10:])
                        print(f"Worker {self.worker_id} - Episode {len(shared_history['episode_rewards'])}, "
                              f"Avg Reward: {avg_reward:.2f}, Steps: {steps_done}/{max_steps}")
                    
                    episode_reward = 0
                    episode_step = 0
                    break
                else:
                    state = next_state
            
            # Si recolectamos suficientes pasos, actualizar modelo
            if len(states) > 0:
                # Si el episodio no terminó, calcular valor del último estado
                if not done:
                    next_value = self.local_model.get_value(state).numpy()
                else:
                    next_value = 0
                    
                # Convertir a arrays de numpy
                states_np = np.array(states, dtype=np.float32)
                actions_np = np.array(actions, dtype=np.float32 if self.continuous else np.int32)
                rewards_np = np.array(rewards, dtype=np.float32)
                dones_np = np.array(dones, dtype=np.float32)
                values_np = np.array(values, dtype=np.float32)
                
                # Calcular retornos y ventajas
                returns, advantages = self.compute_returns_advantages(
                    rewards_np, values_np, dones_np, next_value
                )
                
                # Actualizar modelo
                _, policy_loss, value_loss, entropy_loss = self.train_step(
                    states_np, actions_np, returns, advantages
                )
                
                # Guardar estadísticas
                with threading.Lock():  # Proteger acceso compartido
                    shared_history['policy_losses'].append(policy_loss.numpy())
                    shared_history['value_losses'].append(value_loss.numpy())
                    shared_history['entropy_losses'].append(entropy_loss.numpy())