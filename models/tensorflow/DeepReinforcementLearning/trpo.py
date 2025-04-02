import os, sys
import tensorflow as tf
import numpy as np
import time
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Conv1D, LSTM, Flatten, Concatenate,
    BatchNormalization, Dropout, LayerNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, TypeVar

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from models.config import TRPO_CONFIG


class ActorCriticModel:
    """
    Modelo Actor-Crítico para TRPO que divide la arquitectura en redes separadas
    para política (actor) y valor (crítico).
    
    TRPO requiere modelos separados para actor y crítico en lugar de un único modelo.
    
    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    continuous : bool, opcional
        Si el espacio de acciones es continuo o discreto (default: True)
    hidden_units : Optional[List[int]], opcional
        Unidades en capas ocultas (default: None)
    """
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        continuous: bool = True, 
        hidden_units: Optional[List[int]] = None
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous
        
        # Valores predeterminados para capas ocultas
        if hidden_units is None:
            self.hidden_units = TRPO_CONFIG['hidden_units']
        else:
            self.hidden_units = hidden_units
        
        # Red para el Actor (política)
        self.actor = self._create_actor()
        
        # Red para el Crítico (valor)
        self.critic = self._create_critic()
        
        # Optimizer solo para el crítico (el actor se actualiza con TRPO)
        self.critic_optimizer = Adam(learning_rate=TRPO_CONFIG['critic_learning_rate'])
    
    def _create_actor(self) -> Model:
        """
        Crea la red del actor.
        
        Retorna:
        --------
        Model
            Modelo de Keras para el actor
        """
        inputs = Input(shape=(self.state_dim,))
        x = inputs
        
        # Capas ocultas
        for i, units in enumerate(self.hidden_units):
            x = Dense(units, activation='tanh', name=f'actor_dense_{i}')(x)
            if TRPO_CONFIG['use_layer_norm']:
                x = LayerNormalization(epsilon=TRPO_CONFIG['epsilon'], name=f'actor_ln_{i}')(x)
        
        # Salida depende de si las acciones son continuas o discretas
        if self.continuous:
            # Para acciones continuas (política gaussiana)
            mu = Dense(self.action_dim, activation='tanh', name='actor_mu')(x)
            log_std = tf.Variable(
                initial_value=-0.5 * np.ones(shape=(self.action_dim,), dtype=np.float32),
                trainable=True,
                name='actor_log_std'
            )
            
            return tf.keras.Model(inputs=inputs, outputs=[mu, log_std])
        else:
            # Para acciones discretas (política categórica)
            logits = Dense(self.action_dim, activation=None, name='actor_logits')(x)
            return tf.keras.Model(inputs=inputs, outputs=logits)
    
    def _create_critic(self) -> Model:
        """
        Crea la red del crítico.
        
        Retorna:
        --------
        Model
            Modelo de Keras para el crítico
        """
        inputs = Input(shape=(self.state_dim,))
        x = inputs
        
        # Capas ocultas
        for i, units in enumerate(self.hidden_units):
            x = Dense(units, activation='tanh', name=f'critic_dense_{i}')(x)
            if TRPO_CONFIG['use_layer_norm']:
                x = LayerNormalization(epsilon=TRPO_CONFIG['epsilon'], name=f'critic_ln_{i}')(x)
        
        # Capa de salida (valor del estado)
        value = Dense(1, activation=None, name='value')(x)
        
        return tf.keras.Model(inputs=inputs, outputs=value)
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Obtiene una acción basada en el estado actual.
        
        Parámetros:
        -----------
        state : np.ndarray
            El estado actual
        deterministic : bool, opcional
            Si es True, devuelve la acción con máxima probabilidad (default: False)
        
        Retorna:
        --------
        np.ndarray
            Una acción muestreada de la distribución de política
        """
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        
        if self.continuous:
            mu, log_std = self.actor(state)
            std = tf.exp(log_std)
            
            if deterministic:
                return mu[0].numpy()
            
            # Muestrear de la distribución normal
            action = mu + tf.random.normal(shape=mu.shape) * std
            return action[0].numpy()
        else:
            logits = self.actor(state)
            
            if deterministic:
                return tf.argmax(logits[0]).numpy()
            
            # Muestrear de la distribución categórica
            action = tf.random.categorical(logits, 1)
            return action[0, 0].numpy()
    
    def get_action_distribution_params(self, states: tf.Tensor) -> Union[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
        """
        Obtiene los parámetros de la distribución de política para estados dados.
        
        Parámetros:
        -----------
        states : tf.Tensor
            Estados para evaluar
            
        Retorna:
        --------
        Union[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]
            Parámetros de la distribución (mu, log_std para continuo, logits para discreto)
        """
        return self.actor(states)
    
    def get_value(self, state: np.ndarray) -> float:
        """
        Obtiene el valor estimado para un estado.
        
        Parámetros:
        -----------
        state : np.ndarray
            El estado para evaluar
            
        Retorna:
        --------
        float
            El valor estimado del estado
        """
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        return self.critic(state)[0, 0].numpy()
    
    def get_log_prob(self, states: tf.Tensor, actions: tf.Tensor) -> tf.Tensor:
        """
        Calcula el logaritmo de probabilidad de acciones bajo la política actual.
        
        Parámetros:
        -----------
        states : tf.Tensor
            Estados observados
        actions : tf.Tensor
            Acciones tomadas
            
        Retorna:
        --------
        tf.Tensor
            Logaritmo de probabilidad de las acciones
        """
        if self.continuous:
            mu, log_std = self.actor(states)
            std = tf.exp(log_std)
            
            # Log prob para política gaussiana
            log_probs = -0.5 * (
                tf.reduce_sum(
                    tf.square((actions - mu) / std) + 
                    2 * log_std + 
                    tf.math.log(2.0 * np.pi),
                    axis=1
                )
            )
            return log_probs
        else:
            logits = self.actor(states)
            # Log prob para política categórica
            action_masks = tf.one_hot(tf.cast(actions, tf.int32), self.action_dim)
            log_probs = tf.reduce_sum(
                action_masks * tf.nn.log_softmax(logits),
                axis=1
            )
            return log_probs
    
    def get_kl_divergence(
        self, 
        states: tf.Tensor, 
        old_mu: Optional[tf.Tensor] = None, 
        old_log_std: Optional[tf.Tensor] = None, 
        old_logits: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """
        Calcula la divergencia KL entre la política antigua y la actual.
        
        Parámetros:
        -----------
        states : tf.Tensor
            Estados para evaluar
        old_mu : Optional[tf.Tensor], opcional
            Medias de la política antigua (para continuo) (default: None)
        old_log_std : Optional[tf.Tensor], opcional
            Log std de la política antigua (para continuo) (default: None)
        old_logits : Optional[tf.Tensor], opcional
            Logits de la política antigua (para discreto) (default: None)
            
        Retorna:
        --------
        tf.Tensor
            Divergencia KL media
        """
        if self.continuous:
            mu, log_std = self.actor(states)
            std = tf.exp(log_std)
            old_std = tf.exp(old_log_std)
            
            # KL para distribuciones normales
            kl = tf.reduce_sum(
                log_std - old_log_std + 
                (tf.square(old_std) + tf.square(old_mu - mu)) / (2.0 * tf.square(std)) - 0.5,
                axis=1
            )
            return tf.reduce_mean(kl, axis=0)
        else:
            logits = self.actor(states)
            
            # KL para distribuciones categóricas
            old_logp = tf.nn.log_softmax(old_logits)
            logp = tf.nn.log_softmax(logits)
            kl = tf.reduce_sum(
                tf.exp(old_logp) * (old_logp - logp),
                axis=1
            )
            return tf.reduce_mean(kl, axis=0)
    
    def get_entropy(self, states: tf.Tensor) -> tf.Tensor:
        """
        Calcula la entropía de la política para estados dados.
        
        Parámetros:
        -----------
        states : tf.Tensor
            Estados para evaluar
            
        Retorna:
        --------
        tf.Tensor
            Entropía media de la política
        """
        if self.continuous:
            _, log_std = self.actor(states)
            entropy = tf.reduce_sum(
                log_std + 0.5 * tf.math.log(2.0 * np.pi * np.e),
                axis=1
            )
        else:
            logits = self.actor(states)
            probs = tf.nn.softmax(logits)
            entropy = -tf.reduce_sum(
                probs * tf.math.log(probs + 1e-8),
                axis=1
            )
        
        return tf.reduce_mean(entropy, axis=0)
    
    def get_flat_params(self) -> tf.Tensor:
        """
        Obtiene los parámetros de la red del actor en un vector plano.
        
        Retorna:
        --------
        tf.Tensor
            Vector con los parámetros
        """
        params = []
        for var in self.actor.trainable_variables:
            params.append(tf.reshape(var, [-1]))
        return tf.concat(params, axis=0)
    
    def set_flat_params(self, flat_params: tf.Tensor) -> None:
        """
        Establece los parámetros de la red del actor desde un vector plano.
        
        Parámetros:
        -----------
        flat_params : tf.Tensor
            Vector con los nuevos parámetros
        """
        start_idx = 0
        for var in self.actor.trainable_variables:
            shape = var.shape
            size = tf.reduce_prod(shape, axis=0)
            var.assign(tf.reshape(flat_params[start_idx:start_idx + size], shape))
            start_idx += size


class TRPO:
    """
    Implementación de Trust Region Policy Optimization (TRPO).
    
    TRPO es un algoritmo de optimización de política que actualiza la política
    de forma conservadora, manteniendo las actualizaciones dentro de una región
    de confianza definida por una restricción de divergencia KL.
    
    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    continuous : bool, opcional
        Si el espacio de acciones es continuo o discreto (default: True)
    gamma : float, opcional
        Factor de descuento (default: TRPO_CONFIG['gamma'])
    delta : float, opcional
        Límite de divergencia KL (default: TRPO_CONFIG['delta'])
    hidden_units : Optional[List[int]], opcional
        Unidades en capas ocultas (default: None)
    backtrack_iters : int, opcional
        Iteraciones para búsqueda de línea (default: TRPO_CONFIG['backtrack_iters'])
    backtrack_coeff : float, opcional
        Coeficiente de reducción para búsqueda de línea (default: TRPO_CONFIG['backtrack_coeff'])
    cg_iters : int, opcional
        Iteraciones para gradiente conjugado (default: TRPO_CONFIG['cg_iters'])
    damping : float, opcional
        Término de regularización para estabilidad numérica (default: TRPO_CONFIG['damping'])
    """
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int,
        continuous: bool = True,
        gamma: float = TRPO_CONFIG['gamma'],
        delta: float = TRPO_CONFIG['delta'],
        hidden_units: Optional[List[int]] = None,
        backtrack_iters: int = TRPO_CONFIG['backtrack_iters'],
        backtrack_coeff: float = TRPO_CONFIG['backtrack_coeff'],
        cg_iters: int = TRPO_CONFIG['cg_iters'],
        damping: float = TRPO_CONFIG['damping']
    ) -> None:
        # Configurar semilla para reproducibilidad
        seed = TRPO_CONFIG.get('seed', 42)
        tf.random.set_seed(seed)
        np.random.seed(seed)
        
        # Parámetros del modelo
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous
        self.gamma = gamma  # Factor de descuento
        self.delta = delta  # Límite de divergencia KL
        self.backtrack_iters = backtrack_iters  # Iteraciones para búsqueda de línea
        self.backtrack_coeff = backtrack_coeff  # Coeficiente de reducción para búsqueda de línea
        self.cg_iters = cg_iters  # Iteraciones para gradiente conjugado
        self.damping = damping  # Término de regularización para estabilidad numérica
        
        # Valores predeterminados para capas ocultas
        if hidden_units is None:
            self.hidden_units = TRPO_CONFIG['hidden_units']
        else:
            self.hidden_units = hidden_units
        
        # Crear modelo actor-crítico
        self.model = ActorCriticModel(
            state_dim, 
            action_dim, 
            continuous, 
            self.hidden_units
        )
        
        # Métricas
        self.policy_loss_metric = tf.keras.metrics.Mean('policy_loss')
        self.value_loss_metric = tf.keras.metrics.Mean('value_loss')
        self.kl_metric = tf.keras.metrics.Mean('kl_divergence')
        self.entropy_metric = tf.keras.metrics.Mean('entropy')
    
    def compute_gae(
        self, 
        rewards: np.ndarray, 
        values: np.ndarray, 
        next_values: np.ndarray, 
        dones: np.ndarray, 
        lam: float = TRPO_CONFIG['lambda']
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula el Estimador de Ventaja Generalizada (GAE).
        
        Parámetros:
        -----------
        rewards : np.ndarray
            Recompensas recibidas
        values : np.ndarray
            Valores estimados para los estados actuales
        next_values : np.ndarray
            Valores estimados para los estados siguientes
        dones : np.ndarray
            Indicadores de fin de episodio
        lam : float, opcional
            Factor lambda para GAE (default: TRPO_CONFIG['lambda'])
            
        Retorna:
        --------
        Tuple[np.ndarray, np.ndarray]
            (ventajas, retornos)
        """
        # Añadir el valor del último estado
        values = np.append(values, next_values[-1])
        
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            # Si es terminal, el valor del siguiente estado es 0
            next_non_terminal = 1.0 - dones[t]
            next_value = values[t + 1]
            
            # Delta temporal para GAE
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            
            # Calcular ventaja con GAE
            advantages[t] = last_gae = delta + self.gamma * lam * next_non_terminal * last_gae
            
        # Calcular retornos para entrenar el crítico
        returns = advantages + values[:-1]
        
        # Normalizar ventajas para reducir varianza
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        return advantages, returns
    
    def fisher_vector_product(self, states: tf.Tensor, p: tf.Tensor) -> tf.Tensor:
        """
        Calcula el producto Fisher-vector para el método de gradiente conjugado.
        
        Parámetros:
        -----------
        states : tf.Tensor
            Estados
        p : tf.Tensor
            Vector para multiplicar con la matriz Fisher
            
        Retorna:
        --------
        tf.Tensor
            Producto Fisher-vector
        """
        # Guardar parámetros originales
        old_params = self.model.get_flat_params()
        
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape1:
                # Calcular divergencia KL
                if self.continuous:
                    old_mu, old_log_std = self.model.get_action_distribution_params(states)
                    kl = self.model.get_kl_divergence(states, old_mu, old_log_std)
                else:
                    old_logits = self.model.get_action_distribution_params(states)
                    kl = self.model.get_kl_divergence(states, old_logits=old_logits)
            
            # Grad_kl * p
            grads = tape1.gradient(kl, self.model.actor.trainable_variables)
            flat_grad_kl = tf.concat([tf.reshape(grad, [-1]) for grad in grads], axis=0)
            grad_kl_p = tf.reduce_sum(flat_grad_kl * p, axis=0)
        
        # Hessian * p = grad(grad_kl · p)
        hessian_p_grads = tape2.gradient(grad_kl_p, self.model.actor.trainable_variables)
        flat_hessian_p = tf.concat([tf.reshape(grad, [-1]) for grad in hessian_p_grads], axis=0)
        
        # Restaurar parámetros originales
        self.model.set_flat_params(old_params)
        
        # Añadir término de amortiguación para estabilidad numérica
        return flat_hessian_p + self.damping * p
    
    def conjugate_gradient(self, states: tf.Tensor, b: np.ndarray) -> np.ndarray:
        """
        Resuelve el sistema lineal Ax = b usando el método de gradiente conjugado.
        
        Parámetros:
        -----------
        states : tf.Tensor
            Estados
        b : np.ndarray
            Vector lado derecho de la ecuación
            
        Retorna:
        --------
        np.ndarray
            Solución aproximada x
        """
        x = np.zeros_like(b)
        r = b.copy()
        p = r.copy()
        r_dot_r = np.dot(r, r)
        
        for _ in range(self.cg_iters):
            Ap = self.fisher_vector_product(states, p)
            alpha = r_dot_r / (np.dot(p, Ap) + 1e-8)
            x += alpha * p
            r -= alpha * Ap
            
            new_r_dot_r = np.dot(r, r)
            beta = new_r_dot_r / (r_dot_r + 1e-8)
            p = r + beta * p
            r_dot_r = new_r_dot_r
            
            # Criterio de parada
            if np.sqrt(r_dot_r) < 1e-6:
                break
        
        return x
    
    def update_policy(
        self, 
        states: np.ndarray, 
        actions: np.ndarray, 
        advantages: np.ndarray
    ) -> Dict[str, float]:
        """
        Actualiza la política utilizando el método TRPO.
        
        Parámetros:
        -----------
        states : np.ndarray
            Estados
        actions : np.ndarray
            Acciones
        advantages : np.ndarray
            Ventajas
            
        Retorna:
        --------
        Dict[str, float]
            Métricas de actualización
        """
        start_time = time.time()
        
        # Convertir a tensores
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32 if self.continuous else tf.int32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        
        # Guardar parámetros de distribución antiguos para KL
        if self.continuous:
            old_mu, old_log_std = self.model.get_action_distribution_params(states)
            old_mu = tf.stop_gradient(old_mu)
            old_log_std = tf.stop_gradient(old_log_std)
        else:
            old_logits = self.model.get_action_distribution_params(states)
            old_logits = tf.stop_gradient(old_logits)
        
        # Calcular log probs antiguos
        old_log_probs = self.model.get_log_prob(states, actions)
        old_log_probs = tf.stop_gradient(old_log_probs)
        
        # Calcular gradiente del objetivo surrogate
        with tf.GradientTape() as tape:
            log_probs = self.model.get_log_prob(states, actions)
            ratio = tf.exp(log_probs - old_log_probs)
            surrogate_loss = -tf.reduce_mean(ratio * advantages, axis=0)
        
        policy_grads = tape.gradient(surrogate_loss, self.model.actor.trainable_variables)
        flat_policy_grad = tf.concat([tf.reshape(grad, [-1]) for grad in policy_grads], axis=0)
        
        # Parámetros actuales
        old_params = self.model.get_flat_params()
        
        # Calcular dirección de actualización usando gradiente conjugado
        step_direction = self.conjugate_gradient(states, flat_policy_grad.numpy())
        
        # Calcular tamaño de paso
        fvp = self.fisher_vector_product(states, step_direction)
        shs = 0.5 * np.dot(step_direction, fvp.numpy())
        lm = np.sqrt(shs / self.delta)
        full_step = step_direction / lm
        
        # Guardar parámetros actuales
        current_params = old_params.numpy()
        
        # Backtracking line search
        for i in range(self.backtrack_iters):
            # Probar nuevos parámetros
            new_params = current_params - self.backtrack_coeff**i * full_step
            self.model.set_flat_params(new_params)
            
            # Calcular nuevo surrogate loss
            new_log_probs = self.model.get_log_prob(states, actions)
            new_ratio = tf.exp(new_log_probs - old_log_probs)
            new_surrogate_loss = -tf.reduce_mean(new_ratio * advantages, axis=0)
            
            # Calcular nueva KL divergence
            if self.continuous:
                kl = self.model.get_kl_divergence(states, old_mu, old_log_std)
            else:
                kl = self.model.get_kl_divergence(states, old_logits=old_logits)
            
            improvement = surrogate_loss - new_surrogate_loss
            
            # Verificar si mejora y cumple restricción KL
            if improvement > 0 and kl < self.delta:
                print(f"Policy update: iter {i}, improvement: {improvement.numpy():.6f}, kl: {kl.numpy():.6f}")
                break
                
            # Si llegamos a la última iteración, restaurar parámetros originales
            if i == self.backtrack_iters - 1:
                print("Line search failed. Keeping old parameters.")
                self.model.set_flat_params(old_params)
        
        # Calcular entropía después de la actualización
        entropy = self.model.get_entropy(states)
        
        # Actualizar métricas
        self.policy_loss_metric.update_state(surrogate_loss)
        self.kl_metric.update_state(kl)
        self.entropy_metric.update_state(entropy)
        
        elapsed_time = time.time() - start_time
        return {
            'policy_loss': surrogate_loss.numpy(),
            'kl_divergence': kl.numpy(),
            'entropy': entropy.numpy(),
            'elapsed_time': elapsed_time
        }
    
    def update_value(
        self, 
        states: np.ndarray, 
        returns: np.ndarray, 
        epochs: int = TRPO_CONFIG['value_epochs'], 
        batch_size: int = TRPO_CONFIG['batch_size']
    ) -> Dict[str, float]:
        """
        Actualiza la red de valor (crítico) usando descenso de gradiente.
        
        Parámetros:
        -----------
        states : np.ndarray
            Estados
        returns : np.ndarray
            Retornos objetivo
        epochs : int, opcional
            Número de épocas para entrenar (default: TRPO_CONFIG['value_epochs'])
        batch_size : int, opcional
            Tamaño de lote (default: TRPO_CONFIG['batch_size'])
            
        Retorna:
        --------
        Dict[str, float]
            Estadísticas de actualización
        """
        start_time = time.time()
        
        # Convertir a tensores
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        losses = []
        
        # Crear generador con semilla fija para reproducibilidad
        seed = TRPO_CONFIG.get('seed', 42)
        rng = np.random.default_rng(seed)
        
        for _ in range(epochs):
            # Mezclar datos en cada época
            rng.shuffle(indices)
            
            for start_idx in range(0, dataset_size, batch_size):
                # Obtener lote
                idx = indices[start_idx:min(start_idx + batch_size, dataset_size)]
                batch_states = tf.gather(states, idx)
                batch_returns = tf.gather(returns, idx)
                
                # Actualizar red de valor
                with tf.GradientTape() as tape:
                    values = self.model.critic(batch_states)
                    # Asegurar que values y batch_returns tengan la misma forma
                    values = tf.squeeze(values)
                    loss = tf.reduce_mean(tf.square(values - batch_returns), axis=0)
                
                # Calcular gradientes y aplicarlos
                grads = tape.gradient(loss, self.model.critic.trainable_variables)
                self.model.critic_optimizer.apply_gradients(
                    zip(grads, self.model.critic.trainable_variables))
                
                losses.append(loss.numpy())
        
        # Actualizar métrica
        mean_loss = np.mean(losses)
        self.value_loss_metric.update_state(mean_loss)
        
        elapsed_time = time.time() - start_time
        return {
            'value_loss': mean_loss,
            'elapsed_time': elapsed_time
        }
    
    def collect_trajectories(
        self, 
        env: Any, 
        min_steps: int = TRPO_CONFIG['min_steps_per_update'], 
        render: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Recolecta trayectorias ejecutando la política actual en el entorno.
        
        Parámetros:
        -----------
        env : Any
            Entorno de OpenAI Gym o compatible
        min_steps : int, opcional
            Mínimo número de pasos antes de actualizar (default: TRPO_CONFIG['min_steps_per_update'])
        render : bool, opcional
            Si renderizar el entorno o no (default: False)
            
        Retorna:
        --------
        Dict[str, np.ndarray]
            Datos recolectados
        """
        states, actions, rewards, dones, next_states, values = [], [], [], [], [], []
        
        state, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        episode_rewards = []
        episode_lengths = []
        total_steps = 0
        
        # Recolectar hasta alcanzar min_steps
        while total_steps < min_steps:
            if render:
                env.render()
            
            # Guardar estado actual
            states.append(state)
            
            # Seleccionar acción
            action = self.model.get_action(state)
            actions.append(action)
            
            # Calcular valor del estado
            value = self.model.get_value(state)
            values.append(value)
            
            # Ejecutar acción
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Guardar resultado
            rewards.append(reward)
            dones.append(done)
            next_states.append(next_state)
            
            # Actualizar
            state = next_state
            episode_reward += reward
            episode_length += 1
            total_steps += 1
            
            # Si episodio termina, resetear
            if done:
                print(f"Episodio terminado con recompensa: {episode_reward}, longitud: {episode_length}")
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_reward = 0
                episode_length = 0
                state, _ = env.reset()
                done = False
        
        # Calcular ventajas y retornos
        next_values = [self.model.get_value(next_state) for next_state in next_states]
        advantages, returns = self.compute_gae(
            np.array(rewards), 
            np.array(values), 
            np.array(next_values), 
            np.array(dones)
        )
        
        return {
            'states': np.array(states, dtype=np.float32),
            'actions': np.array(actions, dtype=np.float32 if self.continuous else np.int32),
            'rewards': np.array(rewards, dtype=np.float32),
            'dones': np.array(dones, dtype=np.float32),
            'next_states': np.array(next_states, dtype=np.float32),
            'values': np.array(values, dtype=np.float32),
            'advantages': advantages,
            'returns': returns,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'total_steps': total_steps
        }
    
    def train(
        self, 
        env: Any, 
        iterations: int = TRPO_CONFIG['iterations'], 
        min_steps_per_update: int = TRPO_CONFIG['min_steps_per_update'],
        render: bool = False, 
        evaluate_interval: int = TRPO_CONFIG['evaluate_interval']
    ) -> Dict[str, List[float]]:
        """
        Entrena el agente TRPO.
        
        Parámetros:
        -----------
        env : Any
            Entorno para entrenar
        iterations : int, opcional
            Número de iteraciones de entrenamiento (default: TRPO_CONFIG['iterations'])
        min_steps_per_update : int, opcional
            Mínimo de pasos antes de actualizar la política (default: TRPO_CONFIG['min_steps_per_update'])
        render : bool, opcional
            Si renderizar el entorno o no (default: False)
        evaluate_interval : int, opcional
            Intervalos para evaluación (default: TRPO_CONFIG['evaluate_interval'])
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historia de entrenamiento
        """
        history = {
            'iterations': [],
            'policy_losses': [],
            'value_losses': [],
            'kl_divergences': [],
            'entropies': [],
            'mean_episode_rewards': [],
            'mean_episode_lengths': [],
            'steps_per_iteration': [],
            'evaluation_rewards': []
        }
        
        for i in range(iterations):
            print(f"\nIteración {i+1}/{iterations}")
            start_time = time.time()
            
            # Recolectar experiencias
            data = self.collect_trajectories(env, min_steps_per_update, render)
            
            # Actualizar política
            policy_stats = self.update_policy(data['states'], data['actions'], data['advantages'])
            
            # Actualizar red de valor
            value_stats = self.update_value(data['states'], data['returns'])
            
            # Calcular métricas
            mean_episode_reward = np.mean(data['episode_rewards']) if data['episode_rewards'] else 0
            mean_episode_length = np.mean(data['episode_lengths']) if data['episode_lengths'] else 0
            
            # Resetear métricas de keras
            self.policy_loss_metric.reset_states()
            self.value_loss_metric.reset_states()
            self.kl_metric.reset_states()
            self.entropy_metric.reset_states()
            
            # Guardar historia
            history['iterations'].append(i)
            history['policy_losses'].append(policy_stats['policy_loss'])
            history['value_losses'].append(value_stats['value_loss'])
            history['kl_divergences'].append(policy_stats['kl_divergence'])
            history['entropies'].append(policy_stats['entropy'])
            history['mean_episode_rewards'].append(mean_episode_reward)
            history['mean_episode_lengths'].append(mean_episode_length)
            history['steps_per_iteration'].append(data['total_steps'])
            
            # Evaluar
            if (i + 1) % evaluate_interval == 0:
                eval_reward = self.evaluate(env, episodes=5)
                history['evaluation_rewards'].append(eval_reward)
                print(f"Evaluación: Recompensa media = {eval_reward:.2f}")
            
            # Mostrar estadísticas
            elapsed_time = time.time() - start_time
            print(f"Tiempo total: {elapsed_time:.2f}s")
            print(f"KL Divergence: {policy_stats['kl_divergence']:.6f}")
            print(f"Entropía: {policy_stats['entropy']:.6f}")
            print(f"Recompensa media: {mean_episode_reward:.2f}")
            print(f"Longitud media: {mean_episode_length:.2f}")
        
        return history
    
    def evaluate(
        self, 
        env: Any, 
        episodes: int = 10, 
        render: bool = False
    ) -> float:
        """
        Evalúa el agente TRPO en el entorno.
        
        Parámetros:
        -----------
        env : Any
            Entorno para evaluar
        episodes : int, opcional
            Número de episodios (default: 10)
        render : bool, opcional
            Si renderizar el entorno o no (default: False)
            
        Retorna:
        --------
        float
            Recompensa media
        """
        rewards = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                if render:
                    env.render()
                
                # Usar política determinística para evaluación
                action = self.model.get_action(state, deterministic=True)
                
                # Ejecutar acción
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Actualizar
                state = next_state
                episode_reward += reward
            
            rewards.append(episode_reward)
            print(f"Episodio {episode+1}: Recompensa = {episode_reward:.2f}")
        
        return np.mean(rewards)
    
    def save_model(self, actor_path: str, critic_path: str) -> None:
        """
        Guarda los modelos del actor y crítico.
        
        Parámetros:
        -----------
        actor_path : str
            Ruta para guardar el actor
        critic_path : str
            Ruta para guardar el crítico
        """
        self.model.actor.save_weights(actor_path)
        self.model.critic.save_weights(critic_path)
        print(f"Modelo guardado en {actor_path} y {critic_path}")
    
    def load_model(self, actor_path: str, critic_path: str) -> None:
        """
        Carga los modelos del actor y crítico.
        
        Parámetros:
        -----------
        actor_path : str
            Ruta para cargar el actor
        critic_path : str
            Ruta para cargar el crítico
        """
        # Asegurarse que los modelos están construidos
        dummy_state = np.zeros((1, self.state_dim))
        if self.continuous:
            dummy_state = np.zeros((1, self.state_dim))
            _ = self.model.actor(dummy_state)
            _ = self.model.critic(dummy_state)
            self.model.critic.load_weights(critic_path)
            print(f"Modelo cargado desde {actor_path} y {critic_path}")
    
    def visualize_training(self, history: Dict[str, List[float]], smoothing_window: int = 10) -> None:
        """
        Visualiza la historia de entrenamiento.
        
        Parámetros:
        -----------
        history : Dict[str, List[float]]
            Historia de entrenamiento
        smoothing_window : int, opcional
            Tamaño de ventana para suavizado (default: 10)
        """
        import matplotlib.pyplot as plt
        
        # Constantes para etiquetas
        LABEL_ITERATION = 'Iteración'
        LABEL_ENTROPY = 'Entropía'
        LABEL_LOSS = 'Pérdida'
        
        # Función para suavizar datos
        def smooth(data: List[float], window_size: int) -> np.ndarray:
            """
            Aplica suavizado con media móvil a los datos.
            
            Parámetros:
            -----------
            data : List[float]
                Datos a suavizar
            window_size : int
                Tamaño de la ventana de suavizado
                
            Retorna:
            --------
            np.ndarray
                Datos suavizados
            """
            if len(data) < window_size:
                return np.array(data)
            kernel = np.ones(window_size) / window_size
            return np.convolve(np.array(data), kernel, mode='valid')
        
        # Crear figura con múltiples subplots
        _, axs = plt.subplots(3, 2, figsize=(15, 12))
        
        # 1. Recompensas de episodio
        axs[0, 0].plot(history['mean_episode_rewards'], alpha=0.3, color='blue')
        if len(history['mean_episode_rewards']) > smoothing_window:
            axs[0, 0].plot(
                range(smoothing_window-1, len(history['mean_episode_rewards'])),
                smooth(history['mean_episode_rewards'], smoothing_window),
                color='blue',
                label='Suavizado'
            )
        axs[0, 0].set_title('Recompensa Media por Episodio')
        axs[0, 0].set_xlabel(LABEL_ITERATION)
        axs[0, 0].set_ylabel('Recompensa')
        axs[0, 0].grid(alpha=0.3)
        axs[0, 0].legend()
        axs[0, 0].set_ylabel('Recompensa')
        axs[0, 0].grid(alpha=0.3)
        axs[0, 0].legend()
        
        # 2. Recompensas de evaluación
        if history['evaluation_rewards']:
            eval_interval = len(history['iterations']) // len(history['evaluation_rewards'])
            x_eval = [i * eval_interval for i in range(len(history['evaluation_rewards']))]
            axs[0, 1].plot(x_eval, history['evaluation_rewards'], color='green', marker='o')
            axs[0, 1].plot(x_eval, history['evaluation_rewards'], color='green', marker='o')
            axs[0, 1].set_title('Recompensa de Evaluación')
            axs[0, 1].set_xlabel(LABEL_ITERATION)
            axs[0, 1].set_ylabel('Recompensa Media')
            axs[0, 1].grid(alpha=0.3)
            axs[0, 1].set_xlabel('Iteración')
            axs[0, 1].set_ylabel('Recompensa Media')
            axs[0, 1].grid(alpha=0.3)
        
        # 3. Pérdida de política
        axs[1, 0].plot(history['policy_losses'], alpha=0.3, color='red')
        if len(history['policy_losses']) > smoothing_window:
            axs[1, 0].plot(
                range(smoothing_window-1, len(history['policy_losses'])),
                smooth(history['policy_losses'], smoothing_window),
                color='red',
                label='Suavizado'
            )
        axs[1, 0].set_title('Pérdida de Política')
        axs[1, 0].set_xlabel(LABEL_ITERATION)
        axs[1, 0].set_ylabel(LABEL_LOSS)
        axs[1, 0].grid(alpha=0.3)
        axs[1, 0].legend()
        axs[1, 0].set_ylabel(LABEL_LOSS)
        axs[1, 0].grid(alpha=0.3)
        axs[1, 0].legend()
        
        # 4. Pérdida de valor
        axs[1, 1].plot(history['value_losses'], alpha=0.3, color='purple')
        if len(history['value_losses']) > smoothing_window:
            axs[1, 1].plot(
                range(smoothing_window-1, len(history['value_losses'])),
                smooth(history['value_losses'], smoothing_window),
                color='purple',
                label='Suavizado'
            )
        axs[1, 1].set_title('Pérdida de Valor')
        axs[1, 1].set_xlabel(LABEL_ITERATION)
        axs[1, 1].set_ylabel(LABEL_LOSS)
        axs[1, 1].grid(alpha=0.3)
        axs[1, 1].legend()
        axs[1, 1].set_ylabel(LABEL_LOSS)
        axs[1, 1].grid(alpha=0.3)
        axs[1, 1].legend()
        
        # 5. KL Divergence
        axs[2, 0].plot(history['kl_divergences'], alpha=0.3, color='orange')
        if len(history['kl_divergences']) > smoothing_window:
            axs[2, 0].plot(
                range(smoothing_window-1, len(history['kl_divergences'])),
                smooth(history['kl_divergences'], smoothing_window),
                color='orange',
                label='Suavizado'
            )
        axs[2, 0].set_title('KL Divergence')
        axs[2, 0].set_xlabel(LABEL_ITERATION)
        axs[2, 0].set_ylabel('KL')
        axs[2, 0].grid(alpha=0.3)
        axs[2, 0].legend()
        axs[2, 0].set_ylabel('KL')
        axs[2, 0].grid(alpha=0.3)
        axs[2, 0].legend()
        
        # 6. Entropía
        axs[2, 1].plot(history['entropies'], alpha=0.3, color='brown')
        if len(history['entropies']) > smoothing_window:
            axs[2, 1].plot(
                range(smoothing_window-1, len(history['entropies'])),
                smooth(history['entropies'], smoothing_window),
                color='brown',
                label='Suavizado'
            )
        axs[2, 1].set_title(LABEL_ENTROPY)
        axs[2, 1].set_xlabel(LABEL_ITERATION)
        axs[2, 1].set_ylabel(LABEL_ENTROPY)
        axs[2, 1].grid(alpha=0.3)
        axs[2, 1].legend()
        axs[2, 1].set_ylabel(LABEL_ENTROPY)
        axs[2, 1].grid(alpha=0.3)
        axs[2, 1].legend()
        
        plt.tight_layout()
        plt.show()