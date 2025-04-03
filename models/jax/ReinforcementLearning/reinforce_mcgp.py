import os, sys
import flax
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, NamedTuple
from functools import partial
from flax import linen as nn
from flax.training import train_state
import optax

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from models.config import REINFORCE_CONFIG


class PolicyNetworkState(NamedTuple):
    """Estado para la red neuronal de política"""
    params: Dict
    rng_key: jnp.ndarray
    

class PolicyNetwork(nn.Module):
    """
    Red neuronal de política para REINFORCE.
    
    Esta red mapea estados a distribuciones de probabilidad sobre acciones.
    """
    state_dim: int
    action_dim: int
    hidden_units: Tuple[int, ...]
    continuous: bool
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Ejecuta la red para obtener los parámetros de la distribución de política.
        
        Parámetros:
        -----------
        x : jnp.ndarray
            Tensor de estados de entrada
        training : bool, opcional
            Si está en modo entrenamiento (default: False)
        
        Retorna:
        --------
        Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]
            Para espacios discretos: logits
            Para espacios continuos: (mu, log_sigma)
        """
        # Capas ocultas
        for i, units in enumerate(self.hidden_units):
            x = nn.Dense(features=units, name=f'hidden_{i}')(x)
            x = nn.LayerNorm(epsilon=REINFORCE_CONFIG['epsilon'], name=f'ln_{i}')(x)
            x = nn.relu(x)
            if REINFORCE_CONFIG['dropout_rate'] > 0 and training:
                x = nn.Dropout(
                    rate=REINFORCE_CONFIG['dropout_rate'], 
                    deterministic=not training,
                    name=f'dropout_{i}'
                )(x)
        
        # Capa de salida: depende de si el espacio de acciones es continuo o discreto
        if self.continuous:
            # Para espacios continuos: política gaussiana
            mu = nn.Dense(features=self.action_dim, name='mu')(x)
            mu = nn.tanh(mu)
            
            log_sigma = nn.Dense(features=self.action_dim, name='log_sigma')(x)
            # Limitar el rango de log_sigma para estabilidad
            log_sigma = jnp.clip(log_sigma, -20, 2)
            
            return mu, log_sigma
        else:
            # Para espacios discretos: política categórica
            logits = nn.Dense(features=self.action_dim, name='logits')(x)
            return logits


class REINFORCEState(NamedTuple):
    """Estado para el algoritmo REINFORCE"""
    policy_state: train_state.TrainState
    value_state: Optional[train_state.TrainState] = None
    rng_key: jnp.ndarray = None


class REINFORCE:
    """
    Implementación del algoritmo REINFORCE (Monte Carlo Policy Gradient).
    
    REINFORCE utiliza retornos Monte Carlo completos para actualizar la política,
    haciéndolo un algoritmo simple pero efectivo para aprendizaje de políticas.
    """
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int,
        continuous: bool = False,
        learning_rate: float = REINFORCE_CONFIG['learning_rate'],
        gamma: float = REINFORCE_CONFIG['gamma'],
        hidden_units: Optional[List[int]] = None,
        baseline: bool = REINFORCE_CONFIG['use_baseline'],
        entropy_coef: float = REINFORCE_CONFIG['entropy_coef'],
        seed: int = 42
    ) -> None:
        """
        Inicializa el agente REINFORCE.
        
        Parámetros:
        -----------
        state_dim : int
            Dimensión del espacio de estados
        action_dim : int
            Dimensión del espacio de acciones
        continuous : bool, opcional
            Si el espacio de acciones es continuo (default: False)
        learning_rate : float, opcional
            Tasa de aprendizaje (default: REINFORCE_CONFIG['learning_rate'])
        gamma : float, opcional
            Factor de descuento (default: REINFORCE_CONFIG['gamma'])
        hidden_units : Optional[List[int]], opcional
            Lista con unidades en capas ocultas (default: None)
        baseline : bool, opcional
            Si usar baseline para reducir varianza (default: REINFORCE_CONFIG['use_baseline'])
        entropy_coef : float, opcional
            Coeficiente para regularización por entropía (default: REINFORCE_CONFIG['entropy_coef'])
        seed : int, opcional
            Semilla para reproducibilidad (default: 42)
        """
        # Configurar semillas para reproducibilidad
        self.rng_key = jax.random.PRNGKey(seed)
        
        # Parámetros básicos
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous
        self.gamma = gamma
        self.use_baseline = baseline
        self.entropy_coef = entropy_coef
        
        # Valores predeterminados para capas ocultas
        if hidden_units is None:
            self.hidden_units = tuple(REINFORCE_CONFIG['hidden_units'])
        else:
            self.hidden_units = tuple(hidden_units)
        
        # Crear red de política con Flax
        self.policy_model = PolicyNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_units=self.hidden_units,
            continuous=continuous
        )
        
        # Inicializar parámetros del modelo
        self.rng_key, init_key = jax.random.split(self.rng_key)
        dummy_input = jnp.ones((1, state_dim))
        policy_params = self.policy_model.init(init_key, dummy_input)
        
        # Crear optimizador
        policy_tx = optax.adam(learning_rate=learning_rate)
        self.policy_state = train_state.TrainState.create(
            apply_fn=self.policy_model.apply,
            params=policy_params,
            tx=policy_tx
        )
        
        # Red de valor (baseline) opcional para reducir varianza
        self.value_state = None
        if self.use_baseline:
            self.value_model = self._create_value_network()
            self.rng_key, init_key = jax.random.split(self.rng_key)
            value_params = self.value_model.init(init_key, dummy_input)
            value_tx = optax.adam(learning_rate=learning_rate)
            self.value_state = train_state.TrainState.create(
                apply_fn=self.value_model.apply,
                params=value_params,
                tx=value_tx
            )
        
        # Estado del agente
        self.state = REINFORCEState(
            policy_state=self.policy_state,
            value_state=self.value_state,
            rng_key=self.rng_key
        )
        
        # Compilar funciones para rendimiento óptimo
        self._get_action_discrete = jax.jit(self._get_action_discrete)
        self._get_action_continuous = jax.jit(self._get_action_continuous)
        self._get_log_prob_discrete = jax.jit(self._get_log_prob_discrete)
        self._get_log_prob_continuous = jax.jit(self._get_log_prob_continuous)
        self._get_entropy_discrete = jax.jit(self._get_entropy_discrete)
        self._get_entropy_continuous = jax.jit(self._get_entropy_continuous)
        self._policy_loss_fn = jax.jit(self._policy_loss_fn)
        if self.use_baseline:
            self._value_loss_fn = jax.jit(self._value_loss_fn)
        
        # Métricas
        self.policy_loss_metric = 0.0
        self.entropy_metric = 0.0
        self.baseline_loss_metric = 0.0
        self.returns_metric = 0.0
    
    def _create_value_network(self) -> nn.Module:
        """
        Crea una red neuronal para estimar el valor de estado (baseline).
        
        Retorna:
        --------
        nn.Module
            Módulo de red de valor
        """
        class ValueNetwork(nn.Module):
            """Red neuronal para estimar valores de estado."""
            hidden_units: Tuple[int, ...]
            
            @nn.compact
            def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
                # Capas ocultas
                for i, units in enumerate(self.hidden_units):
                    x = nn.Dense(features=units, name=f'value_hidden_{i}')(x)
                    x = nn.LayerNorm(epsilon=REINFORCE_CONFIG['epsilon'], name=f'value_ln_{i}')(x)
                    x = nn.relu(x)
                    if REINFORCE_CONFIG['dropout_rate'] > 0 and training:
                        x = nn.Dropout(
                            rate=REINFORCE_CONFIG['dropout_rate'], 
                            deterministic=not training,
                            name=f'value_dropout_{i}'
                        )(x)
                
                # Capa de salida: un solo valor
                x = nn.Dense(features=1, name='value')(x)
                return x
        
        return ValueNetwork(hidden_units=self.hidden_units)
    
    def _get_action_discrete(
        self, 
        params: Dict, 
        state: jnp.ndarray, 
        rng_key: jnp.ndarray, 
        deterministic: bool
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Obtiene una acción para espacio discreto.
        
        Parámetros:
        -----------
        params : Dict
            Parámetros de la red de política
        state : jnp.ndarray
            Estado actual
        rng_key : jnp.ndarray
            Llave para generación de números aleatorios
        deterministic : bool
            Si seleccionar la acción determinísticamente
            
        Retorna:
        --------
        Tuple[jnp.ndarray, jnp.ndarray]
            Acción seleccionada y nueva llave aleatoria
        """
        logits = self.policy_model.apply(params, state[None])
        
        if deterministic:
            action = jnp.argmax(logits, axis=-1)[0]
        else:
            rng_key, subkey = jax.random.split(rng_key)
            action = jax.random.categorical(subkey, logits)[0]
        
        return action, rng_key
    
    def _get_action_continuous(
        self, 
        params: Dict, 
        state: jnp.ndarray, 
        rng_key: jnp.ndarray, 
        deterministic: bool
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Obtiene una acción para espacio continuo.
        
        Parámetros:
        -----------
        params : Dict
            Parámetros de la red de política
        state : jnp.ndarray
            Estado actual
        rng_key : jnp.ndarray
            Llave para generación de números aleatorios
        deterministic : bool
            Si seleccionar la acción determinísticamente
            
        Retorna:
        --------
        Tuple[jnp.ndarray, jnp.ndarray]
            Acción seleccionada y nueva llave aleatoria
        """
        mu, log_sigma = self.policy_model.apply(params, state[None])
        
        if deterministic:
            action = mu[0]
        else:
            rng_key, subkey = jax.random.split(rng_key)
            sigma = jnp.exp(log_sigma)
            noise = jax.random.normal(subkey, shape=mu.shape)
            action = mu + sigma * noise
            action = action[0]
        
        return action, rng_key
    
    def get_action(
        self, 
        state: np.ndarray, 
        deterministic: bool = False
    ) -> Union[np.ndarray, int]:
        """
        Obtiene una acción según la política actual.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado actual
        deterministic : bool, opcional
            Si se usa la acción determinística o se muestrea (default: False)

        Retorna:
        --------
        Union[np.ndarray, int]
            La acción seleccionada
        """
        state = jnp.asarray(state)
        
        # Seleccionar acción según el tipo de espacio
        if self.continuous:
            action, new_key = self._get_action_continuous(
                self.state.policy_state.params, 
                state, 
                self.state.rng_key, 
                deterministic
            )
        else:
            action, new_key = self._get_action_discrete(
                self.state.policy_state.params, 
                state, 
                self.state.rng_key, 
                deterministic
            )
        
        # Actualizar llave de aleatoriedad
        self.state = self.state._replace(rng_key=new_key)
        
        return np.asarray(action)
    
    def _get_log_prob_discrete(
        self, 
        params: Dict, 
        states: jnp.ndarray, 
        actions: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Calcula log-probabilidades para acciones discretas.
        
        Parámetros:
        -----------
        params : Dict
            Parámetros de la red de política
        states : jnp.ndarray
            Estados observados
        actions : jnp.ndarray
            Acciones tomadas
            
        Retorna:
        --------
        jnp.ndarray
            Log-probabilidades de las acciones
        """
        logits = self.policy_model.apply(params, states)
        action_masks = jax.nn.one_hot(actions, self.action_dim)
        log_probs = jnp.sum(action_masks * jax.nn.log_softmax(logits), axis=1)
        return log_probs
    
    def _get_log_prob_continuous(
        self, 
        params: Dict, 
        states: jnp.ndarray, 
        actions: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Calcula log-probabilidades para acciones continuas.
        
        Parámetros:
        -----------
        params : Dict
            Parámetros de la red de política
        states : jnp.ndarray
            Estados observados
        actions : jnp.ndarray
            Acciones tomadas
            
        Retorna:
        --------
        jnp.ndarray
            Log-probabilidades de las acciones
        """
        mu, log_sigma = self.policy_model.apply(params, states)
        sigma = jnp.exp(log_sigma)
        
        # Log-prob para distribución gaussiana
        log_probs = -0.5 * (
            jnp.sum(
                jnp.square((actions - mu) / sigma) + 
                2 * log_sigma + 
                jnp.log(2.0 * np.pi),
                axis=1
            )
        )
        return log_probs
    
    def get_log_prob(self, states: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        """
        Calcula el logaritmo de la probabilidad de acciones dadas.
        
        Parámetros:
        -----------
        states : jnp.ndarray
            Estados observados
        actions : jnp.ndarray
            Acciones tomadas
            
        Retorna:
        --------
        jnp.ndarray
            Log-probabilidades de las acciones
        """
        # Seleccionar función según el tipo de espacio
        if self.continuous:
            return self._get_log_prob_continuous(self.state.policy_state.params, states, actions)
        else:
            return self._get_log_prob_discrete(self.state.policy_state.params, states, actions)
    
    def _get_entropy_discrete(self, params: Dict, states: jnp.ndarray) -> jnp.ndarray:
        """
        Calcula la entropía para la política discreta.
        
        Parámetros:
        -----------
        params : Dict
            Parámetros de la red de política
        states : jnp.ndarray
            Estados observados
            
        Retorna:
        --------
        jnp.ndarray
            Entropía de la política
        """
        logits = self.policy_model.apply(params, states)
        probs = jax.nn.softmax(logits)
        entropy = -jnp.sum(probs * jnp.log(probs + 1e-10), axis=1)
        return entropy
    
    def _get_entropy_continuous(self, params: Dict, states: jnp.ndarray) -> jnp.ndarray:
        """
        Calcula la entropía para la política continua.
        
        Parámetros:
        -----------
        params : Dict
            Parámetros de la red de política
        states : jnp.ndarray
            Estados observados
            
        Retorna:
        --------
        jnp.ndarray
            Entropía de la política
        """
        _, log_sigma = self.policy_model.apply(params, states)
        # Entropía de distribución gaussiana: 0.5 * log(2*pi*e*sigma^2)
        entropy = jnp.sum(
            0.5 * jnp.log(2.0 * np.pi * np.e * jnp.exp(2 * log_sigma)),
            axis=1
        )
        return entropy
    
    def get_entropy(self, states: jnp.ndarray) -> jnp.ndarray:
        """
        Calcula la entropía de la política para los estados dados.
        
        Parámetros:
        -----------
        states : jnp.ndarray
            Estados para evaluar
            
        Retorna:
        --------
        jnp.ndarray
            Entropía de la política
        """
        # Seleccionar función según el tipo de espacio
        if self.continuous:
            return self._get_entropy_continuous(self.state.policy_state.params, states)
        else:
            return self._get_entropy_discrete(self.state.policy_state.params, states)
    
    def _policy_loss_fn(
        self, 
        params: Dict, 
        states: jnp.ndarray, 
        actions: jnp.ndarray, 
        returns: jnp.ndarray, 
        values: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Función de pérdida para la red de política.
        
        Parámetros:
        -----------
        params : Dict
            Parámetros de la red de política
        states : jnp.ndarray
            Estados observados
        actions : jnp.ndarray
            Acciones tomadas
        returns : jnp.ndarray
            Retornos calculados
        values : Optional[jnp.ndarray], opcional
            Valores estimados del baseline (default: None)
            
        Retorna:
        --------
        Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]
            Pérdida y métricas auxiliares (pérdida de política, entropía)
        """
        # Calcular log-probabilidades de acciones tomadas
        if self.continuous:
            log_probs = self._get_log_prob_continuous(params, states, actions)
            entropy = self._get_entropy_continuous(params, states)
        else:
            log_probs = self._get_log_prob_discrete(params, states, actions)
            entropy = self._get_entropy_discrete(params, states)
        
        # Si se usa baseline, restar el valor predicho de los retornos
        if values is not None:
            advantages = returns - values
        else:
            advantages = returns
        
        # Calcular pérdida de política (negativa porque queremos maximizar)
        policy_loss = -jnp.mean(log_probs * advantages)
        
        # Calcular entropía media
        mean_entropy = jnp.mean(entropy)
        
        # Pérdida total con regularización de entropía
        loss = policy_loss - self.entropy_coef * mean_entropy
        
        return loss, (policy_loss, mean_entropy)
    
    def _value_loss_fn(self, params: Dict, states: jnp.ndarray, returns: jnp.ndarray) -> jnp.ndarray:
        """
        Función de pérdida para la red de valor.
        
        Parámetros:
        -----------
        params : Dict
            Parámetros de la red de valor
        states : jnp.ndarray
            Estados observados
        returns : jnp.ndarray
            Retornos calculados
            
        Retorna:
        --------
        jnp.ndarray
            Pérdida de la red de valor
        """
        values = jnp.squeeze(self.value_model.apply(params, states))
        loss = jnp.mean(jnp.square(values - returns))
        return loss
    
    def _update_policy(
        self, 
        policy_state: train_state.TrainState, 
        states: jnp.ndarray, 
        actions: jnp.ndarray, 
        returns: jnp.ndarray, 
        values: Optional[jnp.ndarray] = None
    ) -> Tuple[train_state.TrainState, Tuple[float, float]]:
        """
        Actualiza los parámetros de la red de política.
        
        Parámetros:
        -----------
        policy_state : train_state.TrainState
            Estado actual de la red de política
        states : jnp.ndarray
            Estados observados
        actions : jnp.ndarray
            Acciones tomadas
        returns : jnp.ndarray
            Retornos calculados
        values : Optional[jnp.ndarray], opcional
            Valores de estado si se usa baseline (default: None)
            
        Retorna:
        --------
        Tuple[train_state.TrainState, Tuple[float, float]]
            Nuevo estado de la red y métricas (pérdida de política, entropía)
        """
        # Calcular gradientes y actualizar red
        grad_fn = jax.value_and_grad(self._policy_loss_fn, has_aux=True)
        (_, (policy_loss, entropy)), grads = grad_fn(
            policy_state.params, states, actions, returns, values
        )
        
        # Actualizar parámetros
        new_policy_state = policy_state.apply_gradients(grads=grads)
        
        return new_policy_state, (policy_loss, entropy)
    
    def _update_value(
        self, 
        value_state: train_state.TrainState, 
        states: jnp.ndarray, 
        returns: jnp.ndarray
    ) -> Tuple[train_state.TrainState, float]:
        """
        Actualiza los parámetros de la red de valor.
        
        Parámetros:
        -----------
        value_state : train_state.TrainState
            Estado actual de la red de valor
        states : jnp.ndarray
            Estados observados
        returns : jnp.ndarray
            Retornos calculados
            
        Retorna:
        --------
        Tuple[train_state.TrainState, float]
            Nuevo estado de la red y pérdida de valor
        """
        # Calcular gradientes y actualizar red
        grad_fn = jax.value_and_grad(self._value_loss_fn)
        value_loss, grads = grad_fn(value_state.params, states, returns)
        
        # Actualizar parámetros
        new_value_state = value_state.apply_gradients(grads=grads)
        
        return new_value_state, value_loss
    
    def compute_returns(self, rewards: List[float]) -> np.ndarray:
        """
        Calcula los retornos descontados para cada paso de tiempo.
        
        Parámetros:
        -----------
        rewards : List[float]
            Lista de recompensas recibidas
            
        Retorna:
        --------
        np.ndarray
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
    
    def _run_episode(
        self, 
        env: Any, 
        render: bool = False
    ) -> Tuple[List[np.ndarray], List[Union[int, np.ndarray]], List[float], float, int]:
        """
        Ejecuta un episodio completo y recolecta la experiencia.
        
        Parámetros:
        -----------
        env : Any
            Entorno de OpenAI Gym o compatible
        render : bool, opcional
            Si renderizar el entorno durante entrenamiento (default: False)
            
        Retorna:
        --------
        Tuple[List[np.ndarray], List[Union[int, np.ndarray]], List[float], float, int]
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
            action = self.get_action(state)
            
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
    
    def _update_networks(
        self, 
        states: List[np.ndarray], 
        actions: List[Union[int, np.ndarray]], 
        rewards: List[float]
    ) -> Tuple[float, float]:
        """
        Actualiza las redes de política y valor.
        
        Parámetros:
        -----------
        states : List[np.ndarray]
            Lista de estados
        actions : List[Union[int, np.ndarray]]
            Lista de acciones
        rewards : List[float]
            Lista de recompensas
            
        Retorna:
        --------
        Tuple[float, float]
            Tupla con (pérdida_política, entropía)
        """
        # Calcular retornos
        returns = self.compute_returns(rewards)
        
        # Convertir a arrays de JAX
        states = jnp.asarray(states, dtype=jnp.float32)
        if self.continuous:
            actions = jnp.asarray(actions, dtype=jnp.float32)
        else:
            actions = jnp.asarray(actions, dtype=jnp.int32)
        returns = jnp.asarray(returns, dtype=jnp.float32)
        
        # Calcular valores de baseline si se usa
        values = None
        if self.use_baseline:
            values = jnp.squeeze(self.value_model.apply(self.state.value_state.params, states))
            # Actualizar red de valor
            new_value_state, value_loss = self._update_value(
                self.state.value_state, 
                states, 
                returns
            )
            self.baseline_loss_metric = float(value_loss)
        
        # Actualizar red de política
        new_policy_state, (policy_loss, entropy) = self._update_policy(
            self.state.policy_state, 
            states, 
            actions, 
            returns, 
            values
        )
        
        # Actualizar estado del agente
        self.state = REINFORCEState(
            policy_state=new_policy_state,
            value_state=new_value_state if self.use_baseline else None,
            rng_key=self.state.rng_key
        )
        
        # Actualizar métricas
        self.policy_loss_metric = float(policy_loss)
        self.entropy_metric = float(entropy)
        self.returns_metric = float(jnp.mean(returns))
        
        return float(policy_loss), float(entropy)
    
    def _update_history(
        self, 
        history: Dict[str, List[float]], 
        episode_reward: float, 
        episode_length: int
    ) -> None:
        """
        Actualiza la historia de entrenamiento con las métricas actuales.
        
        Parámetros:
        -----------
        history : Dict[str, List[float]]
            Diccionario de historia
        episode_reward : float
            Recompensa total del episodio
        episode_length : int
            Longitud del episodio
        """
        history['episode_rewards'].append(episode_reward)
        history['episode_lengths'].append(episode_length)
        history['policy_losses'].append(self.policy_loss_metric)
        if self.use_baseline:
            history['baseline_losses'].append(self.baseline_loss_metric)
        history['entropies'].append(self.entropy_metric)
        history['mean_returns'].append(self.returns_metric)
    
    def train(
        self, 
        env: Any, 
        episodes: Optional[int] = None, 
        render: bool = False
    ) -> Dict[str, List[float]]:
        """
        Entrena el agente REINFORCE en el entorno dado.
        
        Parámetros:
        -----------
        env : Any
            Entorno de OpenAI Gym o compatible
        episodes : Optional[int], opcional
            Número de episodios de entrenamiento (default: None)
        render : bool, opcional
            Si renderizar el entorno durante entrenamiento (default: False)
            
        Retorna:
        --------
        Dict[str, List[float]]
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
                start_time = time.time()
        
        return history
    
    def evaluate(
        self, 
        env: Any, 
        episodes: int = 10, 
        render: bool = False
    ) -> float:
        """
        Evalúa el agente REINFORCE con su política actual.
        
        Parámetros:
        -----------
        env : Any
            Entorno de OpenAI Gym o compatible
        episodes : int, opcional
            Número de episodios para evaluación (default: 10)
        render : bool, opcional
            Si renderizar el entorno durante evaluación (default: False)
            
        Retorna:
        --------
        float
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
                action = self.get_action(state, deterministic=True)
                
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
    
    def save(self, policy_path: str, baseline_path: Optional[str] = None) -> None:
        """
        Guarda los modelos del agente.
        
        Parámetros:
        -----------
        policy_path : str
            Ruta para guardar la política
        baseline_path : Optional[str], opcional
            Ruta para guardar el baseline (default: None)
        """
        # Guardar política
        with open(policy_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.state.policy_state.params))
        
        # Guardar baseline si existe
        if self.use_baseline and baseline_path:
            with open(baseline_path, 'wb') as f:
                f.write(flax.serialization.to_bytes(self.state.value_state.params))
        
        print(f"Modelo guardado en {policy_path}")
    
    def load(self, policy_path: str, baseline_path: Optional[str] = None) -> None:
        """
        Carga los modelos del agente.
        
        Parámetros:
        -----------
        policy_path : str
            Ruta para cargar la política
        baseline_path : Optional[str], opcional
            Ruta para cargar el baseline (default: None)
        """
        # Cargar política
        with open(policy_path, 'rb') as f:
            policy_params = flax.serialization.from_bytes(
                self.state.policy_state.params,
                f.read()
            )
        
        # Actualizar estado de la política
        policy_state = self.state.policy_state.replace(params=policy_params)
        
        # Cargar baseline si existe
        value_state = self.state.value_state
        if self.use_baseline and baseline_path:
            with open(baseline_path, 'rb') as f:
                value_params = flax.serialization.from_bytes(
                    self.state.value_state.params,
                    f.read()
                )
            value_state = self.state.value_state.replace(params=value_params)
        
        # Actualizar estado del agente
        self.state = REINFORCEState(
            policy_state=policy_state,
            value_state=value_state,
            rng_key=self.state.rng_key
        )
        
        print(f"Modelo cargado desde {policy_path}")
    
    def visualize_training(
        self, 
        history: Optional[Dict[str, List[float]]] = None, 
        smoothing_window: Optional[int] = None
    ) -> None:
        """
        Visualiza las métricas de entrenamiento.
        
        Parámetros:
        -----------
        history : Optional[Dict[str, List[float]]], opcional
            Historia de entrenamiento (default: None)
        smoothing_window : Optional[int], opcional
            Tamaño de ventana para suavizado (default: None)
        """
        if history is None:
            return
        
        if smoothing_window is None:
            smoothing_window = REINFORCE_CONFIG['smoothing_window']
        
        # Función para suavizar datos
        def smooth(data: List[float], window_size: int) -> np.ndarray:
            """Aplica suavizado con media móvil"""
            if len(data) < window_size:
                return data
            kernel = np.ones(window_size) / window_size
            return np.convolve(data, kernel, mode='valid')
        
        # Determinar número de subplots
        n_plots = 4 if self.use_baseline else 3
        _, axs = plt.subplots(n_plots, 1, figsize=(10, 3*n_plots))
        
        # 1. Gráfico de recompensas
        axs[0].plot(history['episode_rewards'], alpha=0.3, color='blue', label='Original')
        if len(history['episode_rewards']) > smoothing_window:
            smoothed_rewards = smooth(history['episode_rewards'], smoothing_window)
            axs[0].plot(range(smoothing_window-1, len(history['episode_rewards'])), 
                      smoothed_rewards, color='blue', 
                      label=f'Suavizado (ventana={smoothing_window})')
        axs[0].set_title('Recompensa por Episodio')
        axs[0].set_xlabel('Episodio')
        axs[0].set_ylabel('Recompensa')
        axs[0].grid(alpha=0.3)
        axs[0].legend()
        
        # 2. Gráfico de longitud de episodios
        axs[1].plot(history['episode_lengths'], alpha=0.3, color='green', label='Original')
        if len(history['episode_lengths']) > smoothing_window:
            smoothed_lengths = smooth(history['episode_lengths'], smoothing_window)
            axs[1].plot(range(smoothing_window-1, len(history['episode_lengths'])), 
                      smoothed_lengths, color='green', 
                      label=f'Suavizado (ventana={smoothing_window})')
        axs[1].set_title('Longitud de Episodios')
        axs[1].set_xlabel('Episodio')
        axs[1].set_ylabel('Pasos')
        axs[1].grid(alpha=0.3)
        axs[1].legend()
        
        # 3. Gráfico de pérdida de política
        axs[2].plot(history['policy_losses'], alpha=0.3, color='red', label='Original')
        if len(history['policy_losses']) > smoothing_window:
            smoothed_losses = smooth(history['policy_losses'], smoothing_window)
            axs[2].plot(range(smoothing_window-1, len(history['policy_losses'])), 
                      smoothed_losses, color='red', 
                      label=f'Suavizado (ventana={smoothing_window})')
        axs[2].set_title('Pérdida de Política')
        axs[2].set_xlabel('Episodio')
        axs[2].set_ylabel('Pérdida')
        axs[2].grid(alpha=0.3)
        axs[2].legend()
        
        # 4. Gráfico de pérdida de baseline (si se usa)
        if self.use_baseline:
            axs[3].plot(history['baseline_losses'], alpha=0.3, color='purple', label='Original')
            if len(history['baseline_losses']) > smoothing_window:
                smoothed_baseline = smooth(history['baseline_losses'], smoothing_window)
                axs[3].plot(range(smoothing_window-1, len(history['baseline_losses'])), 
                          smoothed_baseline, color='purple', 
                          label=f'Suavizado (ventana={smoothing_window})')
            axs[3].set_title('Pérdida de Baseline')
            axs[3].set_xlabel('Episodio')
            axs[3].set_ylabel('Pérdida')
            axs[3].grid(alpha=0.3)
            axs[3].legend()
        
        plt.tight_layout()
        plt.show()