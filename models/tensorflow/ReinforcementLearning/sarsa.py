import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from typing import Dict, List, Tuple, Optional, Union

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from models.config import SARSA_CONFIG

class SARSA:
    """
    Implementación del algoritmo SARSA (State-Action-Reward-State-Action).
    
    SARSA es un algoritmo de aprendizaje por refuerzo on-policy que actualiza
    los valores Q basándose en la política actual, incluyendo la exploración.
    """
    
    def __init__(self, env, config=None):
        """
        Inicializa el agente SARSA.
        
        Args:
            env: Entorno de OpenAI Gym o compatible
            config: Configuración personalizada (opcional)
        """
        self.env = env
        self.config = config or SARSA_CONFIG
        
        # Inicializar parámetros básicos
        self._init_learning_params()
        self._validate_action_space()
        
        # Configurar espacio de estados y tabla Q
        self.discrete_state_space = hasattr(env.observation_space, 'n')
        
        if self.discrete_state_space:
            self._setup_discrete_state_space()
        else:
            self._setup_continuous_state_space()
        
        # Métricas para seguimiento del entrenamiento
        self.episode_rewards = []
        self.episode_lengths = []
        self.epsilon_history = []
        
    def _init_learning_params(self):
        """Inicializa los parámetros de aprendizaje desde la configuración"""
        self.alpha = self.config['learning_rate']
        self.gamma = self.config['gamma']
        self.epsilon = self.config['epsilon_start']
        self.epsilon_min = self.config['epsilon_end']
        self.epsilon_decay = self.config['epsilon_decay']
        self.decay_type = self.config['epsilon_decay_type']
    
    def _validate_action_space(self):
        """Valida que el espacio de acción sea compatible con SARSA"""
        if not hasattr(self.env.action_space, 'n'):
            raise ValueError("SARSA requiere un espacio de acción discreto")
        self.action_space_size = self.env.action_space.n
    
    def _setup_discrete_state_space(self):
        """Configura SARSA para un espacio de estados discreto"""
        self.state_space_size = self.env.observation_space.n
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))
        
        if self.config['optimistic_initialization']:
            self.q_table += self.config['optimistic_initial_value']
        
        # Función para obtener índice de estado
        self.get_state_index = lambda state: state
    
    def _setup_continuous_state_space(self):
        """Configura SARSA para un espacio de estados continuo con discretización"""
        self.state_dim = self.env.observation_space.shape[0]
        self.bins_per_dim = self.config['bins']
        
        # Configurar límites de estado y bins
        self._setup_state_bounds()
        self._create_discretization_bins()
        
        # Crear y configurar tabla Q
        q_shape = tuple([self.bins_per_dim] * self.state_dim + [self.action_space_size])
        self.q_table = np.zeros(q_shape)
        
        if self.config['optimistic_initialization']:
            self.q_table += self.config['optimistic_initial_value']
    
    def _setup_state_bounds(self):
        """Determina los límites para cada dimensión del espacio de estados"""
        if self.config['state_bounds'] is None:
            self.state_bounds = self._get_default_state_bounds()
        else:
            self.state_bounds = self.config['state_bounds']
    
    def _get_default_state_bounds(self):
        """Calcula límites predeterminados para cada dimensión del estado"""
        bounds = []
        for i in range(self.state_dim):
            low = self.env.observation_space.low[i]
            high = self.env.observation_space.high[i]
            
            # Manejar valores infinitos
            if low == float("-inf") or low < -1e6:
                low = -10.0
            if high == float("inf") or high > 1e6:
                high = 10.0
                
            bounds.append((low, high))
        return bounds
    
    def _create_discretization_bins(self):
        """Crea bins para discretización del espacio de estados continuo"""
        self.discrete_states = []
        for low, high in self.state_bounds:
            self.discrete_states.append(np.linspace(low, high, self.bins_per_dim + 1)[1:-1])
    
    def discretize_state(self, state):
        """
        Discretiza un estado continuo.
        
        Args:
            state: Estado continuo del entorno
            
        Returns:
            Tupla con índices discretizados
        """
        if self.discrete_state_space:
            return state
        
        discrete_state = []
        for i, val in enumerate(state):
            # Limitar val al rango definido
            low, high = self.state_bounds[i]
            val = max(low, min(val, high))
            
            # Encontrar el bin correspondiente
            bins = self.discrete_states[i]
            digitized = np.digitize(val, bins)
            discrete_state.append(digitized)
        
        return tuple(discrete_state)
    
    def get_action(self, state, explore=True):
        """
        Selecciona una acción usando política epsilon-greedy.
        
        Args:
            state: Estado actual
            explore: Si debe explorar (True) o ser greedy (False)
            
        Returns:
            Acción seleccionada
        """
        discrete_state = self.discretize_state(state)
        
        # Exploración con probabilidad epsilon
        if explore and self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.action_space_size)
        
        # Explotación: elegir acción con mayor valor Q
        return np.argmax(self.q_table[discrete_state])
    
    def update_q_value(self, state, action, reward, next_state, next_action):
        """
        Actualiza un valor Q usando la regla de actualización SARSA.
        
        Args:
            state: Estado actual
            action: Acción tomada
            reward: Recompensa recibida
            next_state: Siguiente estado
            next_action: Siguiente acción (según política actual)
        """
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        
        # Valor Q actual
        current_q = self.q_table[discrete_state][action]
        
        # Valor Q del siguiente estado-acción
        next_q = self.q_table[discrete_next_state][next_action]
        
        # Actualizar Q usando la regla SARSA
        # Q(s,a) = Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
        self.q_table[discrete_state][action] = new_q
    
    def decay_epsilon(self, episode=None):
        """
        Actualiza epsilon según el esquema de decaimiento configurado.
        
        Args:
            episode: Número de episodio actual (para decaimiento lineal)
        """
        if self.decay_type == 'exponential':
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        elif self.decay_type == 'linear':
            # Requiere conocer el número total de episodios para el decaimiento lineal
            if episode is not None and self.config['episodes'] > 0:
                self.epsilon = max(
                    self.epsilon_min,
                    self.epsilon_min + (self.config['epsilon_start'] - self.epsilon_min) * 
                    (1 - episode / self.config['episodes'])
                )
    
    def train(self, episodes=None, max_steps=None, render=False):
        """
        Entrena el agente SARSA.
        
        Args:
            episodes: Número de episodios de entrenamiento
            max_steps: Límite de pasos por episodio
            render: Si renderizar el entorno durante el entrenamiento
            
        Returns:
            Diccionario con métricas de entrenamiento
        """
        episodes = episodes or self.config['episodes']
        max_steps = max_steps or self.config['max_steps']
        
        start_time = time.time()
        
        for episode in range(episodes):
            state, _ = self.env.reset()
            action = self.get_action(state)  # Seleccionar primera acción
            
            episode_reward = 0
            episode_steps = 0
            
            # Interactuar con el entorno hasta terminar episodio
            for _ in range(max_steps):
                if render:
                    self.env.render()
                
                # Ejecutar acción y observar resultado
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Acumular recompensa
                episode_reward += reward
                episode_steps += 1
                
                # Seleccionar siguiente acción según política actual
                next_action = self.get_action(next_state)
                
                # Actualizar valor Q
                self.update_q_value(state, action, reward, next_state, next_action)
                
                # Actualizar estado y acción para la siguiente iteración
                state = next_state
                action = next_action
                
                if done:
                    break
            
            # Recopilar métricas del episodio
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_steps)
            self.epsilon_history.append(self.epsilon)
            
            # Actualizar epsilon para el siguiente episodio
            self.decay_epsilon(episode)
            
            # Mostrar progreso periódicamente
            if (episode + 1) % self.config['log_interval'] == 0:
                avg_reward = np.mean(self.episode_rewards[-self.config['log_interval']:])
                avg_steps = np.mean(self.episode_lengths[-self.config['log_interval']:])
                elapsed = time.time() - start_time
                print(f"Episodio {episode+1}/{episodes} - Recompensa: {avg_reward:.2f}, "
                      f"Pasos: {avg_steps:.2f}, Epsilon: {self.epsilon:.4f}, "
                      f"Tiempo: {elapsed:.2f}s")
        
        print("Entrenamiento completado!")
        
        # Retornar historial de entrenamiento
        return {
            'rewards': self.episode_rewards,
            'lengths': self.episode_lengths,
            'epsilons': self.epsilon_history
        }
    
    def evaluate(self, episodes=10, render=True, verbose=True):
        """
        Evalúa la política aprendida.
        
        Args:
            episodes: Número de episodios para evaluación
            render: Si renderizar el entorno durante evaluación
            verbose: Si mostrar resultados detallados
            
        Returns:
            Recompensa promedio obtenida
        """
        rewards = []
        steps = []
        
        for episode in range(episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False
            
            while not done and episode_steps < self.config['max_steps']:
                if render:
                    self.env.render()
                
                # Seleccionar acción sin exploración
                action = self.get_action(state, explore=False)
                
                # Ejecutar acción
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Actualizar contadores
                episode_reward += reward
                episode_steps += 1
                state = next_state
            
            rewards.append(episode_reward)
            steps.append(episode_steps)
            
            if verbose:
                print(f"Episodio {episode+1}: Recompensa = {episode_reward}, Pasos = {episode_steps}")
        
        avg_reward = np.mean(rewards)
        avg_steps = np.mean(steps)
        
        print(f"Evaluación completada - Recompensa Media: {avg_reward:.2f}, Pasos Medios: {avg_steps:.2f}")
        
        return avg_reward
    
    def save(self, filepath):
        """
        Guarda la tabla Q y otra información del modelo.
        
        Args:
            filepath: Ruta donde guardar el modelo
        """
        data = {
            'q_table': self.q_table,
            'discrete_states': self.discrete_states if not self.discrete_state_space else None,
            'state_bounds': self.state_bounds if not self.discrete_state_space else None,
            'discrete_state_space': self.discrete_state_space,
            'config': self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Modelo guardado en {filepath}")
    
    def load(self, filepath):
        """
        Carga la tabla Q y otra información del modelo.
        
        Args:
            filepath: Ruta desde donde cargar el modelo
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.q_table = data['q_table']
        self.discrete_state_space = data['discrete_state_space']
        
        if not self.discrete_state_space:
            self.discrete_states = data['discrete_states']
            self.state_bounds = data['state_bounds']
        
        self.config = data['config']
        self.alpha = self.config['learning_rate']
        self.gamma = self.config['gamma']
        
        print(f"Modelo cargado desde {filepath}")
    
    def visualize_training(self, training_history=None, smoothing_window=None):
        """
        Visualiza las métricas de entrenamiento.
        
        Args:
            training_history: Historia de entrenamiento (opcional)
            smoothing_window: Tamaño de ventana para suavizado
        """
        if training_history is None:
            # Usar historial almacenado internamente
            rewards = self.episode_rewards
            lengths = self.episode_lengths
            epsilons = self.epsilon_history
        else:
            # Usar historial proporcionado
            rewards = training_history['rewards']
            lengths = training_history['lengths']
            epsilons = training_history['epsilons']
        
        smoothing_window = smoothing_window or self.config['smoothing_window']
        
        # Función para suavizar datos
        def smooth(data, window_size):
            if len(data) < window_size:
                return data
            kernel = np.ones(window_size) / window_size
            return np.convolve(data, kernel, mode='valid')
        
        # Crear figura con tres subplots
        _, axs = plt.subplots(3, 1, figsize=(10, 12))
        
        # 1. Gráfico de recompensas
        axs[0].plot(rewards, alpha=0.3, color='blue', label='Raw')
        if len(rewards) > smoothing_window:
            smoothed_rewards = smooth(rewards, smoothing_window)
            axs[0].plot(
                range(smoothing_window-1, len(rewards)),
                smoothed_rewards,
                color='blue',
                label=f'Suavizado (window={smoothing_window})'
            )
        axs[0].set_title('Recompensas por Episodio')
        axs[0].set_xlabel('Episodio')
        axs[0].set_ylabel('Recompensa')
        axs[0].legend()
        axs[0].grid(alpha=0.3)
        
        # 2. Gráfico de longitud de episodios
        axs[1].plot(lengths, alpha=0.3, color='green', label='Raw')
        if len(lengths) > smoothing_window:
            smoothed_lengths = smooth(lengths, smoothing_window)
            axs[1].plot(
                range(smoothing_window-1, len(lengths)),
                smoothed_lengths,
                color='green',
                label=f'Suavizado (window={smoothing_window})'
            )
        axs[1].set_title('Longitud de Episodios')
        axs[1].set_xlabel('Episodio')
        axs[1].set_ylabel('Pasos')
        axs[1].legend()
        axs[1].grid(alpha=0.3)
        
        # 3. Gráfico de epsilon
        axs[2].plot(epsilons, color='red')
        axs[2].set_title('Valor de Epsilon (Exploración)')
        axs[2].set_xlabel('Episodio')
        axs[2].set_ylabel('Epsilon')
        axs[2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_policy(self, state_dims=(0, 1), resolution=100):
        """
        Visualiza la política aprendida para entornos con estados continuos.
        
        Args:
            state_dims: Tupla con las dos dimensiones del estado a visualizar
            resolution: Resolución de la visualización
        """
        if self.discrete_state_space:
            print("Visualización de política no disponible para espacios de estado discretos nativos")
            return
        
        if self.state_dim < 2:
            print("Visualización requiere al menos 2 dimensiones de estado")
            return
        
        # Crear malla para visualización
        dim1, dim2 = state_dims
        x_range = np.linspace(self.state_bounds[dim1][0], self.state_bounds[dim1][1], resolution)
        y_range = np.linspace(self.state_bounds[dim2][0], self.state_bounds[dim2][1], resolution)
        X, Y = np.meshgrid(x_range, y_range)
        
        # Calcular acciones para cada punto de la malla
        policy = np.zeros(X.shape, dtype=int)
        
        # Estado base (valores medios para dimensiones no visualizadas)
        base_state = []
        for i in range(self.state_dim):
            if i == dim1 or i == dim2:
                base_state.append(0)  # Placeholder
            else:
                # Usar punto medio para dimensiones no visualizadas
                base_state.append((self.state_bounds[i][0] + self.state_bounds[i][1]) / 2)
        
        for i in range(resolution):
            for j in range(resolution):
                # Crear estado para este punto
                state = base_state.copy()
                state[dim1] = X[i, j]
                state[dim2] = Y[i, j]
                
                # Obtener acción según política actual
                action = self.get_action(state, explore=False)
                policy[i, j] = action
        
        # Visualizar política
        plt.figure(figsize=(10, 8))
        policy_plot = plt.pcolormesh(X, Y, policy, cmap='viridis', alpha=0.7, shading='auto')
        plt.colorbar(policy_plot, label='Acción')
        plt.title('Visualización de Política')
        plt.xlabel(f'Dimensión de Estado {dim1}')
        plt.ylabel(f'Dimensión de Estado {dim2}')
        plt.grid(alpha=0.2)
        plt.show()
    
    def visualize_value_function(self, state_dims=(0, 1), resolution=100):
        """
        Visualiza la función de valor aprendida para entornos con estados continuos.
        
        Args:
            state_dims: Tupla con las dos dimensiones del estado a visualizar
            resolution: Resolución de la visualización
        """
        if self.discrete_state_space:
            print("Visualización de valores no disponible para espacios de estado discretos nativos")
            return
        
        if self.state_dim < 2:
            print("Visualización requiere al menos 2 dimensiones de estado")
            return
        
        # Crear malla para visualización
        dim1, dim2 = state_dims
        x_range = np.linspace(self.state_bounds[dim1][0], self.state_bounds[dim1][1], resolution)
        y_range = np.linspace(self.state_bounds[dim2][0], self.state_bounds[dim2][1], resolution)
        X, Y = np.meshgrid(x_range, y_range)
        
        # Calcular valores para cada punto de la malla
        values = np.zeros(X.shape)
        
        # Estado base (valores medios para dimensiones no visualizadas)
        base_state = []
        for i in range(self.state_dim):
            if i == dim1 or i == dim2:
                base_state.append(0)  # Placeholder
            else:
                # Usar punto medio para dimensiones no visualizadas
                base_state.append((self.state_bounds[i][0] + self.state_bounds[i][1]) / 2)
        
        for i in range(resolution):
            for j in range(resolution):
                # Crear estado para este punto
                state = base_state.copy()
                state[dim1] = X[i, j]
                state[dim2] = Y[i, j]
                
                # Obtener valor máximo para este estado
                discrete_state = self.discretize_state(state)
                values[i, j] = np.max(self.q_table[discrete_state])
        
        # Visualizar valores
        plt.figure(figsize=(10, 8))
        value_plot = plt.pcolormesh(X, Y, values, cmap='plasma', alpha=0.7, shading='auto')
        plt.colorbar(value_plot, label='Valor')
        plt.title('Visualización de Función de Valor')
        plt.xlabel(f'Dimensión de Estado {dim1}')
        plt.ylabel(f'Dimensión de Estado {dim2}')
        plt.grid(alpha=0.2)
        plt.show()