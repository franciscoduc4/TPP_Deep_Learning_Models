import os, sys
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import tensorflow as tf

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from models.config import QLEARNING_CONFIG


class QLearning:
    """
    Implementación de Q-Learning tabular para espacios de estados y acciones discretos.
    
    Este algoritmo aprende una función de valor-acción (Q) a través de experiencias
    recolectadas mediante interacción con el entorno. Utiliza una tabla para almacenar
    los valores Q para cada par estado-acción.
    """
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = QLEARNING_CONFIG['learning_rate'],
        gamma: float = QLEARNING_CONFIG['gamma'],
        epsilon_start: float = QLEARNING_CONFIG['epsilon_start'],
        epsilon_end: float = QLEARNING_CONFIG['epsilon_end'],
        epsilon_decay: float = QLEARNING_CONFIG['epsilon_decay'],
        use_decay_schedule: str = QLEARNING_CONFIG['use_decay_schedule'],
        decay_steps: int = QLEARNING_CONFIG['decay_steps'],
        seed: int = 42
    ) -> None:
        """
        Inicializa el agente Q-Learning.
        
        Parámetros:
        -----------
        n_states : int
            Número de estados en el espacio de estados discreto
        n_actions : int
            Número de acciones en el espacio de acciones discreto
        learning_rate : float, opcional
            Tasa de aprendizaje (alpha) (default: QLEARNING_CONFIG['learning_rate'])
        gamma : float, opcional
            Factor de descuento (default: QLEARNING_CONFIG['gamma'])
        epsilon_start : float, opcional
            Valor inicial de epsilon para política epsilon-greedy (default: QLEARNING_CONFIG['epsilon_start'])
        epsilon_end : float, opcional
            Valor final de epsilon para política epsilon-greedy (default: QLEARNING_CONFIG['epsilon_end'])
        epsilon_decay : float, opcional
            Factor de decaimiento para epsilon (default: QLEARNING_CONFIG['epsilon_decay'])
        use_decay_schedule : str, opcional
            Tipo de decaimiento ('linear', 'exponential', o None) (default: QLEARNING_CONFIG['use_decay_schedule'])
        decay_steps : int, opcional
            Número de pasos para decaer epsilon (si se usa decay schedule) (default: QLEARNING_CONFIG['decay_steps'])
        seed : int, opcional
            Semilla para reproducibilidad (default: 42)
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.use_decay_schedule = use_decay_schedule
        self.decay_steps = decay_steps
        
        # Configurar generador de números aleatorios con semilla fija
        self.rng = np.random.Generator(np.random.PCG64(seed))
        
        # Inicializar la tabla Q con valores optimistas o ceros
        if QLEARNING_CONFIG['optimistic_init']:
            self.q_table = np.ones((n_states, n_actions)) * QLEARNING_CONFIG['optimistic_value']
        else:
            self.q_table = np.zeros((n_states, n_actions))
        
        # Contador total de pasos
        self.total_steps = 0
        
        # Métricas
        self.rewards_history = []
    
    def get_action(self, state: int) -> int:
        """
        Selecciona una acción usando la política epsilon-greedy.
        
        Parámetros:
        -----------
        state : int
            Estado actual
            
        Retorna:
        --------
        int
            Acción seleccionada
        """
        if self.rng.random() < self.epsilon:
            # Explorar: acción aleatoria
            return self.rng.integers(0, self.n_actions)
        else:
            # Explotar: mejor acción según la tabla Q
            return np.argmax(self.q_table[state])
    
    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        """
        Actualiza la tabla Q usando la regla de Q-Learning.
        
        Parámetros:
        -----------
        state : int
            Estado actual
        action : int
            Acción tomada
        reward : float
            Recompensa recibida
        next_state : int
            Estado siguiente
        done : bool
            Si el episodio ha terminado
            
        Retorna:
        --------
        float
            TD-error calculado
        """
        # Calcular valor Q objetivo con Q-Learning (off-policy TD control)
        if done:
            # Si es estado terminal, no hay recompensa futura
            target_q = reward
        else:
            # Q-Learning: seleccionar máxima acción en estado siguiente (greedy)
            target_q = reward + self.gamma * np.max(self.q_table[next_state])
        
        # Valor Q actual
        current_q = self.q_table[state, action]
        
        # Calcular TD error
        td_error = target_q - current_q
        
        # Actualizar valor Q
        self.q_table[state, action] += self.learning_rate * td_error
        
        return td_error
    
    def update_epsilon(self, step: Optional[int] = None) -> None:
        """
        Actualiza el valor de epsilon según la política de decaimiento.
        
        Parámetros:
        -----------
        step : Optional[int], opcional
            Paso actual para decaimiento programado (default: None)
        """
        if self.use_decay_schedule == 'linear':
            # Decaimiento lineal
            if step is not None:
                fraction = min(1.0, step / self.decay_steps)
                self.epsilon = self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)
            else:
                self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        elif self.use_decay_schedule == 'exponential':
            # Decaimiento exponencial
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        # Si no hay decay schedule, epsilon se mantiene constante
    
    def _run_episode(self, env: Any, max_steps: int, render: bool) -> Tuple[float, int]:
        """
        Ejecuta un episodio de entrenamiento.
        
        Parámetros:
        -----------
        env : Any
            Entorno compatible con OpenAI Gym
        max_steps : int
            Máximo número de pasos por episodio
        render : bool
            Si renderizar el entorno durante el entrenamiento
            
        Retorna:
        --------
        Tuple[float, int]
            Recompensa total y número de pasos del episodio
        """
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        for _ in range(max_steps):
            if render:
                env.render()
            
            # Seleccionar y ejecutar acción
            action = self.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Actualizar tabla Q
            self.update(state, action, reward, next_state, done)
            
            # Actualizar contadores y estadísticas
            state = next_state
            episode_reward += reward
            steps += 1
            self.total_steps += 1
            
            # Actualizar epsilon por paso si corresponde
            if self.use_decay_schedule:
                self.update_epsilon(self.total_steps)
            
            if done:
                break
                
        return episode_reward, steps
    
    def _update_history(
        self, 
        history: Dict[str, List[float]], 
        episode_reward: float, 
        steps: int, 
        episode_rewards_window: List[float], 
        log_interval: int
    ) -> float:
        """
        Actualiza el historial de entrenamiento con los resultados del episodio.
        
        Parámetros:
        -----------
        history : Dict[str, List[float]]
            Historial de entrenamiento
        episode_reward : float
            Recompensa del episodio
        steps : int
            Pasos del episodio
        episode_rewards_window : List[float]
            Ventana de recompensas recientes
        log_interval : int
            Intervalo para el cálculo de promedio móvil
            
        Retorna:
        --------
        float
            Recompensa promedio actual
        """
        history['episode_rewards'].append(episode_reward)
        history['episode_lengths'].append(steps)
        history['epsilons'].append(self.epsilon)
        
        # Mantener una ventana de recompensas para promedio móvil
        episode_rewards_window.append(episode_reward)
        if len(episode_rewards_window) > log_interval:
            episode_rewards_window.pop(0)
        
        avg_reward = np.mean(episode_rewards_window)
        history['avg_rewards'].append(avg_reward)
        
        return avg_reward
    
    def train(
        self, 
        env: Any, 
        episodes: int = QLEARNING_CONFIG['episodes'],
        max_steps: int = QLEARNING_CONFIG['max_steps_per_episode'],
        render: bool = False,
        log_interval: int = QLEARNING_CONFIG['log_interval']
    ) -> Dict[str, List[float]]:
        """
        Entrena el agente Q-Learning en el entorno.
        
        Parámetros:
        -----------
        env : Any
            Entorno compatible con OpenAI Gym
        episodes : int, opcional
            Número de episodios para entrenar (default: QLEARNING_CONFIG['episodes'])
        max_steps : int, opcional
            Máximo número de pasos por episodio (default: QLEARNING_CONFIG['max_steps_per_episode'])
        render : bool, opcional
            Si renderizar el entorno durante el entrenamiento (default: False)
        log_interval : int, opcional
            Cada cuántos episodios mostrar estadísticas (default: QLEARNING_CONFIG['log_interval'])
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historia de entrenamiento
        """
        history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'epsilons': [],
            'avg_rewards': []
        }
        
        # Variables para seguimiento
        episode_rewards_window = []
        start_time = time.time()
        
        for episode in range(episodes):
            # Ejecutar episodio
            episode_reward, steps = self._run_episode(env, max_steps, render)
            
            # Actualizar epsilon después de cada episodio si no se hace por paso
            if not self.use_decay_schedule:
                self.update_epsilon()
            
            # Actualizar estadísticas
            avg_reward = self._update_history(history, episode_reward, steps, 
                                            episode_rewards_window, log_interval)
            
            # Mostrar progreso
            if (episode + 1) % log_interval == 0:
                elapsed_time = time.time() - start_time
                print(f"Episodio {episode+1}/{episodes} - "
                      f"Recompensa: {episode_reward:.2f}, "
                      f"Promedio: {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.4f}, "
                      f"Tiempo: {elapsed_time:.2f}s")
                start_time = time.time()
        
        return history
    
    def evaluate(self, env: Any, episodes: int = 10, render: bool = False) -> float:
        """
        Evalúa el agente entrenado.
        
        Parámetros:
        -----------
        env : Any
            Entorno compatible con OpenAI Gym
        episodes : int, opcional
            Número de episodios para evaluar (default: 10)
        render : bool, opcional
            Si renderizar el entorno durante la evaluación (default: False)
            
        Retorna:
        --------
        float
            Recompensa promedio
        """
        rewards = []
        episode_lengths = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            episode_reward = 0
            steps = 0
            done = False
            
            while not done:
                if render:
                    env.render()
                
                # Seleccionar la mejor acción (sin exploración)
                action = np.argmax(self.q_table[state])
                
                # Ejecutar acción
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Actualizar estado y recompensa
                state = next_state
                episode_reward += reward
                steps += 1
            
            rewards.append(episode_reward)
            episode_lengths.append(steps)
            print(f"Episodio de evaluación {episode+1}/{episodes} - Recompensa: {episode_reward:.2f}")
        
        avg_reward = np.mean(rewards)
        avg_length = np.mean(episode_lengths)
        print(f"Recompensa promedio de evaluación: {avg_reward:.2f}, "
              f"Longitud promedio: {avg_length:.2f}")
        
        return avg_reward
    
    def save_qtable(self, filepath: str) -> None:
        """
        Guarda la tabla Q en un archivo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta para guardar la tabla Q
        """
        np.save(filepath, self.q_table)
        print(f"Tabla Q guardada en {filepath}")
    
    def load_qtable(self, filepath: str) -> None:
        """
        Carga la tabla Q desde un archivo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta para cargar la tabla Q
        """
        self.q_table = np.load(filepath)
        print(f"Tabla Q cargada desde {filepath}")
    
    def visualize_training(self, history: Dict[str, List[float]], window_size: int = 10) -> None:
        """
        Visualiza los resultados del entrenamiento.
        
        Parámetros:
        -----------
        history : Dict[str, List[float]]
            Historia de entrenamiento
        window_size : int, opcional
            Tamaño de ventana para suavizado (default: 10)
        """
        def smooth(data: List[float], window_size: int) -> np.ndarray:
            """Aplica suavizado con media móvil a los datos"""
            kernel = np.ones(window_size) / window_size
            return np.convolve(np.array(data), kernel, mode='valid')
        
        _, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Recompensas por episodio
        axs[0, 0].plot(history['episode_rewards'], alpha=0.3, color='blue', label='Original')
        if len(history['episode_rewards']) > window_size:
            axs[0, 0].plot(
                range(window_size-1, len(history['episode_rewards'])),
                smooth(history['episode_rewards'], window_size),
                color='blue',
                label=f'Suavizado (ventana={window_size})'
            )
        axs[0, 0].set_title('Recompensa por Episodio')
        axs[0, 0].set_xlabel('Episodio')
        axs[0, 0].set_ylabel('Recompensa')
        axs[0, 0].grid(alpha=0.3)
        axs[0, 0].legend()
        
        # 2. Recompensa promedio
        axs[0, 1].plot(history['avg_rewards'], color='green')
        axs[0, 1].set_title('Recompensa Promedio')
        axs[0, 1].set_xlabel('Episodio')
        axs[0, 1].set_ylabel('Recompensa Promedio')
        axs[0, 1].grid(alpha=0.3)
        
        # 3. Epsilon
        axs[1, 0].plot(history['epsilons'], color='red')
        axs[1, 0].set_title('Epsilon (Exploración)')
        axs[1, 0].set_xlabel('Episodio')
        axs[1, 0].set_ylabel('Epsilon')
        axs[1, 0].grid(alpha=0.3)
        
        # 4. Longitud de episodios
        axs[1, 1].plot(history['episode_lengths'], alpha=0.3, color='purple', label='Original')
        if len(history['episode_lengths']) > window_size:
            axs[1, 1].plot(
                range(window_size-1, len(history['episode_lengths'])),
                smooth(history['episode_lengths'], window_size),
                color='purple',
                label=f'Suavizado (ventana={window_size})'
            )
        axs[1, 1].set_title('Longitud de Episodio')
        axs[1, 1].set_xlabel('Episodio')
        axs[1, 1].set_ylabel('Pasos')
        axs[1, 1].grid(alpha=0.3)
        axs[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def _setup_grid_visualization(self, ax: plt.Axes, rows: int, cols: int) -> None:
        """
        Configura la visualización básica de la cuadrícula.
        
        Parámetros:
        -----------
        ax : plt.Axes
            Eje para la visualización
        rows : int
            Número de filas en la grilla
        cols : int
            Número de columnas en la grilla
        """
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.invert_yaxis()
        
        # Dibujar líneas de cuadrícula
        for i in range(rows+1):
            ax.axhline(i, color='black', alpha=0.3)
        for j in range(cols+1):
            ax.axvline(j, color='black', alpha=0.3)
    
    def _draw_policy_arrows(self, ax: plt.Axes, row: int, col: int, state: int, arrows: Dict[int, Tuple[float, float]]) -> None:
        """
        Dibuja flechas que representan la política para un estado.
        
        Parámetros:
        -----------
        ax : plt.Axes
            Eje para la visualización
        row : int
            Fila del estado en la grilla
        col : int
            Columna del estado en la grilla
        state : int
            Índice del estado
        arrows : Dict[int, Tuple[float, float]]
            Diccionario que mapea acciones a vectores de dirección
        """
        if np.all(self.q_table[state] == 0):
            # Si todas las acciones son iguales, no hay preferencia clara
            for action in range(self.n_actions):
                dx, dy = arrows[action]
                ax.arrow(col + 0.5, row + 0.5, dx, dy, head_width=0.1, head_length=0.1, 
                        fc='gray', ec='gray', alpha=0.3)
        else:
            # Dibujar la acción con mayor valor Q
            best_action = np.argmax(self.q_table[state])
            dx, dy = arrows[best_action]
            ax.arrow(col + 0.5, row + 0.5, dx, dy, head_width=0.1, head_length=0.1, 
                    fc='blue', ec='blue')
            
            # Mostrar el valor Q
            ax.text(col + 0.5, row + 0.7, f"{np.max(self.q_table[state]):.2f}", 
                   ha='center', va='center', fontsize=8)
    
    def visualize_policy(self, env: Any, grid_size: Optional[Tuple[int, int]] = None) -> None:
        """
        Visualiza la política aprendida para entornos de tipo grilla.
        
        Parámetros:
        -----------
        env : Any
            Entorno, preferentemente de tipo grilla
        grid_size : Optional[Tuple[int, int]], opcional
            Tamaño de la grilla (filas, columnas) (default: None)
        """
        if grid_size is None:
            try:
                grid_size = (env.nrow, env.ncol)  # Para FrozenLake
            except AttributeError:
                print("El entorno no parece ser una grilla o no se proporcionó grid_size")
                return
        
        rows, cols = grid_size
        
        # Crear un mapa de flechas para visualizar la política
        _, ax = plt.subplots(figsize=(10, 10))
        self._setup_grid_visualization(ax, rows, cols)
        
        # Mapeo de acciones a flechas: 0=izquierda, 1=abajo, 2=derecha, 3=arriba
        arrows = {
            0: (-0.2, 0),   # Izquierda
            1: (0, 0.2),    # Abajo
            2: (0.2, 0),    # Derecha
            3: (0, -0.2)    # Arriba
        }
        
        # Dibujar flechas para cada celda
        for row in range(rows):
            for col in range(cols):
                state = row * cols + col  # Esto puede variar según cómo se mapeen los estados
                self._draw_policy_arrows(ax, row, col, state, arrows)
        
        plt.title('Política Aprendida')
        plt.tight_layout()
        plt.show()
    
    def visualize_value_function(self, env: Any, grid_size: Optional[Tuple[int, int]] = None) -> None:
        """
        Visualiza la función de valor aprendida para entornos de tipo grilla.
        
        Parámetros:
        -----------
        env : Any
            Entorno, preferentemente de tipo grilla
        grid_size : Optional[Tuple[int, int]], opcional
            Tamaño de la grilla (filas, columnas) (default: None)
        """
        if grid_size is None:
            try:
                grid_size = (env.nrow, env.ncol)  # Para FrozenLake
            except AttributeError:
                print("El entorno no parece ser una grilla o no se proporcionó grid_size")
                return
        
        rows, cols = grid_size
        
        # Calcular la función de valor: V(s) = max_a Q(s,a)
        value_function = np.max(self.q_table, axis=1).reshape(rows, cols)
        
        _, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(value_function, cmap='viridis')
        
        # Añadir barra de color
        cbar = plt.colorbar(im)
        cbar.set_label('Valor Esperado')
        
        # Añadir etiquetas
        for i in range(rows):
            for j in range(cols):
                state = i * cols + j
                value = np.max(self.q_table[state])
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", 
                        color="w" if value < (value_function.max() + value_function.min())/2 else "black")
        
        plt.title('Función de Valor (V)')
        plt.tight_layout()
        plt.show()
    
    def compare_with_optimal(self, optimal_policy: np.ndarray) -> float:
        """
        Compara la política aprendida con una política óptima conocida.
        
        Parámetros:
        -----------
        optimal_policy : np.ndarray
            Política óptima conocida como array de acciones para cada estado
            
        Retorna:
        --------
        float
            Porcentaje de acciones que coinciden con la política óptima
        """
        # Obtener la política actual
        current_policy = np.array([np.argmax(self.q_table[s]) for s in range(self.n_states)])
        
        # Calcular coincidencias
        matches = np.sum(current_policy == optimal_policy)
        
        # Calcular porcentaje de coincidencia
        match_percentage = matches / self.n_states * 100
        
        print(f"Coincidencia con política óptima: {match_percentage:.2f}%")
        
        return match_percentage
    
    def get_q_table(self) -> np.ndarray:
        """
        Devuelve la tabla Q actual.
        
        Retorna:
        --------
        np.ndarray
            Tabla Q actual
        """
        return self.q_table