import os, sys
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Optional, Union, Any
import pickle

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from models.config import VALUE_ITERATION_CONFIG


class ValueIteration:
    """
    Implementación del algoritmo de Iteración de Valor (Value Iteration).
    
    La Iteración de Valor es un método de programación dinámica que encuentra la política
    óptima calculando directamente la función de valor óptima utilizando la ecuación de
    optimalidad de Bellman, sin mantener explícitamente una política durante el proceso.
    """
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float = VALUE_ITERATION_CONFIG['gamma'],
        theta: float = VALUE_ITERATION_CONFIG['theta'],
        max_iterations: int = VALUE_ITERATION_CONFIG['max_iterations']
    ) -> None:
        """
        Inicializa el agente de Iteración de Valor.
        
        Parámetros:
        -----------
        n_states : int
            Número de estados en el entorno
        n_actions : int
            Número de acciones en el entorno
        gamma : float, opcional
            Factor de descuento (default: VALUE_ITERATION_CONFIG['gamma'])
        theta : float, opcional
            Umbral para convergencia (default: VALUE_ITERATION_CONFIG['theta'])
        max_iterations : int, opcional
            Número máximo de iteraciones (default: VALUE_ITERATION_CONFIG['max_iterations'])
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations
        
        # Inicializar función de valor
        self.V = np.zeros(n_states)
        
        # La política se deriva de la función de valor (no se mantiene explícitamente)
        self.policy = np.zeros((n_states, n_actions))
        
        # Para métricas
        self.value_changes = []
        self.iteration_times = []
    
    def _calculate_action_values(self, state: int, transitions: Dict[int, Dict[int, List]]) -> np.ndarray:
        """
        Calcula los valores Q para todas las acciones en un estado.
        
        Parámetros:
        -----------
        state : int
            Estado para calcular valores de acción
        transitions : Dict[int, Dict[int, List]]
            Diccionario con las transiciones del entorno
            
        Retorna:
        --------
        np.ndarray
            Valores Q para todas las acciones en el estado dado
        """
        action_values = np.zeros(self.n_actions)
        
        for a in range(self.n_actions):
            # Calcular el valor esperado para la acción a desde el estado state
            for prob, next_s, r, done in transitions[state][a]:
                # Valor esperado usando la ecuación de Bellman
                action_values[a] += prob * (r + self.gamma * self.V[next_s] * (not done))
        
        return action_values
    
    def value_update(self, env: Any) -> float:
        """
        Realiza una iteración de actualización de la función de valor.
        
        Parámetros:
        -----------
        env : Any
            Entorno con dinámicas de transición
            
        Retorna:
        --------
        float
            Delta máximo (cambio máximo en la función de valor)
        """
        delta = 0
        
        for s in range(self.n_states):
            v_old = self.V[s]
            
            # Calcular el valor Q para cada acción y tomar el máximo
            action_values = self._calculate_action_values(s, env.P)
            
            # Actualizar el valor del estado con el máximo valor de acción
            self.V[s] = np.max(action_values)
            
            # Actualizar delta
            delta = max(delta, abs(v_old - self.V[s]))
        
        return delta
    
    def extract_policy(self, env: Any) -> np.ndarray:
        """
        Extrae la política óptima a partir de la función de valor.
        
        Parámetros:
        -----------
        env : Any
            Entorno con dinámicas de transición
            
        Retorna:
        --------
        np.ndarray
            Política óptima (determinística)
        """
        policy = np.zeros((self.n_states, self.n_actions))
        
        for s in range(self.n_states):
            # Calcular el valor Q para cada acción
            action_values = self._calculate_action_values(s, env.P)
            
            # Política determinística: asignar probabilidad 1.0 a la mejor acción
            best_action = np.argmax(action_values)
            policy[s, best_action] = 1.0
        
        return policy
    
    def train(self, env: Any) -> Dict[str, List]:
        """
        Entrena al agente usando iteración de valor.
        
        Parámetros:
        -----------
        env : Any
            Entorno con dinámicas de transición
            
        Retorna:
        --------
        Dict[str, List]
            Diccionario con historial de entrenamiento
        """
        print("Iniciando iteración de valor...")
        
        iterations = 0
        start_time = time.time()
        self.value_changes = []
        self.iteration_times = []
        
        for i in range(self.max_iterations):
            iteration_start = time.time()
            
            # Actualizar función de valor
            delta = self.value_update(env)
            
            # Registrar cambio de valor
            self.value_changes.append(delta)
            
            # Registrar tiempo de iteración
            iteration_time = time.time() - iteration_start
            self.iteration_times.append(iteration_time)
            
            iterations = i + 1
            
            print(f"Iteración {iterations}: Delta = {delta:.6f}, Tiempo = {iteration_time:.2f} segundos")
            
            # Verificar convergencia
            if delta < self.theta:
                print("¡Convergencia alcanzada!")
                break
        
        # Extraer política óptima
        self.policy = self.extract_policy(env)
        
        total_time = time.time() - start_time
        print(f"Iteración de valor completada en {iterations} iteraciones, {total_time:.2f} segundos")
        
        history = {
            'iterations': iterations,
            'value_changes': self.value_changes,
            'iteration_times': self.iteration_times,
            'total_time': total_time
        }
        
        return history
    
    def get_action(self, state: int) -> int:
        """
        Devuelve la mejor acción para un estado dado según la política actual.
        
        Parámetros:
        -----------
        state : int
            Estado actual
            
        Retorna:
        --------
        int
            Mejor acción
        """
        return np.argmax(self.policy[state])
    
    def get_value(self, state: int) -> float:
        """
        Devuelve el valor de un estado según la función de valor actual.
        
        Parámetros:
        -----------
        state : int
            Estado para obtener su valor
            
        Retorna:
        --------
        float
            Valor del estado
        """
        return self.V[state]
    
    def evaluate(self, env: Any, max_steps: int = 100, episodes: int = 10) -> float:
        """
        Evalúa la política actual en el entorno.
        
        Parámetros:
        -----------
        env : Any
            Entorno para evaluar
        max_steps : int, opcional
            Pasos máximos por episodio (default: 100)
        episodes : int, opcional
            Número de episodios para evaluar (default: 10)
            
        Retorna:
        --------
        float
            Recompensa promedio en los episodios
        """
        total_rewards = []
        episode_lengths = []
        
        for ep in range(episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done and steps < max_steps:
                action = self.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                total_reward += reward
                state = next_state
                steps += 1
            
            total_rewards.append(total_reward)
            episode_lengths.append(steps)
            print(f"Episodio {ep+1}: Recompensa = {total_reward}, Pasos = {steps}")
        
        avg_reward = np.mean(total_rewards)
        avg_length = np.mean(episode_lengths)
        print(f"Evaluación: recompensa media en {episodes} episodios = {avg_reward:.2f}, " +
              f"longitud media = {avg_length:.2f}")
        
        return avg_reward
    
    def save(self, filepath: str) -> None:
        """
        Guarda la política y función de valor en un archivo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta donde guardar el modelo
        """
        data = {
            'policy': self.policy,
            'V': self.V,
            'n_states': self.n_states,
            'n_actions': self.n_actions,
            'gamma': self.gamma
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Modelo guardado en {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Carga la política y función de valor desde un archivo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta desde donde cargar el modelo
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.policy = data['policy']
        self.V = data['V']
        self.n_states = data['n_states']
        self.n_actions = data['n_actions']
        self.gamma = data['gamma']
        
        print(f"Modelo cargado desde {filepath}")
    
    def _get_grid_position(self, env: Any, state: int) -> Tuple[int, int]:
        """
        Obtiene la posición en la cuadrícula para un estado.
        
        Parámetros:
        -----------
        env : Any
            Entorno con estructura de cuadrícula
        state : int
            Estado a convertir en posición
            
        Retorna:
        --------
        Tuple[int, int]
            Posición (i, j) en la cuadrícula
        """
        if hasattr(env, 'state_mapping'):
            return env.state_mapping(state)
        else:
            # Asumir orden row-major
            return state // env.shape[1], state % env.shape[1]
    
    def _is_terminal_state(self, env: Any, state: int) -> bool:
        """
        Determina si un estado es terminal.
        
        Parámetros:
        -----------
        env : Any
            Entorno con transiciones
        state : int
            Estado a comprobar
            
        Retorna:
        --------
        bool
            True si el estado es terminal, False en caso contrario
        """
        for a in range(self.n_actions):
            for _, _, _, done in env.P[state][a]:
                if done:
                    return True
        return False
    
    def _setup_grid(self, ax: plt.Axes, grid_shape: Tuple[int, int]) -> None:
        """
        Configura la cuadrícula para visualización.
        
        Parámetros:
        -----------
        ax : plt.Axes
            Ejes para dibujar
        grid_shape : Tuple[int, int]
            Forma de la cuadrícula (filas, columnas)
        """
        # Configurar límites
        ax.set_xlim([0, grid_shape[1]])
        ax.set_ylim([0, grid_shape[0]])
        
        # Dibujar líneas de cuadrícula
        for i in range(grid_shape[1] + 1):
            ax.axvline(i, color='black', linestyle='-')
        for j in range(grid_shape[0] + 1):
            ax.axhline(j, color='black', linestyle='-')
            
    def _draw_action_arrows(self, ax: plt.Axes, env: Any, grid_shape: Tuple[int, int]) -> None:
        """
        Dibuja flechas para las acciones en cada estado no terminal.
        
        Parámetros:
        -----------
        ax : plt.Axes
            Ejes para dibujar
        env : Any
            Entorno con estructura de cuadrícula
        grid_shape : Tuple[int, int]
            Forma de la cuadrícula (filas, columnas)
        """
        # Definir direcciones de flechas
        directions = {
            0: (0, -0.4),  # Izquierda
            1: (0, 0.4),   # Derecha
            2: (-0.4, 0),  # Abajo
            3: (0.4, 0)    # Arriba
        }
        
        for s in range(self.n_states):
            if self._is_terminal_state(env, s):
                continue
                
            i, j = self._get_grid_position(env, s)
            action = self.get_action(s)
            
            if action in directions:
                dx, dy = directions[action]
                ax.arrow(j + 0.5, grid_shape[0] - i - 0.5, dx, dy, 
                         head_width=0.1, head_length=0.1, fc='blue', ec='blue')
    
    def _show_state_values(self, ax: plt.Axes, env: Any, grid_shape: Tuple[int, int]) -> None:
        """
        Muestra los valores de cada estado en la cuadrícula.
        
        Parámetros:
        -----------
        ax : plt.Axes
            Ejes para dibujar
        env : Any
            Entorno con estructura de cuadrícula
        grid_shape : Tuple[int, int]
            Forma de la cuadrícula (filas, columnas)
        """
        for s in range(self.n_states):
            i, j = self._get_grid_position(env, s)
            value = self.get_value(s)
            ax.text(j + 0.5, grid_shape[0] - i - 0.5, f"{value:.2f}", 
                   ha='center', va='center', color='red', fontsize=9)
    
    def visualize_policy(self, env: Any, title: str = "Política Óptima") -> None:
        """
        Visualiza la política para entornos de tipo cuadrícula.
        
        Parámetros:
        -----------
        env : Any
            Entorno con estructura de cuadrícula
        title : str, opcional
            Título para la visualización (default: "Política Óptima")
        """
        if not hasattr(env, 'shape'):
            print("El entorno no tiene estructura de cuadrícula para visualización")
            return
        
        grid_shape = env.shape
        _, ax = plt.subplots(figsize=(8, 8))
        
        # Configurar y dibujar cuadrícula
        self._setup_grid(ax, grid_shape)
        
        # Dibujar flechas para las acciones
        self._draw_action_arrows(ax, env, grid_shape)
        
        # Mostrar valores de estados
        self._show_state_values(ax, env, grid_shape)
        
        ax.set_title(title)
        plt.tight_layout()
        plt.show()
    
    def visualize_value_function(self, env: Any, title: str = "Función de Valor") -> None:
        """
        Visualiza la función de valor para entornos de tipo cuadrícula.
        
        Parámetros:
        -----------
        env : Any
            Entorno con estructura de cuadrícula
        title : str, opcional
            Título para la visualización (default: "Función de Valor")
        """
        if not hasattr(env, 'shape'):
            print("El entorno no tiene estructura de cuadrícula para visualización")
            return
        
        grid_shape = env.shape
        
        # Crear matriz para visualización
        value_grid = np.zeros(grid_shape)
        
        # Llenar matriz con valores
        for s in range(self.n_states):
            if hasattr(env, 'state_mapping'):
                i, j = env.state_mapping(s)
            else:
                i, j = s // grid_shape[1], s % grid_shape[1]
                
            value_grid[i, j] = self.V[s]
        
        _, ax = plt.subplots(figsize=(10, 8))
        
        # Crear mapa de calor
        im = ax.imshow(value_grid, cmap='viridis')
        
        # Añadir barra de color
        plt.colorbar(im, ax=ax, label='Valor')
        
        # Mostrar valores en cada celda
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                ax.text(j, i, f"{value_grid[i, j]:.2f}", ha='center', va='center',
                        color='white' if value_grid[i, j] < np.max(value_grid)/1.5 else 'black')
        
        ax.set_title(title)
        plt.tight_layout()
        plt.show()
    
    def visualize_training(self, history: Dict[str, List]) -> None:
        """
        Visualiza las métricas de entrenamiento.
        
        Parámetros:
        -----------
        history : Dict[str, List]
            Diccionario con historial de entrenamiento
        """
        _, axs = plt.subplots(2, 1, figsize=(12, 8))
        
        # Gráfico de cambios en la función de valor (delta)
        axs[0].plot(range(1, len(history['value_changes']) + 1), 
                    history['value_changes'])
        axs[0].set_title('Cambios en la Función de Valor (Delta)')
        axs[0].set_xlabel('Iteración')
        axs[0].set_ylabel('Delta')
        axs[0].set_yscale('log')  # Escala logarítmica para ver mejor la convergencia
        axs[0].grid(True)
        
        # Gráfico de tiempos de iteración
        axs[1].plot(range(1, len(history['iteration_times']) + 1), 
                    history['iteration_times'])
        axs[1].set_title('Tiempos de Iteración')
        axs[1].set_xlabel('Iteración')
        axs[1].set_ylabel('Tiempo (segundos)')
        axs[1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def vectorized_value_iteration(self, env: Any) -> Dict[str, List]:
        """
        Implementa una versión vectorizada de iteración de valor usando operaciones de NumPy.
        
        Parámetros:
        -----------
        env : Any
            Entorno con dinámicas de transición
            
        Retorna:
        --------
        Dict[str, List]
            Diccionario con historial de entrenamiento
        """
        print("Iniciando iteración de valor vectorizada...")
        
        # Preparar estructuras de datos para cálculo vectorizado
        transitions = np.zeros((self.n_states, self.n_actions, self.n_states))
        rewards = np.zeros((self.n_states, self.n_actions, self.n_states))
        terminals = np.zeros((self.n_states, self.n_actions, self.n_states), dtype=bool)
        
        # Convertir modelo de transición a matrices
        for s in range(self.n_states):
            for a in range(self.n_actions):
                for prob, next_s, r, done in env.P[s][a]:
                    transitions[s, a, next_s] += prob
                    rewards[s, a, next_s] = r
                    terminals[s, a, next_s] = done
        
        iterations = 0
        start_time = time.time()
        self.value_changes = []
        self.iteration_times = []
        
        for i in range(self.max_iterations):
            iteration_start = time.time()
            
            # Calcular los valores Q para todos los estados y acciones
            # Q(s,a) = Σ_s' P(s'|s,a) * [R(s,a,s') + γ*V(s')*(1-terminal(s'))]
            v_expanded = np.expand_dims(self.V, axis=(0, 1))
            q_values = np.sum(
                transitions * (rewards + self.gamma * v_expanded * ~terminals), 
                axis=2
            )
            
            # Actualizar con los valores máximos
            v_old = self.V.copy()
            self.V = np.max(q_values, axis=1)
            
            # Calcular delta
            delta = np.max(np.abs(v_old - self.V))
            
            # Registrar métricas
            self.value_changes.append(delta)
            iteration_time = time.time() - iteration_start
            self.iteration_times.append(iteration_time)
            
            iterations = i + 1
            
            print(f"Iteración {iterations}: Delta = {delta:.6f}, Tiempo = {iteration_time:.2f} segundos")
            
            # Verificar convergencia
            if delta < self.theta:
                print("¡Convergencia alcanzada!")
                break
        
        # Extraer política óptima
        q_values = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                for prob, next_s, r, done in env.P[s][a]:
                    q_values[s, a] += prob * (r + self.gamma * self.V[next_s] * (not done))
        
        # Política determinística
        best_actions = np.argmax(q_values, axis=1)
        self.policy = np.zeros((self.n_states, self.n_actions))
        self.policy[np.arange(self.n_states), best_actions] = 1.0
        
        total_time = time.time() - start_time
        print(f"Iteración de valor vectorizada completada en {iterations} iteraciones, "
              f"{total_time:.2f} segundos")
        
        history = {
            'iterations': iterations,
            'value_changes': self.value_changes,
            'iteration_times': self.iteration_times,
            'total_time': total_time
        }
        
        return history