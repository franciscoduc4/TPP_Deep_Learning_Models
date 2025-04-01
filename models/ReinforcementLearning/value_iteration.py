import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Optional, Union
import pickle
from ..config import VALUE_ITERATION_CONFIG

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
    ):
        """
        Inicializa el agente de Iteración de Valor.
        
        Args:
            n_states: Número de estados en el entorno
            n_actions: Número de acciones en el entorno
            gamma: Factor de descuento
            theta: Umbral para convergencia
            max_iterations: Número máximo de iteraciones
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
    
    def value_update(self, env) -> float:
        """
        Realiza una iteración de actualización de la función de valor.
        
        Args:
            env: Entorno con dinámicas de transición
            
        Returns:
            Delta máximo (cambio máximo en la función de valor)
        """
        delta = 0
        
        for s in range(self.n_states):
            v_old = self.V[s]
            
            # Calcular el valor Q para cada acción y tomar el máximo
            action_values = np.zeros(self.n_actions)
            
            for a in range(self.n_actions):
                # Calcular el valor esperado para la acción a desde el estado s
                for prob, next_s, r, done in env.P[s][a]:
                    # Valor esperado usando la ecuación de Bellman
                    action_values[a] += prob * (r + self.gamma * self.V[next_s] * (not done))
            
            # Actualizar el valor del estado con el máximo valor de acción
            self.V[s] = np.max(action_values)
            
            # Actualizar delta
            delta = max(delta, abs(v_old - self.V[s]))
        
        return delta
    
    def extract_policy(self, env):
        """
        Extrae la política óptima a partir de la función de valor.
        
        Args:
            env: Entorno con dinámicas de transición
            
        Returns:
            Política óptima (determinística)
        """
        policy = np.zeros((self.n_states, self.n_actions))
        
        for s in range(self.n_states):
            # Calcular el valor Q para cada acción
            action_values = np.zeros(self.n_actions)
            
            for a in range(self.n_actions):
                for prob, next_s, r, done in env.P[s][a]:
                    action_values[a] += prob * (r + self.gamma * self.V[next_s] * (not done))
            
            # Política determinística: asignar probabilidad 1.0 a la mejor acción
            best_action = np.argmax(action_values)
            policy[s, best_action] = 1.0
        
        return policy
    
    def train(self, env) -> Dict[str, List]:
        """
        Entrena al agente usando iteración de valor.
        
        Args:
            env: Entorno con dinámicas de transición
            
        Returns:
            Diccionario con historial de entrenamiento
        """
        print("Iniciando iteración de valor...")
        
        iterations = 0
        start_time = time.time()
        
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
        
        Args:
            state: Estado actual
            
        Returns:
            Mejor acción
        """
        return np.argmax(self.policy[state])
    
    def get_value(self, state: int) -> float:
        """
        Devuelve el valor de un estado según la función de valor actual.
        
        Args:
            state: Estado para obtener su valor
            
        Returns:
            Valor del estado
        """
        return self.V[state]
    
    def evaluate(self, env, max_steps: int = 100, episodes: int = 10) -> float:
        """
        Evalúa la política actual en el entorno.
        
        Args:
            env: Entorno para evaluar
            max_steps: Pasos máximos por episodio
            episodes: Número de episodios para evaluar
            
        Returns:
            Recompensa promedio en los episodios
        """
        total_rewards = []
        
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
            print(f"Episodio {ep+1}: Recompensa = {total_reward}, Pasos = {steps}")
        
        avg_reward = np.mean(total_rewards)
        print(f"Evaluación: recompensa media en {episodes} episodios = {avg_reward:.2f}")
        
        return avg_reward
    
    def save(self, filepath: str) -> None:
        """
        Guarda la política y función de valor en un archivo.
        
        Args:
            filepath: Ruta donde guardar el modelo
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
        
        Args:
            filepath: Ruta desde donde cargar el modelo
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.policy = data['policy']
        self.V = data['V']
        self.n_states = data['n_states']
        self.n_actions = data['n_actions']
        self.gamma = data['gamma']
        
        print(f"Modelo cargado desde {filepath}")
    
    def visualize_policy(self, env, title: str = "Política Óptima"):
        """
        Visualiza la política para entornos de tipo cuadrícula.
        
        Args:
            env: Entorno con estructura de cuadrícula
            title: Título para la visualización
        """
        if not hasattr(env, 'shape'):
            print("El entorno no tiene estructura de cuadrícula para visualización")
            return
        
        grid_shape = env.shape
        _, ax = plt.subplots(figsize=(8, 8))
        
        # Crear cuadrícula
        ax.set_xlim([0, grid_shape[1]])
        ax.set_ylim([0, grid_shape[0]])
        
        # Dibujar líneas de cuadrícula
        for i in range(grid_shape[1] + 1):
            ax.axvline(i, color='black', linestyle='-')
        for j in range(grid_shape[0] + 1):
            ax.axhline(j, color='black', linestyle='-')
        
        # Dibujar flechas para las acciones
        for s in range(self.n_states):
            if hasattr(env, 'state_mapping'):
                # Convertir índice de estado a posición en cuadrícula
                i, j = env.state_mapping(s)
            else:
                # Asumir orden row-major
                i, j = s // grid_shape[1], s % grid_shape[1]
            
            # Omitir estados terminales
            if any(info[2] for a in range(self.n_actions) for _, _, info, _ in env.P[s][a]):
                continue
                
            action = self.get_action(s)
            
            # Definir direcciones de flechas
            directions = {
                0: (0, -0.4),  # Izquierda
                1: (0, 0.4),   # Derecha
                2: (-0.4, 0),  # Abajo
                3: (0.4, 0)    # Arriba
            }
            
            if action in directions:
                dx, dy = directions[action]
                ax.arrow(j + 0.5, grid_shape[0] - i - 0.5, dx, dy, 
                         head_width=0.1, head_length=0.1, fc='blue', ec='blue')
        
        # Mostrar valores de estados
        for s in range(self.n_states):
            if hasattr(env, 'state_mapping'):
                i, j = env.state_mapping(s)
            else:
                i, j = s // grid_shape[1], s % grid_shape[1]
            
            value = self.get_value(s)
            ax.text(j + 0.5, grid_shape[0] - i - 0.5, f"{value:.2f}", 
                   ha='center', va='center', color='red', fontsize=9)
        
        ax.set_title(title)
        plt.tight_layout()
        plt.show()
    
    def visualize_value_function(self, env, title: str = "Función de Valor"):
        """
        Visualiza la función de valor para entornos de tipo cuadrícula.
        
        Args:
            env: Entorno con estructura de cuadrícula
            title: Título para la visualización
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
        
        Args:
            history: Diccionario con historial de entrenamiento
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