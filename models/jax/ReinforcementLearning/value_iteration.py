import os, sys
import numpy as np
import matplotlib.pyplot as plt
import time
import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional, Union, Any, NamedTuple
import pickle

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from models.config import VALUE_ITERATION_CONFIG


class ValueIterationState(NamedTuple):
    """Estado interno para la iteración de valor."""
    V: jnp.ndarray
    policy: jnp.ndarray
    value_changes: List[float]
    iteration_times: List[float]


class ValueIteration:
    """
    Implementación del algoritmo de Iteración de Valor (Value Iteration) con JAX.
    
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
        
        # Inicializar función de valor y política
        V = jnp.zeros(n_states)
        policy = jnp.zeros((n_states, n_actions))
        
        # Crear estado interno
        self.state = ValueIterationState(
            V=V,
            policy=policy,
            value_changes=[],
            iteration_times=[]
        )
        
        # Compilar funciones puras para mejor rendimiento
        self._calculate_action_values = jax.jit(self._calculate_action_values)

    def _calculate_action_values(
        self, 
        V: jnp.ndarray, 
        transitions: Dict[int, Dict[int, List]], 
        state: int
    ) -> jnp.ndarray:
        """
        Calcula los valores Q para todas las acciones en un estado.
        
        Parámetros:
        -----------
        V : jnp.ndarray
            Función de valor actual
        transitions : Dict[int, Dict[int, List]]
            Diccionario con las transiciones del entorno
        state : int
            Estado para calcular valores de acción
            
        Retorna:
        --------
        jnp.ndarray
            Valores Q para todas las acciones en el estado dado
        """
        action_values = jnp.zeros(self.n_actions)
        
        for a in range(self.n_actions):
            for prob, next_s, r, done in transitions[state][a]:
                # Valor esperado usando la ecuación de Bellman
                not_done = jnp.logical_not(done)
                action_values = action_values.at[a].add(
                    prob * (r + self.gamma * V[next_s] * not_done)
                )
        
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
        delta = 0.0
        new_V = self.state.V
        
        for s in range(self.n_states):
            v_old = self.state.V[s]
            
            # Calcular el valor Q para cada acción y tomar el máximo
            action_values = self._calculate_action_values(self.state.V, env.P, s)
            
            # Actualizar el valor del estado con el máximo valor de acción
            new_V = new_V.at[s].set(jnp.max(action_values))
            
            # Actualizar delta
            delta = max(delta, abs(v_old - new_V[s]))
        
        # Actualizar estado
        self.state = self.state._replace(V=new_V)
        
        return float(delta)

    def extract_policy(self, env: Any) -> jnp.ndarray:
        """
        Extrae la política óptima a partir de la función de valor.
        
        Parámetros:
        -----------
        env : Any
            Entorno con dinámicas de transición
            
        Retorna:
        --------
        jnp.ndarray
            Política óptima (determinística)
        """
        policy = jnp.zeros((self.n_states, self.n_actions))
        
        for s in range(self.n_states):
            # Calcular el valor Q para cada acción
            action_values = self._calculate_action_values(self.state.V, env.P, s)
            
            # Política determinística: asignar probabilidad 1.0 a la mejor acción
            best_action = jnp.argmax(action_values)
            policy = policy.at[s, best_action].set(1.0)
        
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
        value_changes = []
        iteration_times = []
        
        for i in range(self.max_iterations):
            iteration_start = time.time()
            
            # Actualizar función de valor
            delta = self.value_update(env)
            
            # Registrar cambio de valor
            value_changes.append(float(delta))
            
            # Registrar tiempo de iteración
            iteration_time = time.time() - iteration_start
            iteration_times.append(iteration_time)
            
            iterations = i + 1
            
            print(f"Iteración {iterations}: Delta = {delta:.6f}, Tiempo = {iteration_time:.2f} segundos")
            
            # Verificar convergencia
            if delta < self.theta:
                print("¡Convergencia alcanzada!")
                break
        
        # Extraer política óptima
        policy = self.extract_policy(env)
        
        # Actualizar estado
        self.state = self.state._replace(
            policy=policy,
            value_changes=value_changes,
            iteration_times=iteration_times
        )
        
        total_time = time.time() - start_time
        print(f"Iteración de valor completada en {iterations} iteraciones, {total_time:.2f} segundos")
        
        history = {
            'iterations': iterations,
            'value_changes': value_changes,
            'iteration_times': iteration_times,
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
        return int(jnp.argmax(self.state.policy[state]))

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
        return float(self.state.V[state])

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
        
        Parámetros:
        -----------
        filepath : str
            Ruta donde guardar el modelo
        """
        data = {
            'policy': np.array(self.state.policy),
            'V': np.array(self.state.V),
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
        
        # Actualizar estado
        self.state = self.state._replace(
            V=jnp.array(data['V']),
            policy=jnp.array(data['policy'])
        )
        self.n_states = data['n_states']
        self.n_actions = data['n_actions']
        self.gamma = data['gamma']
        
        print(f"Modelo cargado desde {filepath}")

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
                
            value_grid[i, j] = self.get_value(s)
        
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

    def parallel_value_iteration(self, env: Any) -> Dict[str, List]:
        """
        Implementa iteración de valor con cálculos paralelizados usando JAX.
        
        Parámetros:
        -----------
        env : Any
            Entorno con dinámicas de transición
            
        Retorna:
        --------
        Dict[str, List]
            Diccionario con historial de entrenamiento
        """
        print("Iniciando iteración de valor paralela...")
        
        iterations = 0
        start_time = time.time()
        value_changes = []
        iteration_times = []
        
        # Preparar las funciones para cálculo en paralelo
        value_fn = jax.vmap(
            lambda s, V: self._calculate_action_values(V, env.P, s),
            in_axes=(0, None)
        )
        
        for i in range(self.max_iterations):
            iteration_start = time.time()
            
            # Calcular valores de acción para todos los estados en paralelo
            states = jnp.arange(self.n_states)
            all_action_values = value_fn(states, self.state.V)
            
            # Actualizar función de valor
            new_V = jnp.max(all_action_values, axis=1)
            
            # Calcular delta
            delta = jnp.max(jnp.abs(new_V - self.state.V))
            
            # Actualizar estado
            _ = self.state.V
            self.state = self.state._replace(V=new_V)
            
            # Registrar métricas
            value_changes.append(float(delta))
            iteration_time = time.time() - iteration_start
            iteration_times.append(iteration_time)
            
            iterations = i + 1
            
            print(f"Iteración {iterations}: Delta = {float(delta):.6f}, "
                  f"Tiempo = {iteration_time:.2f} segundos")
            
            # Verificar convergencia
            if delta < self.theta:
                print("¡Convergencia alcanzada!")
                break
        
        # Extraer política óptima
        policy = self.extract_policy(env)
        
        # Actualizar estado final
        self.state = self.state._replace(
            policy=policy,
            value_changes=value_changes,
            iteration_times=iteration_times
        )
        
        total_time = time.time() - start_time
        print(f"Iteración de valor paralela completada en {iterations} iteraciones, "
              f"{total_time:.2f} segundos")
        
        history = {
            'iterations': iterations,
            'value_changes': value_changes,
            'iteration_times': iteration_times,
            'total_time': total_time
        }
        
        return history