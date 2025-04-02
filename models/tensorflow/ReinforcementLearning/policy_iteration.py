import os, sys
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Optional, Union
import pickle

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from models.config import POLICY_ITERATION_CONFIG

class PolicyIteration:
    """
    Implementación del algoritmo de Iteración de Política.
    
    La Iteración de Política alterna entre Evaluación de Política (calcular la función
    de valor para la política actual) y Mejora de Política (hacer la política codiciosa
    respecto a la función de valor actual).
    """
    # Constants for plot labels
    ITERATION_LABEL = 'Iteración'
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float = POLICY_ITERATION_CONFIG['gamma'],
        theta: float = POLICY_ITERATION_CONFIG['theta'],
        max_iterations: int = POLICY_ITERATION_CONFIG['max_iterations'],
        max_iterations_eval: int = POLICY_ITERATION_CONFIG['max_iterations_eval']
    ):
        """
        Inicializa el agente de Iteración de Política.
        
        Args:
            n_states: Número de estados en el entorno
            n_actions: Número de acciones en el entorno
            gamma: Factor de descuento
            theta: Umbral para convergencia
            max_iterations: Número máximo de iteraciones de iteración de política
            max_iterations_eval: Número máximo de iteraciones para evaluación de política
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations
        self.max_iterations_eval = max_iterations_eval
        
        # Inicializar función de valor
        self.V = np.zeros(n_states)
        
        # Inicializar política (aleatoria uniforme)
        self.policy = np.ones((n_states, n_actions)) / n_actions
        
        # Para métricas
        self.policy_changes = []
        self.value_changes = []
        self.policy_iteration_times = []
        self.eval_iteration_counts = []
    
    def _calculate_state_value(self, env, state, policy, V):
        """
        Calcula el valor de un estado dado según la política actual.
        
        Args:
            env: Entorno con dinámicas de transición
            state: Estado a evaluar
            policy: Política actual
            V: Función de valor actual
            
        Returns:
            Nuevo valor del estado
        """
        v_new = 0
        
        # Calcular valor esperado al seguir la política en el estado
        for a in range(self.n_actions):
            if policy[state, a] > 0:  # Solo considerar acciones con probabilidad no cero
                # Obtener información del siguiente estado y recompensa
                for prob, next_s, r, done in env.P[state][a]:
                    # Actualizar valor del estado usando la ecuación de Bellman
                    v_new += policy[state, a] * prob * (r + self.gamma * V[next_s] * (not done))
        
        return v_new
    
    def policy_evaluation(self, env, policy: np.ndarray) -> np.ndarray:
        """
        Evalúa la política actual calculando su función de valor.
        
        Args:
            env: Entorno con dinámicas de transición
            policy: Política a evaluar (distribución de probabilidad sobre acciones para cada estado)
            
        Returns:
            Función de valor para la política dada
        """
        V = np.zeros(self.n_states)
        
        for i in range(self.max_iterations_eval):
            delta = 0
            
            for s in range(self.n_states):
                v_old = V[s]
                V[s] = self._calculate_state_value(env, s, policy, V)
                delta = max(delta, abs(v_old - V[s]))
            
            # Verificar convergencia
            if delta < self.theta:
                break
        
        self.eval_iteration_counts.append(i + 1)
        return V
    
    def policy_improvement(self, env, V: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Mejora la política haciéndola codiciosa respecto a la función de valor.
        
        Args:
            env: Entorno con dinámicas de transición
            V: Función de valor actual
            
        Returns:
            Tupla de (nueva política, política_estable)
        """
        policy = np.zeros((self.n_states, self.n_actions))
        policy_stable = True
        
        for s in range(self.n_states):
            # Encontrar la mejor acción anterior
            old_action = np.argmax(self.policy[s])
            
            # Calcular valores de acción para el estado actual
            action_values = np.zeros(self.n_actions)
            
            for a in range(self.n_actions):
                for prob, next_s, r, done in env.P[s][a]:
                    action_values[a] += prob * (r + self.gamma * V[next_s] * (not done))
            
            # Obtener la nueva mejor acción (con valor máximo)
            best_action = np.argmax(action_values)
            
            # Actualizar política: determinística (probabilidad 1.0 para la mejor acción)
            policy[s, best_action] = 1.0
            
            # Verificar si la política cambió
            if old_action != best_action:
                policy_stable = False
        
        return policy, policy_stable
    
    def train(self, env) -> Dict[str, List]:
        """
        Entrena al agente usando iteración de política.
        
        Args:
            env: Entorno con dinámicas de transición
            
        Returns:
            Diccionario con historial de entrenamiento
        """
        print("Iniciando iteración de política...")
        
        policy_stable = False
        iterations = 0
        
        start_time = time.time()
        
        while not policy_stable and iterations < self.max_iterations:
            iteration_start = time.time()
            
            # Evaluación de Política: Calcular función de valor para la política actual
            self.V = self.policy_evaluation(env, self.policy)
            
            # Calcular cambio de valor para métricas
            if iterations > 0:
                value_change = np.mean(np.abs(self.V - old_V))
                self.value_changes.append(value_change)
            old_V = self.V.copy()
            
            # Mejora de Política: Actualizar política basada en nueva función de valor
            new_policy, policy_stable = self.policy_improvement(env, self.V)
            
            # Calcular cambio de política para métricas
            if iterations > 0:
                policy_change = np.sum(np.abs(new_policy - self.policy)) / (2 * self.n_states)
                self.policy_changes.append(policy_change)
            
            self.policy = new_policy
            iterations += 1
            
            # Registrar tiempo transcurrido
            iteration_time = time.time() - iteration_start
            self.policy_iteration_times.append(iteration_time)
            
            print(f"Iteración {iterations}: {iteration_time:.2f} segundos, " + 
                  f"Iteraciones de evaluación: {self.eval_iteration_counts[-1]}")
            
            if policy_stable:
                print("¡Política convergida!")
        
        total_time = time.time() - start_time
        print(f"Iteración de política completada en {iterations} iteraciones, {total_time:.2f} segundos")
        
        history = {
            'iterations': iterations,
            'policy_changes': self.policy_changes,
            'value_changes': self.value_changes,
            'iteration_times': self.policy_iteration_times,
            'eval_iterations': self.eval_iteration_counts,
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
        
        for _ in range(episodes):
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
    
    def visualize_policy(self, env, title: str = "Política"):
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
    
    def visualize_training(self, history: Dict[str, List]) -> None:
        """
        Visualiza las métricas de entrenamiento.
        
        Args:
            history: Diccionario con historial de entrenamiento
        """
        _, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Gráfico de cambios en la política
        if history['policy_changes']:
            axs[0, 0].set_title('Cambios en la Política')
            axs[0, 0].set_xlabel(self.ITERATION_LABEL)
            axs[0, 0].set_ylabel('Cambio de Política')
            axs[0, 0].grid(True)
            axs[0, 0].set_ylabel('Cambio de Política')
            axs[0, 0].grid(True)
        
        # Gráfico de cambios en la función de valor
        if history['value_changes']:
            axs[0, 1].set_title('Cambios en la Función de Valor')
            axs[0, 1].set_xlabel(self.ITERATION_LABEL)
            axs[0, 1].set_ylabel('Cambio Promedio de Valor')
            axs[0, 1].grid(True)
            axs[0, 1].set_ylabel('Cambio Promedio de Valor')
            axs[0, 1].grid(True)
        
        # Gráfico de tiempos de iteración
        axs[1, 0].set_title('Tiempos de Iteración')
        axs[1, 0].set_xlabel(self.ITERATION_LABEL)
        axs[1, 0].set_ylabel('Tiempo (segundos)')
        axs[1, 0].grid(True)
        axs[1, 0].set_ylabel('Tiempo (segundos)')
        axs[1, 0].grid(True)
        
        # Gráfico de iteraciones de evaluación
        axs[1, 1].plot(range(1, len(history['eval_iterations']) + 1), 
                      history['eval_iterations'])
        axs[1, 1].set_title('Iteraciones de Evaluación de Política')
        axs[1, 1].set_xlabel('Iteración de Política')
        axs[1, 1].set_ylabel('Número de Iteraciones')
        axs[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()