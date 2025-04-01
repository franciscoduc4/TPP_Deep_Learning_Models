import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Optional, Union
import pickle
from ..config import MONTE_CARLO_CONFIG

class MonteCarlo:
    """
    Implementación de métodos Monte Carlo para predicción y control en aprendizaje por refuerzo.
    
    Esta clase proporciona implementaciones de:
    1. Predicción Monte Carlo (first-visit y every-visit) para evaluar políticas
    2. Control Monte Carlo (on-policy y off-policy) para encontrar políticas óptimas
    
    Se incluyen algoritmos como:
    - First-visit MC prediction
    - Every-visit MC prediction
    - Monte Carlo Exploring Starts (MCES)
    - On-policy MC control con epsilon-greedy
    - Off-policy MC control con importance sampling
    """
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float = MONTE_CARLO_CONFIG['gamma'],
        epsilon_start: float = MONTE_CARLO_CONFIG['epsilon_start'],
        epsilon_end: float = MONTE_CARLO_CONFIG['epsilon_end'],
        epsilon_decay: float = MONTE_CARLO_CONFIG['epsilon_decay'],
        first_visit: bool = MONTE_CARLO_CONFIG['first_visit'],
        evaluation_mode: bool = False
    ):
        """
        Inicializa el agente de Monte Carlo.
        
        Args:
            n_states: Número de estados en el entorno
            n_actions: Número de acciones en el entorno
            gamma: Factor de descuento para recompensas futuras
            epsilon_start: Valor inicial de epsilon para políticas epsilon-greedy
            epsilon_end: Valor mínimo de epsilon
            epsilon_decay: Factor de decaimiento de epsilon
            first_visit: Si True, usa first-visit MC, sino every-visit MC
            evaluation_mode: Si True, inicializa en modo evaluación de política (sin control)
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.first_visit = first_visit
        self.evaluation_mode = evaluation_mode
        
        # Inicializar tablas de valor de acción (Q) y política
        self.q_table = np.zeros((n_states, n_actions))
        
        # Para modo de evaluación, la política es fija (proporcionada externamente)
        # Para control, comenzamos con una política epsilon-greedy derivada de Q
        self.policy = np.ones((n_states, n_actions)) / n_actions  # Inicialmente equiprobable
        
        # Contadores para calcular promedios incrementales
        self.returns_sum = np.zeros((n_states, n_actions))
        self.returns_count = np.zeros((n_states, n_actions))
        
        # Para evaluación de política (valor de estado)
        self.v_table = np.zeros(n_states)
        self.state_returns_sum = np.zeros(n_states)
        self.state_returns_count = np.zeros(n_states)
        
        # Para off-policy Monte Carlo
        self.c_table = np.zeros((n_states, n_actions))  # Pesos acumulativos para importance sampling
        
        # Métricas
        self.episode_rewards = []
        self.episode_lengths = []
        self.policy_changes = []
        self.value_changes = []
        self.epsilon_history = []
    
    def reset_counters(self):
        """
        Reinicia los contadores de retornos para un nuevo entrenamiento.
        """
        self.returns_sum = np.zeros((self.n_states, self.n_actions))
        self.returns_count = np.zeros((self.n_states, self.n_actions))
        self.state_returns_sum = np.zeros(self.n_states)
        self.state_returns_count = np.zeros(self.n_states)
        self.c_table = np.zeros((self.n_states, self.n_actions))
    
    def get_action(self, state: int, explore: bool = True) -> int:
        """
        Selecciona una acción según la política actual, con exploración opcional.
        
        Args:
            state: Estado actual
            explore: Si es True, usa política epsilon-greedy; si es False, usa política greedy
            
        Returns:
            La acción seleccionada
        """
        if explore and np.random.random() < self.epsilon:
            # Exploración: acción aleatoria
            return np.random.randint(self.n_actions)
        else:
            # Explotación: mejor acción según la política actual
            if self.evaluation_mode:
                # En modo evaluación, usamos la política directamente
                action_probs = self.policy[state]
                return np.random.choice(self.n_actions, p=action_probs)
            else:
                # En modo control, derivamos la acción greedy de los valores Q
                return np.argmax(self.q_table[state])
    
    def update_policy(self, state: int):
        """
        Actualiza la política para el estado dado basándose en los valores Q actuales.
        
        Args:
            state: Estado para el cual actualizar la política
            
        Returns:
            Boolean indicando si la política cambió
        """
        if self.evaluation_mode:
            # En modo evaluación, no se actualiza la política
            return False
        
        old_action = np.argmax(self.policy[state])
        best_action = np.argmax(self.q_table[state])
        
        # Política epsilon-greedy basada en Q
        self.policy[state] = np.zeros(self.n_actions)
        
        # Probabilidad pequeña de exploración
        self.policy[state] += self.epsilon / self.n_actions
        
        # Mayor probabilidad para la mejor acción
        self.policy[state][best_action] += (1 - self.epsilon)
        
        return old_action != best_action
    
    def decay_epsilon(self, episode: int = None):
        """
        Reduce el valor de epsilon según la estrategia de decaimiento.
        
        Args:
            episode: Número del episodio actual (para decaimientos basados en episodios)
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)
    
    def calculate_returns(self, rewards: list) -> list:
        """
        Calcula los retornos descontados para cada paso de tiempo en un episodio.
        
        Args:
            rewards: Lista de recompensas recibidas durante el episodio
            
        Returns:
            Lista de retornos (G_t) para cada paso de tiempo
        """
        returns = np.zeros(len(rewards))
        G = 0
        
        # Recorremos las recompensas en orden inverso
        for t in range(len(rewards) - 1, -1, -1):
            G = rewards[t] + self.gamma * G
            returns[t] = G
            
        return returns
    
    def monte_carlo_prediction(self, episodes: List[Tuple[List[int], List[int], List[float]]]):
        """
        Realiza predicción Monte Carlo (evaluación de política) usando episodios proporcionados.
        
        Args:
            episodes: Lista de episodios, cada uno como una tupla de 
                     (estados, acciones, recompensas)
        
        Returns:
            v_table actualizada (función de valor de estado)
        """
        old_v = self.v_table.copy()
        
        for states, actions, rewards in episodes:
            returns = self.calculate_returns(rewards)
            
            # Procesar cada paso en el episodio
            visited_state_steps = set()
            
            for t in range(len(states)):
                state = states[t]
                
                # Para first-visit MC, solo consideramos la primera visita a cada estado
                if self.first_visit and state in visited_state_steps:
                    continue
                    
                visited_state_steps.add(state)
                
                # Actualizar el conteo y la suma de retornos para este estado
                self.state_returns_sum[state] += returns[t]
                self.state_returns_count[state] += 1
                
                # Actualizar la función de valor usando promedio incremental
                if self.state_returns_count[state] > 0:
                    self.v_table[state] = self.state_returns_sum[state] / self.state_returns_count[state]
        
        # Calcular cambio en la función de valor
        value_change = np.mean(np.abs(self.v_table - old_v))
        self.value_changes.append(value_change)
        
        return self.v_table
    
    def monte_carlo_control_on_policy(self, env, episodes: int = MONTE_CARLO_CONFIG['episodes'], 
                                     max_steps: int = MONTE_CARLO_CONFIG['max_steps'],
                                     render: bool = False):
        """
        Implementa control Monte Carlo on-policy con epsilon-greedy.
        
        Args:
            env: Entorno de OpenAI Gym o similar
            episodes: Número de episodios a ejecutar
            max_steps: Número máximo de pasos por episodio
            render: Si renderizar o no el entorno
            
        Returns:
            Historia del entrenamiento
        """
        start_time = time.time()
        
        for episode in range(episodes):
            # Inicializar episodio
            state, _ = env.reset()
            episode_states = []
            episode_actions = []
            episode_rewards = []
            
            # Ejecutar un episodio completo
            for t in range(max_steps):
                if render:
                    env.render()
                
                # Seleccionar acción según política epsilon-greedy
                action = self.get_action(state, explore=True)
                
                # Dar un paso en el entorno
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Guardar transición
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)
                
                # Actualizar estado
                state = next_state
                
                if done:
                    break
            
            # Calcular retornos para el episodio
            returns = self.calculate_returns(episode_rewards)
            
            # Actualizar función de valor de acción (Q)
            policy_changed = False
            visited_state_action_pairs = set()
            
            for t in range(len(episode_states)):
                state = episode_states[t]
                action = episode_actions[t]
                
                # Para first-visit MC, solo considerar primera visita a cada par estado-acción
                state_action = (state, action)
                if self.first_visit and state_action in visited_state_action_pairs:
                    continue
                
                visited_state_action_pairs.add(state_action)
                
                # Actualizar conteos y sumas para este par estado-acción
                self.returns_sum[state, action] += returns[t]
                self.returns_count[state, action] += 1
                
                # Actualizar valor Q usando promedio incremental
                if self.returns_count[state, action] > 0:
                    self.q_table[state, action] = self.returns_sum[state, action] / self.returns_count[state, action]
                    
                    # Actualizar política basada en nuevo valor Q
                    if self.update_policy(state):
                        policy_changed = True
            
            # Registrar métricas del episodio
            self.episode_rewards.append(sum(episode_rewards))
            self.episode_lengths.append(len(episode_rewards))
            self.policy_changes.append(1 if policy_changed else 0)
            
            # Decaer epsilon
            self.decay_epsilon(episode)
            
            # Mostrar progreso periódicamente
            if (episode + 1) % MONTE_CARLO_CONFIG['log_interval'] == 0 or episode == 0:
                avg_reward = np.mean(self.episode_rewards[-MONTE_CARLO_CONFIG['log_interval']:])
                elapsed_time = time.time() - start_time
                
                print(f"Episodio {episode+1}/{episodes} - Recompensa promedio: {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.4f}, Tiempo: {elapsed_time:.2f}s")
        
        # Crear historial de entrenamiento
        history = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'policy_changes': self.policy_changes,
            'value_changes': self.value_changes,
            'epsilon_history': self.epsilon_history,
            'training_time': time.time() - start_time
        }
        
        return history
    
    def monte_carlo_control_off_policy(self, env, episodes: int = MONTE_CARLO_CONFIG['episodes'], 
                                      max_steps: int = MONTE_CARLO_CONFIG['max_steps'],
                                      render: bool = False):
        """
        Implementa control Monte Carlo off-policy con importance sampling.
        
        Args:
            env: Entorno de OpenAI Gym o similar
            episodes: Número de episodios a ejecutar
            max_steps: Número máximo de pasos por episodio
            render: Si renderizar o no el entorno
            
        Returns:
            Historia del entrenamiento
        """
        start_time = time.time()
        
        # Política de comportamiento (behavior policy) - más exploratoria
        behavior_epsilon = max(0.1, self.epsilon)
        
        for episode in range(episodes):
            # Inicializar episodio
            state, _ = env.reset()
            episode_states = []
            episode_actions = []
            episode_rewards = []
            episode_behavior_probs = []  # Probabilidades bajo política de comportamiento
            
            # Ejecutar un episodio completo usando política de comportamiento
            for t in range(max_steps):
                if render:
                    env.render()
                
                # Seleccionar acción usando política de comportamiento (más exploratoria)
                if np.random.random() < behavior_epsilon:
                    action = np.random.randint(self.n_actions)
                    behavior_prob = behavior_epsilon / self.n_actions
                else:
                    action = np.argmax(self.q_table[state])
                    behavior_prob = 1 - behavior_epsilon + (behavior_epsilon / self.n_actions)
                
                # Dar un paso en el entorno
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Guardar transición
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)
                episode_behavior_probs.append(behavior_prob)
                
                # Actualizar estado
                state = next_state
                
                if done:
                    break
            
            # Calcular retornos para el episodio
            returns = self.calculate_returns(episode_rewards)
            
            # Calcular ratios de importancia y actualizar Q
            G = 0.0
            W = 1.0  # Peso de importancia inicial
            
            # Recorremos el episodio en orden inverso (importante para OFF-policy MC)
            for t in range(len(episode_states) - 1, -1, -1):
                state = episode_states[t]
                action = episode_actions[t]
                reward = episode_rewards[t]
                
                # Actualizar retorno acumulado
                G = reward + self.gamma * G
                
                # Actualizar contador de visitas para este par estado-acción
                self.c_table[state, action] += W
                
                # Actualizar función Q usando importance sampling
                self.q_table[state, action] += (W / self.c_table[state, action]) * (G - self.q_table[state, action])
                
                # Actualizar política target (greedy respecto a Q)
                self.update_policy(state)
                
                # Obtener probabilidad bajo policy target (greedy)
                target_policy_prob = 1.0 if action == np.argmax(self.q_table[state]) else 0.0
                
                # Actualizar ratio de importancia
                if target_policy_prob == 0.0:
                    # Si la acción no sería elegida por la política target,
                    # terminamos el procesamiento de este episodio
                    break
                
                W *= target_policy_prob / episode_behavior_probs[t]
            
            # Registrar métricas del episodio
            self.episode_rewards.append(sum(episode_rewards))
            self.episode_lengths.append(len(episode_rewards))
            
            # Decaer epsilon
            self.decay_epsilon(episode)
            
            # Mostrar progreso periódicamente
            if (episode + 1) % MONTE_CARLO_CONFIG['log_interval'] == 0 or episode == 0:
                avg_reward = np.mean(self.episode_rewards[-MONTE_CARLO_CONFIG['log_interval']:])
                elapsed_time = time.time() - start_time
                
                print(f"Episodio {episode+1}/{episodes} - Recompensa promedio: {avg_reward:.2f}, "
                      f"Tiempo: {elapsed_time:.2f}s")
        
        # Crear historial de entrenamiento
        history = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'epsilon_history': self.epsilon_history,
            'training_time': time.time() - start_time
        }
        
        return history
    
    def monte_carlo_exploring_starts(self, env, episodes: int = MONTE_CARLO_CONFIG['episodes'], max_steps: int = MONTE_CARLO_CONFIG['max_steps'], render: bool = False):
        """
        Implementa control Monte Carlo con exploring starts (MCES).
        
        Nota: Este método solo funciona para entornos que permiten establecer el estado inicial.
        
        Args:
            env: Entorno de OpenAI Gym o similar con soporte para establecer estado
            episodes: Número de episodios a ejecutar
            max_steps: Número máximo de pasos por episodio
            render: Si renderizar o no el entorno
            
        Returns:
            Historia del entrenamiento
        """
        start_time = time.time()
        
        # Verificar si el entorno soporta establecer estados
        if not hasattr(env, 'set_state'):
            print("Advertencia: Este entorno no parece soportar 'set_state'. El método MCES puede no funcionar correctamente.")
        
        for episode in range(episodes):
            # Iniciar con un estado aleatorio (exploring start)
            if hasattr(env, 'set_state'):
                random_state = np.random.randint(0, self.n_states)
                env.set_state(random_state)
                state = random_state
            else:
                # Si no podemos establecer el estado, iniciamos normalmente
                state, _ = env.reset()
                
            # Seleccionar una primera acción aleatoria para garantizar exploring starts
            action = np.random.randint(0, self.n_actions)
            
            # Ejecutar la primera acción
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Inicializar listas para guardar la trayectoria
            episode_states = [state]
            episode_actions = [action]
            episode_rewards = [reward]
            
            # Continuar el episodio usando la política actual
            state = next_state
            steps = 1
            
            while not done and steps < max_steps:
                if render:
                    env.render()
                
                # Seleccionar acción según política actual (sin exploración adicional)
                action = self.get_action(state, explore=False)
                
                # Dar un paso en el entorno
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Guardar transición
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)
                
                # Actualizar estado
                state = next_state
                steps += 1
                
                if done:
                    break
            
            # Calcular retornos para el episodio
            returns = self.calculate_returns(episode_rewards)
            
            # Actualizar función Q y política
            policy_changed = False
            visited_state_action_pairs = set()
            
            for t in range(len(episode_states)):
                state = episode_states[t]
                action = episode_actions[t]
                
                # Para first-visit MC, solo considerar primera visita a cada par estado-acción
                state_action = (state, action)
                if self.first_visit and state_action in visited_state_action_pairs:
                    continue
                
                visited_state_action_pairs.add(state_action)
                
                # Actualizar conteos y sumas para este par estado-acción
                self.returns_sum[state, action] += returns[t]
                self.returns_count[state, action] += 1
                
                # Actualizar valor Q usando promedio incremental
                if self.returns_count[state, action] > 0:
                    self.q_table[state, action] = self.returns_sum[state, action] / self.returns_count[state, action]
                    
                    # Actualizar política (determinística para MCES)
                    old_action = np.argmax(self.policy[state])
                    best_action = np.argmax(self.q_table[state])
                    
                    if old_action != best_action:
                        policy_changed = True
                        # En MCES, la política es totalmente greedy (determinística)
                        self.policy[state] = np.zeros(self.n_actions)
                        self.policy[state][best_action] = 1.0
            
            # Registrar métricas del episodio
            self.episode_rewards.append(sum(episode_rewards))
            self.episode_lengths.append(len(episode_rewards))
            self.policy_changes.append(1 if policy_changed else 0)
            
            # Mostrar progreso periódicamente
            if (episode + 1) % MONTE_CARLO_CONFIG['log_interval'] == 0 or episode == 0:
                avg_reward = np.mean(self.episode_rewards[-MONTE_CARLO_CONFIG['log_interval']:])
                elapsed_time = time.time() - start_time
                
                print(f"Episodio {episode+1}/{episodes} - Recompensa promedio: {avg_reward:.2f}, "
                      f"Tiempo: {elapsed_time:.2f}s")
        
        # Crear historial de entrenamiento
        history = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'policy_changes': self.policy_changes,
            'training_time': time.time() - start_time
        }
        
        return history
    
    def evaluate(self, env, episodes: int = 10, max_steps: int = 1000, render: bool = False) -> float:
        """
        Evalúa la política actual en el entorno.
        
        Args:
            env: Entorno para evaluar
            episodes: Número de episodios para la evaluación
            max_steps: Máximo número de pasos por episodio
            render: Si mostrar o no la visualización del entorno
            
        Returns:
            Recompensa promedio por episodio
        """
        total_rewards = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done and steps < max_steps:
                if render:
                    env.render()
                
                # Seleccionar acción según la política actual, sin exploración
                action = self.get_action(state, explore=False)
                
                # Ejecutar acción
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Actualizar
                total_reward += reward
                state = next_state
                steps += 1
            
            total_rewards.append(total_reward)
            print(f"Episodio {episode+1}: Recompensa = {total_reward}, Pasos = {steps}")
        
        avg_reward = np.mean(total_rewards)
        print(f"Evaluación: Recompensa promedio = {avg_reward:.2f}")
        
        return avg_reward
    
    def save(self, filepath: str) -> None:
        """
        Guarda el modelo (tablas Q, política, etc.) en un archivo.
        
        Args:
            filepath: Ruta donde guardar el modelo
        """
        data = {
            'q_table': self.q_table,
            'policy': self.policy,
            'v_table': self.v_table,
            'n_states': self.n_states,
            'n_actions': self.n_actions,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'first_visit': self.first_visit,
            'evaluation_mode': self.evaluation_mode
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Modelo guardado en {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Carga el modelo desde un archivo.
        
        Args:
            filepath: Ruta desde donde cargar el modelo
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.q_table = data['q_table']
        self.policy = data['policy']
        self.v_table = data['v_table']
        self.n_states = data['n_states']
        self.n_actions = data['n_actions']
        self.gamma = data['gamma']
        self.epsilon = data['epsilon']
        self.first_visit = data['first_visit']
        self.evaluation_mode = data['evaluation_mode']
        
        print(f"Modelo cargado desde {filepath}")
    
    def visualize_policy(self, env, title: str = "Política") -> None:
        """
        Visualiza la política actual para entornos tipo cuadrícula.
        
        Args:
            env: Entorno con estructura de cuadrícula
            title: Título para la visualización
        """
        if not hasattr(env, 'shape'):
            print("El entorno no tiene estructura de cuadrícula para visualización")
            return
        
        grid_shape = env.shape
        fig, ax = plt.subplots(figsize=(8, 8))
        
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
            
            # Evitar estados terminales
            if hasattr(env, 'is_terminal') and env.is_terminal(s):
                continue
                
            if self.evaluation_mode:
                # En modo evaluación, mostramos la acción con mayor probabilidad
                action = np.argmax(self.policy[s])
            else:
                # En modo control, mostramos la acción con mayor valor Q
                action = np.argmax(self.q_table[s])
            
            # Definir direcciones de flechas (estas pueden variar según el entorno)
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
        
        # Mostrar valores Q o V
        for s in range(self.n_states):
            if hasattr(env, 'state_mapping'):
                i, j = env.state_mapping(s)
            else:
                i, j = s // grid_shape[1], s % grid_shape[1]
            
            if self.evaluation_mode:
                value = self.v_table[s]
            else:
                value = np.max(self.q_table[s])
            
            ax.text(j + 0.5, grid_shape[0] - i - 0.5, f"{value:.2f}", 
                   ha='center', va='center', color='red', fontsize=9)
        
        ax.set_title(title)
        plt.tight_layout()
        plt.show()
    
    def visualize_value_function(self, env, title: str = "Función de Valor"):
        """
        Visualiza la función de valor para entornos tipo cuadrícula.
        
        Args:
            env: Entorno con estructura de cuadrícula
            title: Título para la visualización
        """
        if not hasattr(env, 'shape'):
            print("El entorno no tiene estructura de cuadrícula para visualización")
            return
        
        grid_shape = env.shape
        
        # Crear matriz para visualización
        if self.evaluation_mode:
            value_grid = np.zeros(grid_shape)
        else:
            # Para control, usamos el máximo valor Q en cada estado
            value_grid = np.zeros(grid_shape)
        
        # Llenar matriz con valores
        for s in range(self.n_states):
            if hasattr(env, 'state_mapping'):
                i, j = env.state_mapping(s)
            else:
                i, j = s // grid_shape[1], s % grid_shape[1]
                
            if self.evaluation_mode:
                value_grid[i, j] = self.v_table[s]
            else:
                value_grid[i, j] = np.max(self.q_table[s])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
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
    
    def visualize_training(self, history: Dict[str, List] = None):
        """
        Visualiza métricas de entrenamiento.
        
        Args:
            history: Diccionario con historial de entrenamiento (opcional)
        """
        if history is None:
            # Si no se proporciona historia, usar datos internos
            history = {
                'episode_rewards': self.episode_rewards,
                'episode_lengths': self.episode_lengths,
                'policy_changes': self.policy_changes,
                'value_changes': self.value_changes if len(self.value_changes) > 0 else None,
                'epsilon_history': self.epsilon_history
            }
        
        # Configuración de la figura
        n_plots = 3
        if 'value_changes' in history and history['value_changes']:
            n_plots += 1
        if 'policy_changes' in history and history['policy_changes']:
            n_plots += 1
            
        fig, axs = plt.subplots(n_plots, 1, figsize=(12, 3*n_plots))
        
        plot_idx = 0
        
        # Gráfico de recompensas
        axs[plot_idx].plot(history['episode_rewards'])
        axs[plot_idx].set_title('Recompensas por Episodio')
        axs[plot_idx].set_xlabel('Episodio')
        axs[plot_idx].set_ylabel('Recompensa Total')
        axs[plot_idx].grid(True)
        
        # Suavizar curva para mejor visualización
        window_size = min(len(history['episode_rewards']) // 10 + 1, 100)
        if window_size > 1:
            smoothed = np.convolve(history['episode_rewards'], 
                                  np.ones(window_size)/window_size, mode='valid')
            axs[plot_idx].plot(range(window_size-1, len(history['episode_rewards'])), 
                              smoothed, 'r-', linewidth=2, label=f'Suavizado (ventana={window_size})')
            axs[plot_idx].legend()
        
        plot_idx += 1
        
        # Gráfico de longitud de episodios
        axs[plot_idx].plot(history['episode_lengths'])
        axs[plot_idx].set_title('Longitud de Episodios')
        axs[plot_idx].set_xlabel('Episodio')
        axs[plot_idx].set_ylabel('Pasos')
        axs[plot_idx].grid(True)
        
        plot_idx += 1
        
        # Gráfico de epsilon (si existe)
        if 'epsilon_history' in history and history['epsilon_history']:
            axs[plot_idx].plot(history['epsilon_history'])
            axs[plot_idx].set_title('Epsilon (Exploración)')
            axs[plot_idx].set_xlabel('Episodio')
            axs[plot_idx].set_ylabel('Epsilon')
            axs[plot_idx].grid(True)
            
            plot_idx += 1
        
        # Gráfico de cambios en la política (si existe)
        if 'policy_changes' in history and history['policy_changes']:
            axs[plot_idx].plot(history['policy_changes'])
            axs[plot_idx].set_title('Cambios en la Política')
            axs[plot_idx].set_xlabel('Episodio')
            axs[plot_idx].set_ylabel('Cambio (0=No, 1=Sí)')
            axs[plot_idx].grid(True)
            
            plot_idx += 1
        
        # Gráfico de cambios en valores (si existe)
        if 'value_changes' in history and history['value_changes']:
            axs[plot_idx].plot(history['value_changes'])
            axs[plot_idx].set_title('Cambios en Valores')
            axs[plot_idx].set_xlabel('Actualización')
            axs[plot_idx].set_ylabel('Cambio Promedio')
            axs[plot_idx].set_yscale('log')  # Escala logarítmica para ver mejor la convergencia
            axs[plot_idx].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def train(self, env, method: str = 'on_policy', episodes: int = None, max_steps: int = None,
             render: bool = False):
        """
        Método principal para entrenar el agente con el algoritmo Monte Carlo seleccionado.
        
        Args:
            env: Entorno para entrenar
            method: Método de entrenamiento ('on_policy', 'off_policy', 'exploring_starts')
            episodes: Número de episodios (si None, usa valor de configuración)
            max_steps: Pasos máximos por episodio (si None, usa valor de configuración)
            render: Si mostrar o no la visualización del entorno
            
        Returns:
            Historia de entrenamiento
        """
        if episodes is None:
            episodes = MONTE_CARLO_CONFIG['episodes']
        
        if max_steps is None:
            max_steps = MONTE_CARLO_CONFIG['max_steps']
        
        # Resetear contadores para nuevo entrenamiento
        self.reset_counters()
        
        # Seleccionar método de entrenamiento
        if method == 'on_policy':
            return self.monte_carlo_control_on_policy(env, episodes, max_steps, render)
        elif method == 'off_policy':
            return self.monte_carlo_control_off_policy(env, episodes, max_steps, render)
        elif method == 'exploring_starts':
            return self.monte_carlo_exploring_starts(env, episodes, max_steps, render)
        else:
            raise ValueError(f"Método desconocido: {method}. Use 'on_policy', 'off_policy' o 'exploring_starts'")
    
    def visualize_action_values(self, state: int, title: str = None):
        """
        Visualiza los valores Q para todas las acciones en un estado específico.
        
        Args:
            state: Estado para visualizar valores de acción
            title: Título opcional para el gráfico
        """
        if not title:
            title = f"Valores Q para el Estado {state}"
        
        actions = np.arange(self.n_actions)
        values = self.q_table[state]
        
        plt.figure(figsize=(10, 6))
        plt.bar(actions, values, color='skyblue')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Añadir valores encima de cada barra
        for i, v in enumerate(values):
            plt.text(i, v + 0.05, f"{v:.2f}", ha='center')
        
        # Resaltar la mejor acción
        best_action = np.argmax(values)
        plt.bar(best_action, values[best_action], color='green', label='Mejor Acción')
        
        plt.xlabel('Acciones')
        plt.ylabel('Valor Q')
        plt.title(title)
        plt.xticks(actions)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def compare_visits(self, env, episodes=100, max_steps=1000):
        """
        Compara first-visit y every-visit Monte Carlo para la evaluación de política.
        
        Args:
            env: Entorno para evaluar
            episodes: Número de episodios para la comparación
            max_steps: Pasos máximos por episodio
            
        Returns:
            Diccionario con resultados de la comparación
        """
        print("Comparando first-visit vs every-visit Monte Carlo...")
        
        # Guardar configuración original
        original_first_visit = self.first_visit
        
        # Crear agentes para comparar
        first_visit_agent = MonteCarlo(
            self.n_states, 
            self.n_actions,
            gamma=self.gamma,
            first_visit=True,
            evaluation_mode=True
        )
        
        every_visit_agent = MonteCarlo(
            self.n_states, 
            self.n_actions,
            gamma=self.gamma,
            first_visit=False,
            evaluation_mode=True
        )
        
        # Establecer la misma política para ambos agentes
        first_visit_agent.policy = self.policy.copy()
        every_visit_agent.policy = self.policy.copy()
        
        # Recopilar episodios
        collected_episodes = []
        for _ in range(episodes):
            state, _ = env.reset()
            states = []
            actions = []
            rewards = []
            done = False
            step = 0
            
            while not done and step < max_steps:
                # Elegir acción según la política actual
                action = self.get_action(state)
                
                # Ejecutar acción
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Guardar transición
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                
                # Actualizar
                state = next_state
                step += 1
            
            collected_episodes.append((states, actions, rewards))
        
        # Evaluar política con ambos métodos
        first_v = first_visit_agent.monte_carlo_prediction(collected_episodes)
        every_v = every_visit_agent.monte_carlo_prediction(collected_episodes)
        
        # Calcular diferencias
        diff = np.abs(first_v - every_v)
        mean_diff = np.mean(diff)
        max_diff = np.max(diff)
        
        # Visualizar diferencias
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 3, 1)
        plt.plot(first_v, label='First-visit')
        plt.plot(every_v, label='Every-visit')
        plt.xlabel('Estado')
        plt.ylabel('Valor')
        plt.title('Comparación de Valores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.plot(diff)
        plt.xlabel('Estado')
        plt.ylabel('Diferencia Absoluta')
        plt.title(f'Diferencia (Media: {mean_diff:.4f})')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        plt.hist(diff, bins=20)
        plt.xlabel('Diferencia')
        plt.ylabel('Frecuencia')
        plt.title('Histograma de Diferencias')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Restaurar configuración original
        self.first_visit = original_first_visit
        
        return {
            'first_visit': first_v,
            'every_visit': every_v,
            'mean_diff': mean_diff,
            'max_diff': max_diff
        }
    
    def weighted_importance_sampling(self, env, episodes=MONTE_CARLO_CONFIG['episodes'], 
                                   max_steps=MONTE_CARLO_CONFIG['max_steps'], 
                                   render=False):
        """
        Implementa control Monte Carlo off-policy con weighted importance sampling.
        Este método tiende a ser más estable que el importance sampling ordinario.
        
        Args:
            env: Entorno de OpenAI Gym o similar
            episodes: Número de episodios a ejecutar
            max_steps: Número máximo de pasos por episodio
            render: Si renderizar o no el entorno
            
        Returns:
            Historia del entrenamiento
        """
        start_time = time.time()
        
        # Inicializar matrices para weighted importance sampling
        self.q_table = np.zeros((self.n_states, self.n_actions))  # Valores Q
        self.c_table = np.zeros((self.n_states, self.n_actions))  # Pesos acumulados
        
        # Política de comportamiento (behavior policy) - más exploratoria
        behavior_epsilon = max(0.1, self.epsilon)
        
        for episode in range(episodes):
            # Inicializar episodio
            state, _ = env.reset()
            episode_states = []
            episode_actions = []
            episode_rewards = []
            episode_behavior_probs = []  # Probabilidades bajo política de comportamiento
            
            # Ejecutar un episodio completo usando política de comportamiento
            for t in range(max_steps):
                if render:
                    env.render()
                
                # Seleccionar acción usando política de comportamiento (más exploratoria)
                if np.random.random() < behavior_epsilon:
                    action = np.random.randint(self.n_actions)
                    behavior_prob = behavior_epsilon / self.n_actions
                else:
                    action = np.argmax(self.q_table[state])
                    behavior_prob = 1 - behavior_epsilon + (behavior_epsilon / self.n_actions)
                
                # Dar un paso en el entorno
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Guardar transición
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)
                episode_behavior_probs.append(behavior_prob)
                
                # Actualizar estado
                state = next_state
                
                if done:
                    break
            
            # Calcular retornos para el episodio
            returns = self.calculate_returns(episode_rewards)
            
            # Procesar el episodio en orden inverso
            G = 0.0
            W = 1.0  # Peso de importancia inicial
            
            for t in range(len(episode_states) - 1, -1, -1):
                state = episode_states[t]
                action = episode_actions[t]
                reward = episode_rewards[t]
                
                # Actualizar retorno acumulado
                G = reward + self.gamma * G
                
                # Incrementar el contador de visitas con el peso de importancia
                self.c_table[state, action] += W
                
                # Actualizar valor Q usando weighted importance sampling
                if self.c_table[state, action] > 0:
                    # Fórmula de weighted importance sampling: Q += W/C * (G - Q)
                    self.q_table[state, action] += (W / self.c_table[state, action]) * (G - self.q_table[state, action])
                
                # Actualizar política (greedy respecto a Q)
                for a in range(self.n_actions):
                    if a == np.argmax(self.q_table[state]):
                        self.policy[state, a] = 1.0
                    else:
                        self.policy[state, a] = 0.0
                
                # Si la acción no habría sido tomada por la política target, detenemos la actualización
                if action != np.argmax(self.q_table[state]):
                    break
                
                # Actualizar ratio de importancia
                target_prob = 1.0  # Política greedy
                W *= target_prob / episode_behavior_probs[t]
            
            # Registrar métricas del episodio
            self.episode_rewards.append(sum(episode_rewards))
            self.episode_lengths.append(len(episode_rewards))
            
            # Mostrar progreso periódicamente
            if (episode + 1) % MONTE_CARLO_CONFIG['log_interval'] == 0 or episode == 0:
                avg_reward = np.mean(self.episode_rewards[-MONTE_CARLO_CONFIG['log_interval']:])
                elapsed_time = time.time() - start_time
                
                print(f"Episodio {episode+1}/{episodes} - Recompensa promedio: {avg_reward:.2f}, "
                      f"Tiempo: {elapsed_time:.2f}s")
        
        # Crear historial de entrenamiento
        history = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'training_time': time.time() - start_time
        }
        
        return history
    
    def incremental_monte_carlo(self, env, episodes=MONTE_CARLO_CONFIG['episodes'], 
                               max_steps=MONTE_CARLO_CONFIG['max_steps'],
                               render=False):
        """
        Implementa una versión incremental de Monte Carlo control que actualiza
        los valores Q después de cada paso en lugar de al final del episodio.
        
        Args:
            env: Entorno de OpenAI Gym o similar
            episodes: Número de episodios a ejecutar
            max_steps: Número máximo de pasos por episodio
            render: Si renderizar o no el entorno
            
        Returns:
            Historia del entrenamiento
        """
        start_time = time.time()
        
        for episode in range(episodes):
            # Inicializar episodio
            state, _ = env.reset()
            done = False
            step = 0
            
            # Inicializar buffer para el episodio actual
            episode_buffer = []
            
            # Ejecutar un episodio completo
            while not done and step < max_steps:
                if render:
                    env.render()
                
                # Seleccionar acción según política epsilon-greedy
                action = self.get_action(state, explore=True)
                
                # Dar un paso en el entorno
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Guardar transición en buffer
                episode_buffer.append((state, action, reward))
                
                # Actualizar estado
                state = next_state
                step += 1
            
            # Procesar el episodio y actualizar valores Q
            G = 0
            for t in range(len(episode_buffer) - 1, -1, -1):
                state, action, reward = episode_buffer[t]
                
                # Actualizar retorno acumulado
                G = reward + self.gamma * G
                
                # Verificar si es primera visita (si es necesario)
                if self.first_visit:
                    first_occurrence = True
                    for i in range(t):
                        if episode_buffer[i][0] == state and episode_buffer[i][1] == action:
                            first_occurrence = False
                            break
                    
                    if not first_occurrence:
                        continue
                
                # Actualizar contadores y valores Q
                self.returns_count[state, action] += 1
                
                # Actualización incremental:
                # Q(s,a) = Q(s,a) + (1/N) * (G - Q(s,a))
                # donde N es el número de visitas a (s,a)
                alpha = 1.0 / self.returns_count[state, action]
                self.q_table[state, action] += alpha * (G - self.q_table[state, action])
                
                # Actualizar política basada en nuevos valores Q
                self.update_policy(state)
            
            # Registrar métricas del episodio
            self.episode_rewards.append(sum(r for _, _, r in episode_buffer))
            self.episode_lengths.append(len(episode_buffer))
            
            # Decaer epsilon
            self.decay_epsilon(episode)
            
            # Mostrar progreso periódicamente
            if (episode + 1) % MONTE_CARLO_CONFIG['log_interval'] == 0 or episode == 0:
                avg_reward = np.mean(self.episode_rewards[-MONTE_CARLO_CONFIG['log_interval']:])
                elapsed_time = time.time() - start_time
                
                print(f"Episodio {episode+1}/{episodes} - Recompensa promedio: {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.4f}, Tiempo: {elapsed_time:.2f}s")
        
        # Crear historial de entrenamiento
        history = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'epsilon_history': self.epsilon_history,
            'training_time': time.time() - start_time
        }
        
        return history
    
    def batch_monte_carlo(self, env, batch_size=10, iterations=MONTE_CARLO_CONFIG['episodes'] // 10, 
                         max_steps=MONTE_CARLO_CONFIG['max_steps'], render=False):
        """
        Implementación de Monte Carlo por lotes, donde los valores Q son actualizados
        después de recopilar múltiples episodios.
        
        Args:
            env: Entorno de OpenAI Gym o similar
            batch_size: Número de episodios por lote
            iterations: Número de iteraciones (lotes) a ejecutar
            max_steps: Número máximo de pasos por episodio
            render: Si renderizar o no el entorno
            
        Returns:
            Historia del entrenamiento
        """
        start_time = time.time()
        
        for iteration in range(iterations):
            # Recopilar un lote de episodios
            batch_episodes = []
            batch_rewards_sum = 0
            batch_steps_sum = 0
            
            for _ in range(batch_size):
                # Recopilar un episodio
                state, _ = env.reset()
                episode_states = []
                episode_actions = []
                episode_rewards = []
                done = False
                step = 0
                
                while not done and step < max_steps:
                    if render:
                        env.render()
                    
                    # Seleccionar acción según política epsilon-greedy
                    action = self.get_action(state, explore=True)
                    
                    # Dar un paso en el entorno
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    
                    # Guardar transición
                    episode_states.append(state)
                    episode_actions.append(action)
                    episode_rewards.append(reward)
                    
                    # Actualizar estado
                    state = next_state
                    step += 1
                
                # Guardar episodio completo
                batch_episodes.append((episode_states, episode_actions, episode_rewards))
                batch_rewards_sum += sum(episode_rewards)
                batch_steps_sum += len(episode_rewards)
            
            # Procesar todos los episodios del lote
            policy_changed = False
            
            for states, actions, rewards in batch_episodes:
                # Calcular retornos
                returns = self.calculate_returns(rewards)
                
                # Actualizar valores Q
                visited_state_action_pairs = set()
                
                for t in range(len(states)):
                    state = states[t]
                    action = actions[t]
                    
                    # Para first-visit MC, solo considerar primera visita a cada par estado-acción
                    state_action = (state, action)
                    if self.first_visit and state_action in visited_state_action_pairs:
                        continue
                    
                    visited_state_action_pairs.add(state_action)
                    
                    # Actualizar conteos y sumas para este par estado-acción
                    self.returns_sum[state, action] += returns[t]
                    self.returns_count[state, action] += 1
                    
                    # Actualizar valor Q usando promedio incremental
                    if self.returns_count[state, action] > 0:
                        self.q_table[state, action] = self.returns_sum[state, action] / self.returns_count[state, action]
            
            # Actualizar política para todos los estados basada en nuevos valores Q
            for s in range(self.n_states):
                if self.update_policy(s):
                    policy_changed = True
            
            # Registrar métricas del lote
            self.episode_rewards.append(batch_rewards_sum / batch_size)
            self.episode_lengths.append(batch_steps_sum / batch_size)
            self.policy_changes.append(1 if policy_changed else 0)
            
            # Decaer epsilon
            self.decay_epsilon(iteration * batch_size)
            
            # Mostrar progreso
            avg_reward = self.episode_rewards[-1]
            elapsed_time = time.time() - start_time
            
            print(f"Iteración {iteration+1}/{iterations} - Recompensa promedio: {avg_reward:.2f}, "
                  f"Epsilon: {self.epsilon:.4f}, Tiempo: {elapsed_time:.2f}s")
        
        # Crear historial de entrenamiento
        history = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'policy_changes': self.policy_changes,
            'epsilon_history': self.epsilon_history,
            'training_time': time.time() - start_time
        }
        
        return history
    
    def visualize_importance_weights(self, env, episodes=10, max_steps=100):
        """
        Visualiza los pesos de importance sampling de Monte Carlo off-policy.
        
        Args:
            env: Entorno para recopilar datos
            episodes: Número de episodios para visualizar
            max_steps: Pasos máximos por episodio
        """
        # Asegurarse que tenemos pesos de importance sampling acumulados
        if np.sum(self.c_table) == 0:
            print("No hay pesos de importance sampling para visualizar. Ejecute monte_carlo_control_off_policy primero.")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Visualizar distribución de pesos
        plt.subplot(2, 2, 1)
        weights = self.c_table.flatten()
        weights = weights[weights > 0]  # Solo pesos positivos
        plt.hist(weights, bins=50)
        plt.title('Distribución de Pesos de Importance Sampling')
        plt.xlabel('Peso')
        plt.ylabel('Frecuencia')
        plt.grid(True, alpha=0.3)
        
        # Visualizar relación entre pesos y valores Q
        plt.subplot(2, 2, 2)
        x = []
        y = []
        for s in range(self.n_states):
            for a in range(self.n_actions):
                if self.c_table[s, a] > 0:
                    x.append(self.c_table[s, a])
                    y.append(self.q_table[s, a])
        
        plt.scatter(x, y, alpha=0.5)
        plt.title('Relación entre Pesos y Valores Q')
        plt.xlabel('Peso (C)')
        plt.ylabel('Valor Q')
        plt.xscale('log')  # Escala logarítmica para mejor visualización
        plt.grid(True, alpha=0.3)
        
        # Recopilar algunos episodios y visualizar la evolución de los pesos
        importance_weights = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            behavior_epsilon = 0.1  # Política de comportamiento más exploratoria
            
            # Política target (greedy)
            target_policy = np.zeros((self.n_states, self.n_actions))
            for s in range(self.n_states):
                target_policy[s, np.argmax(self.q_table[s])] = 1.0
                
            trajectory = []
            weights = []
            W = 1.0
            
            for t in range(max_steps):
                # Seleccionar acción usando política de comportamiento
                if np.random.random() < behavior_epsilon:
                    action = np.random.randint(self.n_actions)
                    behavior_prob = behavior_epsilon / self.n_actions
                else:
                    action = np.argmax(self.q_table[state])
                    behavior_prob = 1 - behavior_epsilon + (behavior_epsilon / self.n_actions)
                
                # Ejecutar acción
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Calcular probabilidad bajo política target
                target_prob = target_policy[state, action]
                
                # Actualizar peso de importancia
                if behavior_prob > 0:
                    W *= target_prob / behavior_prob
                
                # Guardar datos
                trajectory.append((state, action))
                weights.append(W)
                
                # Actualizar estado
                state = next_state
                
                if done or target_prob == 0:
                    break
            
            importance_weights.append(weights)
        
        # Visualizar evolución de pesos a lo largo de episodios
        plt.subplot(2, 2, 3)
        for i, weights in enumerate(importance_weights):
            plt.plot(weights, label=f'Episodio {i+1}' if i < 10 else None)
        
        plt.title('Evolución de Pesos de Importancia')
        plt.xlabel('Paso')
        plt.ylabel('Peso')
        plt.yscale('log')  # Escala logarítmica para mejor visualización
        plt.grid(True, alpha=0.3)
        if episodes <= 10:
            plt.legend()
        
        # Visualizar varianza de pesos entre episodios
        plt.subplot(2, 2, 4)
        max_len = max(len(w) for w in importance_weights)
        padded_weights = []
        
        # Rellenar con NaN para tener longitudes iguales
        for w in importance_weights:
            padded = w + [float('nan')] * (max_len - len(w))
            padded_weights.append(padded)
        
        weights_array = np.array(padded_weights)
        mean_weights = np.nanmean(weights_array, axis=0)
        std_weights = np.nanstd(weights_array, axis=0)
        
        steps = np.arange(max_len)
        plt.plot(steps, mean_weights, 'b-', label='Media')
        plt.fill_between(steps, mean_weights - std_weights, mean_weights + std_weights, 
                         color='b', alpha=0.2, label='Desviación Estándar')
        
        plt.title('Estadísticas de Pesos por Paso')
        plt.xlabel('Paso')
        plt.ylabel('Peso')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_convergence_comparison(self, env, methods=['on_policy', 'weighted', 'batch'], episodes=1000):
        """
        Compara la convergencia de diferentes métodos Monte Carlo en un mismo gráfico.
        
        Args:
            env: Entorno para la comparación
            methods: Lista de métodos a comparar
            episodes: Número de episodios para cada método
            
        Returns:
            Diccionario con historiales de entrenamiento
        """
        method_map = {
            'on_policy': (self.monte_carlo_control_on_policy, 'On-Policy MC', 'blue'),
            'off_policy': (self.monte_carlo_control_off_policy, 'Off-Policy MC', 'red'),
            'exploring_starts': (self.monte_carlo_exploring_starts, 'MCES', 'green'),
            'weighted': (self.weighted_importance_sampling, 'Weighted IS', 'purple'),
            'incremental': (self.incremental_monte_carlo, 'Incremental MC', 'orange'),
            'batch': (self.batch_monte_carlo, 'Batch MC', 'brown')
        }
        
        all_histories = {}
        
        plt.figure(figsize=(12, 6))
        
        for method_name in methods:
            if method_name not in method_map:
                print(f"Método desconocido: {method_name}")
                continue
                
            method_func, label, color = method_map[method_name]
            
            # Reiniciar el agente
            self.reset_counters()
            self.episode_rewards = []
            self.episode_lengths = []
            self.policy_changes = []
            self.value_changes = []
            self.epsilon_history = []
            self.epsilon = self.epsilon_start
            
            # Entrenar con este método
            print(f"\nEntrenando con método: {label}")
            if method_name == 'batch':
                # Batch necesita parámetros especiales
                batch_size = 10
                iterations = episodes // batch_size
                history = method_func(env, batch_size=batch_size, iterations=iterations, max_steps=MONTE_CARLO_CONFIG['max_steps'])
            else:
                history = method_func(env, episodes=episodes, max_steps=MONTE_CARLO_CONFIG['max_steps'])
            
            # Aplicar suavizado para mejor visualización
            window_size = min(len(history['episode_rewards']) // 10 + 1, 100)
            if window_size > 1 and len(history['episode_rewards']) > window_size:
                smoothed_rewards = np.convolve(
                    history['episode_rewards'], 
                    np.ones(window_size)/window_size, 
                    mode='valid'
                )
                x = range(window_size-1, len(history['episode_rewards']))
            else:
                smoothed_rewards = history['episode_rewards']
                x = range(len(smoothed_rewards))
            
            # Graficar resultados
            plt.plot(x, smoothed_rewards, color=color, label=f"{label}")
            
            # Guardar historia
            all_histories[method_name] = history
        
        plt.title('Comparación de Convergencia de Métodos Monte Carlo')
        plt.xlabel('Episodio')
        plt.ylabel('Recompensa Media (suavizada)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return all_histories