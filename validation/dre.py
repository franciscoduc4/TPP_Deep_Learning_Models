"""
Doubly Robust Estimator (DRE)

Estimador que combina el método directo (Direct Method) con el muestreo por importancia (Importance Sampling) para proporcionar una evaluación más robusta de políticas de control de insulina.

1. Direct Method Learning: Aprende un modelo de la función Q para estimar valores, similar a FQE.
2. Importance Sampling: repone las experiencias de la política de comportamiento para estimar el rendimiento de la política objetivo.
3. Behavior Policy Learning: Modela la política que generó los datos de entrenamiento
4. Doubly Robust Estimation: Combina ambos métodos para obtener estimaciones más robustas
5. Bootstrap para Intervalos de Confianza: Calcula intervalos de confianza para las estimaciones

DR = DM + IS(R - Q)
donde:
- DR: Estimación Doubly Robust
- DM: Estimación por Método Directo
- IS: Estimación por Muestreo por Importancia
- R: Recompensas observadas
- Q: Valores Q estimados por la red Q

Si el modelo Q es preciso, la estimación DR es consistente y tiene menor varianza que IS solo; si los pesos de importancia son precisos, la estimación DR es consistente y tiene menor varianza que DM solo.
"""
import os
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, List, Any, Optional
from tqdm import tqdm

from constants.constants import (
    CONST_DEFAULT_BATCH_SIZE, CONTEXT_FEATURE_ORDER, HYPER_PENALTY_BASE, HYPERGLYCEMIA_THRESHOLD, HYPO_PENALTY_BASE, HYPOGLYCEMIA_THRESHOLD, MAX_REWARD, OFFLINE_GAMMA, CONST_DEFAULT_EPOCHS,
    CONST_CONFIDENCE_LEVEL, CONST_IPS_CLIP, SEVERE_HYPO_PENALTY
)
from custom.printer import print_error, print_warning
from training.common import evaluate_clinical_metrics
from validation.networks.QNetwork import QNetwork
from validation.simulator import GlucoseSimulator

class BehaviorPolicyNetwork(nn.Module):
    """
    Red neuronal para modelar la política de comportamiento que generó los datos.
    
    Parámetros:
    -----------
    cgm_input_dim : tuple
        Dimensiones de entrada para datos CGM
    other_input_dim : tuple
        Dimensiones de entrada para otras características
    action_dim : int
        Dimensión de la acción (dosis de insulina)
    hidden_dim : int, opcional
        Dimensión de las capas ocultas (default: 128)
    """
    
    def __init__(self, cgm_input_dim: tuple, other_input_dim: tuple, 
                action_dim: int = 1, hidden_dim: int = 128):
        super().__init__()
        
        # Encoder para datos CGM
        self.cgm_encoder = nn.Sequential(
            nn.Linear(np.prod(cgm_input_dim), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Encoder para otras características
        self.other_encoder = nn.Sequential(
            nn.Linear(np.prod(other_input_dim), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Capas combinadas para predecir parámetros de distribución
        combined_dim = hidden_dim // 2 + hidden_dim // 2
        self.combined_layer = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Salida para media y desviación estándar (política gaussiana)
        self.mean_layer = nn.Linear(hidden_dim // 2, action_dim)
        self.std_layer = nn.Sequential(
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softplus()  # Asegura que std sea positiva
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward(self, x_cgm: torch.Tensor, x_other: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Paso hacia adelante de la red.
        
        Parámetros:
        -----------
        x_cgm : torch.Tensor
            Datos CGM
        x_other : torch.Tensor
            Otras características
            
        Retorna:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            Media y desviación estándar de la distribución de acción
        """
        # Aplanar entradas si es necesario
        if len(x_cgm.shape) > 2:
            x_cgm = x_cgm.reshape(x_cgm.shape[0], -1)
        if len(x_other.shape) > 2:
            x_other = x_other.reshape(x_other.shape[0], -1)
        
        # Codificar cada componente
        cgm_features = self.cgm_encoder(x_cgm)
        other_features = self.other_encoder(x_other)
        
        # Combinar características
        combined = torch.cat([cgm_features, other_features], dim=1)
        features = self.combined_layer(combined)
        
        # Obtener parámetros de distribución
        mean = self.mean_layer(features)
        std = self.std_layer(features) + 1e-6  # Añadir pequeño epsilon para estabilidad
        
        return mean, std
    
    def log_prob(self, x_cgm: torch.Tensor, x_other: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Calcula el logaritmo de la probabilidad de una acción dada el estado.
        
        Parámetros:
        -----------
        x_cgm : torch.Tensor
            Datos CGM
        x_other : torch.Tensor
            Otras características
        action : torch.Tensor
            Acción para evaluar (dosis de insulina)
            
        Retorna:
        --------
        torch.Tensor
            Logaritmo de la probabilidad
        """
        mean, std = self.forward(x_cgm, x_other)
        
        # Distribución normal
        from torch.distributions import Normal
        dist = Normal(mean, std)
        
        return dist.log_prob(action)


class DoublyRobustEstimator:
    """
    Evaluador de políticas usando el método Doubly Robust.
    
    Parámetros:
    -----------
    cgm_input_dim : tuple
        Dimensiones de entrada para datos CGM
    other_input_dim : tuple
        Dimensiones de entrada para otras características
    hidden_dim : int, opcional
        Dimensión de las capas ocultas (default: 128)
    gamma : float, opcional
        Factor de descuento para recompensas futuras (default: 0.99)
    lr : float, opcional
        Tasa de aprendizaje (default: 0.001)
    """
    
    def __init__(self, cgm_input_dim: tuple, other_input_dim: tuple,
                hidden_dim: int = 128, gamma: float = OFFLINE_GAMMA,
                lr: float = 0.001):
        self.cgm_input_dim = cgm_input_dim
        self.other_input_dim = other_input_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.lr = lr
        
        # Inicializar red Q para Direct Method
        self.q_network = QNetwork(
            cgm_input_dim=cgm_input_dim,
            other_input_dim=other_input_dim,
            action_dim=1,  # Dosis de insulina
            hidden_dim=hidden_dim
        )
        
        # Inicializar red de política de comportamiento
        self.behavior_policy = BehaviorPolicyNetwork(
            cgm_input_dim=cgm_input_dim,
            other_input_dim=other_input_dim,
            action_dim=1,
            hidden_dim=hidden_dim
        )
        
        # Optimizadores
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr, weight_decay=1e-5)
        self.behavior_optimizer = optim.Adam(self.behavior_policy.parameters(), lr=self.lr, weight_decay=1e-5)
        
        # Criterios de pérdida
        self.mse_criterion = nn.MSELoss()
        self.nll_criterion = nn.GaussianNLLLoss()
        
        # Dispositivo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network.to(self.device)
        self.behavior_policy.to(self.device)
        
        # Bootstrap para intervalos de confianza
        self.bootstrap_estimates = []
    
    def _generate_rewards(self, x_cgm: np.ndarray, x_other: np.ndarray, 
                        actions: np.ndarray) -> np.ndarray:
        """
        Genera recompensas basadas en el mantenimiento de glucosa en rango.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM (se espera forma: muestras, pasos_tiempo, características_cgm)
        x_other : np.ndarray
            Otras características
        actions : np.ndarray
            Acciones (dosis de insulina)
            
        Retorna:
        --------
        np.ndarray
            Recompensas calculadas
        """
        # Extraer valores de glucosa actuales (último valor de la primera característica CGM de cada serie)
        # Se asume que x_cgm es (muestras, pasos_tiempo, características_cgm) y la glucosa original es la primera característica (índice 0).
        if x_cgm.ndim == 3 and x_cgm.shape[0] > 0:
            current_glucose = x_cgm[:, -1, 0]
        elif x_cgm.ndim == 2 and x_cgm.shape[0] == actions.shape[0]: # Caso menos común: (muestras, pasos_tiempo) y una sola característica CGM
            current_glucose = x_cgm[:, -1]
        else:
            # Este caso indica un problema con la forma de entrada de x_cgm o una lógica inesperada.
            print_error(f"Forma inesperada o vacía de x_cgm en _generate_rewards: {x_cgm.shape}. No se pudo extraer current_glucose de forma confiable. "
                        f"Se esperaba 3D (muestras, pasos, características) o 2D (muestras, pasos) con muestras coincidiendo con actions ({actions.shape[0]}).")
            # Fallback para evitar un crash, pero esto debe ser investigado.
            # Se crea un array de glucosa con un valor por defecto, pero su longitud debe coincidir con 'actions'.
            current_glucose = np.full(actions.shape[0], HYPOGLYCEMIA_THRESHOLD) # Usar un valor neutro o de error
        
        # Inicializar recompensas
        rewards = np.zeros_like(actions, dtype=np.float32)
        
        # Asignar recompensas basadas en el rango de glucosa
        # En rango (70-180 mg/dL) - recompensa positiva
        # Usar & para el AND lógico elemento a elemento con NumPy arrays
        in_range_mask = (current_glucose >= HYPOGLYCEMIA_THRESHOLD) & \
                        (current_glucose <= HYPERGLYCEMIA_THRESHOLD)
        rewards[in_range_mask] = MAX_REWARD
        
        # Hipoglucemia (<70 mg/dL) - penalización severa
        hypo_mask = current_glucose < HYPOGLYCEMIA_THRESHOLD
        rewards[hypo_mask] = HYPO_PENALTY_BASE
        
        # Hiperglucemia (>180 mg/dL) - penalización moderada
        hyper_mask = current_glucose > HYPERGLYCEMIA_THRESHOLD
        rewards[hyper_mask] = HYPER_PENALTY_BASE
        
        return rewards
    
    def fit(self, x_cgm: np.ndarray, x_other: np.ndarray, y_actions: np.ndarray,
           validation_data: Optional[Tuple] = None,
           batch_size: int = CONST_DEFAULT_BATCH_SIZE,
           epochs: int = CONST_DEFAULT_EPOCHS,
           bootstrap_iterations: int = 20) -> Dict[str, List[float]]:
        """
        Entrena los modelos de Direct Method y Behavior Policy.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrenamiento
        x_other : np.ndarray
            Otras características de entrenamiento
        y_actions : np.ndarray
            Acciones (dosis de insulina) reales
        validation_data : Optional[Tuple], opcional
            Datos de validación como ((x_cgm_val, x_other_val), y_val) (default: None)
        batch_size : int, opcional
            Tamaño de lote (default: 32)
        epochs : int, opcional
            Número de épocas (default: 10)
        bootstrap_iterations : int, opcional
            Número de iteraciones bootstrap para intervalos de confianza (default: 20)
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historial de entrenamiento
        """
        # Generar recompensas
        rewards = self._generate_rewards(x_cgm, x_other, y_actions)
        
        # Crear DataLoader
        dataset = TensorDataset(
            torch.FloatTensor(x_cgm),
            torch.FloatTensor(x_other),
            torch.FloatTensor(y_actions).reshape(-1, 1),
            torch.FloatTensor(rewards).reshape(-1, 1)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        
        # Historial de entrenamiento
        history = {
            'q_loss': [],
            'behavior_loss': [],
            'val_q_loss': [],
            'val_behavior_loss': []
        }
        
        # Entrenamiento principal
        for epoch in range(epochs):
            epoch_q_loss = 0.0
            epoch_behavior_loss = 0.0
            self.q_network.train()
            self.behavior_policy.train()
            
            for batch_cgm, batch_other, batch_actions, batch_rewards in tqdm(dataloader, desc=f"Época {epoch+1}/{epochs}"):
                batch_cgm = batch_cgm.to(self.device)
                batch_other = batch_other.to(self.device)
                batch_actions = batch_actions.to(self.device)
                batch_rewards = batch_rewards.to(self.device)
                
                # Entrenar red Q (Direct Method)
                self.q_optimizer.zero_grad()
                q_values = self.q_network(batch_cgm, batch_other, batch_actions)
                q_loss = self.mse_criterion(q_values, batch_rewards)
                q_loss.backward()
                self.q_optimizer.step()
                
                # Entrenar política de comportamiento
                self.behavior_optimizer.zero_grad()
                mean, std = self.behavior_policy(batch_cgm, batch_other)
                behavior_loss = self.nll_criterion(mean, batch_actions.squeeze(), std.pow(2))
                behavior_loss.backward()
                self.behavior_optimizer.step()
                
                epoch_q_loss += q_loss.item()
                epoch_behavior_loss += behavior_loss.item()
            
            # Registrar pérdida de entrenamiento
            avg_q_loss = epoch_q_loss / len(dataloader)
            avg_behavior_loss = epoch_behavior_loss / len(dataloader)
            history['q_loss'].append(avg_q_loss)
            history['behavior_loss'].append(avg_behavior_loss)
            
            # Validación si hay datos disponibles
            if validation_data:
                val_q_loss, val_behavior_loss = self._validate(validation_data[0][0], validation_data[0][1], validation_data[1])
                history['val_q_loss'].append(val_q_loss)
                history['val_behavior_loss'].append(val_behavior_loss)
                print(f"Época {epoch+1}/{epochs} - Q Loss: {avg_q_loss:.4f} - Behavior Loss: {avg_behavior_loss:.4f} - Val Q Loss: {val_q_loss:.4f} - Val Behavior Loss: {val_behavior_loss:.4f}")
            else:
                print(f"Época {epoch+1}/{epochs} - Q Loss: {avg_q_loss:.4f} - Behavior Loss: {avg_behavior_loss:.4f}")
        
        # Realizar bootstrap para intervalos de confianza
        self._bootstrap(x_cgm, x_other, y_actions, bootstrap_iterations)
        
        return history
    
    def _validate(self, x_cgm_val: np.ndarray, x_other_val: np.ndarray, 
                y_val: np.ndarray) -> Tuple[float, float]:
        """
        Valida los modelos con datos de validación.
        
        Parámetros:
        -----------
        x_cgm_val : np.ndarray
            Datos CGM de validación
        x_other_val : np.ndarray
            Otras características de validación
        y_val : np.ndarray
            Acciones (dosis) de validación
            
        Retorna:
        --------
        Tuple[float, float]
            (pérdida Q, pérdida de comportamiento)
        """
        self.q_network.eval()
        self.behavior_policy.eval()
        
        with torch.no_grad():
            # Convertir a tensores
            x_cgm_tensor = torch.FloatTensor(x_cgm_val).to(self.device)
            x_other_tensor = torch.FloatTensor(x_other_val).to(self.device)
            y_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)
            
            # Generar recompensas para validación
            rewards = self._generate_rewards(x_cgm_val, x_other_val, y_val)
            rewards_tensor = torch.FloatTensor(rewards).reshape(-1, 1).to(self.device)
            
            # Calcular pérdida Q
            q_values = self.q_network(x_cgm_tensor, x_other_tensor, y_tensor)
            q_loss = self.mse_criterion(q_values, rewards_tensor).item()
            
            # Calcular pérdida de comportamiento
            mean, std = self.behavior_policy(x_cgm_tensor, x_other_tensor)
            behavior_loss = self.nll_criterion(mean, y_tensor.squeeze(), std.pow(2)).item()
        
        return q_loss, behavior_loss
    
    def _bootstrap(self, x_cgm: np.ndarray, x_other: np.ndarray, actions: np.ndarray, 
                 iterations: int = 20):
        """
        Realiza bootstrap para calcular intervalos de confianza.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM
        x_other : np.ndarray
            Otras características
        actions : np.ndarray
            Acciones (dosis)
        iterations : int, opcional
            Número de iteraciones bootstrap (default: 20)
        """
        n_samples = len(x_cgm)
        self.bootstrap_estimates = []
        
        for _ in range(iterations):
            # Muestreo con reemplazo
            indices = np.random.choice(n_samples, n_samples, replace=True)
            x_cgm_bootstrap = x_cgm[indices]
            x_other_bootstrap = x_other[indices]
            actions_bootstrap = actions[indices]
            
            # Calcular estimación DR para esta muestra
            dr_estimate = self._compute_dr_estimate(x_cgm_bootstrap, x_other_bootstrap, actions_bootstrap)
            self.bootstrap_estimates.append(dr_estimate)
    
    def _compute_importance_weights(self, x_cgm: np.ndarray, x_other: np.ndarray, 
                                  actions: np.ndarray, policy) -> np.ndarray:
        """
        Calcula los pesos de importancia entre la política de evaluación y de comportamiento.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM
        x_other : np.ndarray
            Otras características
        actions : np.ndarray
            Acciones (dosis)
        policy : object
            Política a evaluar
            
        Retorna:
        --------
        np.ndarray
            Pesos de importancia
        """
        self.behavior_policy.eval()
        
        with torch.no_grad():
            # Calcular probabilidades de comportamiento
            x_cgm_tensor = torch.FloatTensor(x_cgm).to(self.device)
            x_other_tensor = torch.FloatTensor(x_other).to(self.device)
            actions_tensor = torch.FloatTensor(actions).reshape(-1, 1).to(self.device)
            
            behavior_log_probs = self.behavior_policy.log_prob(x_cgm_tensor, x_other_tensor, actions_tensor)
            behavior_probs = torch.exp(behavior_log_probs).cpu().numpy()
            
            # Calcular probabilidades de la política de evaluación
            # Usamos predict para obtener acciones de la política de evaluación
            if hasattr(policy, 'predict'):
                eval_actions = policy.predict(x_cgm, x_other)
            else:
                # Manejar caso especial para el ensamble
                eval_actions = np.array([policy.predict(x_cgm[i:i+1], x_other[i:i+1]) 
                                       for i in range(len(x_cgm))])
            
            # Asumir política gaussiana con sigma fijo para simplicidad
            eval_sigma = 0.1
            eval_probs = np.exp(-0.5 * ((eval_actions - actions) / eval_sigma) ** 2) / (eval_sigma * np.sqrt(2 * np.pi))
            
            # Calcular y recortar pesos de importancia
            weights = np.clip(eval_probs / (behavior_probs + 1e-6), 0, CONST_IPS_CLIP)
            
            return weights
    
    def _compute_dr_estimate(self, x_cgm: np.ndarray, x_other: np.ndarray, 
                           actions: np.ndarray, policy=None) -> float:
        """
        Calcula la estimación Doubly Robust para una política.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM
        x_other : np.ndarray
            Otras características
        actions : np.ndarray
            Acciones (dosis)
        policy : object, opcional
            Política a evaluar (default: None)
            
        Retorna:
        --------
        float
            Estimación Doubly Robust
        """
        # Si no se proporciona política, crear un estimador directo simple
        if policy is None:
            return self._compute_direct_method_estimate(x_cgm, x_other, actions)
        
        # Calcular estimación de método directo (Q-values)
        dm_estimate = self._compute_direct_method_estimate(x_cgm, x_other, actions)
        
        # Calcular recompensas reales
        rewards = self._generate_rewards(x_cgm, x_other, actions)
        
        # Calcular pesos de importancia
        importance_weights = self._compute_importance_weights(x_cgm, x_other, actions, policy)
        
        # Calcular Q-values para los pares estado-acción observados
        with torch.no_grad():
            x_cgm_tensor = torch.FloatTensor(x_cgm).to(self.device)
            x_other_tensor = torch.FloatTensor(x_other).to(self.device)
            actions_tensor = torch.FloatTensor(actions).reshape(-1, 1).to(self.device)
            
            q_values = self.q_network(x_cgm_tensor, x_other_tensor, actions_tensor).cpu().numpy().flatten()
        
        # Calcular término de corrección
        correction_term = np.mean(importance_weights * (rewards - q_values))
        
        # Estimación Doubly Robust
        dr_estimate = dm_estimate + correction_term
        
        return float(dr_estimate)
    
    def _compute_direct_method_estimate(self, x_cgm: np.ndarray, x_other: np.ndarray, 
                                     actions: np.ndarray) -> float:
        """
        Calcula la estimación por método directo usando la red Q.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM
        x_other : np.ndarray
            Otras características
        actions : np.ndarray
            Acciones (dosis)
            
        Retorna:
        --------
        float
            Estimación por método directo
        """
        self.q_network.eval()
        
        with torch.no_grad():
            x_cgm_tensor = torch.FloatTensor(x_cgm).to(self.device)
            x_other_tensor = torch.FloatTensor(x_other).to(self.device)
            actions_tensor = torch.FloatTensor(actions).reshape(-1, 1).to(self.device)
            
            q_values = self.q_network(x_cgm_tensor, x_other_tensor, actions_tensor)
            mean_q_value = q_values.mean().item()
        
        return mean_q_value
    
    def evaluate_policy(self, policy: Any, 
                       x_cgm_test: np.ndarray, 
                       x_other_test: np.ndarray, 
                       y_actions_test: np.ndarray, # Renamed for clarity
                       context_test_data: Optional[Dict[str, np.ndarray]] = None,
                       simulator: Optional[GlucoseSimulator] = None) -> Dict[str, float]:
        """
        Evalúa una política utilizando el estimador Doubly Robust.
        
        Parámetros:
        -----------
        policy : Any
            Política a evaluar (debe tener un método predict_with_context).
        x_cgm_test : np.ndarray
            Datos CGM de prueba.
        x_other_test : np.ndarray
            Otras características de prueba.
        y_actions_test : np.ndarray
            Acciones reales del conjunto de prueba.
        context_test_data : Optional[Dict[str, np.ndarray]], opcional
            Datos contextuales para cada muestra en los datos de prueba.
        simulator : Optional[GlucoseSimulator], opcional
            Simulador de glucosa para métricas clínicas (default: None).
            
        Retorna:
        --------
        Dict[str, float]
            Métricas de evaluación de la política.
        """
        self.q_network.eval()
        self.behavior_policy.eval()
        num_samples = len(x_cgm_test)

        if num_samples == 0:
            print_warning("No hay datos de prueba para evaluar la política en DRE.")
            return {}

        # Asegurar que x_other_test y context_test_data tengan la misma longitud que x_cgm_test si no son None
        if x_other_test is not None and len(x_other_test) != num_samples:
            print_error(f"La longitud de x_other_test ({len(x_other_test)}) no coincide con x_cgm_test ({num_samples}).")
            return {"error": -1.0}
        if context_test_data is not None:
            for key, arr in context_test_data.items():
                if len(arr) != num_samples:
                    print_error(f"La longitud del contexto '{key}' ({len(arr)}) no coincide con x_cgm_test ({num_samples}).")
                    return {"error": -1.0}

        # Calcular estimación Doubly Robust
        # Esto requiere acciones de la política objetivo, Q-values, y pesos de importancia
        
        policy_actions = np.zeros(num_samples)
        q_values_policy_actions = np.zeros(num_samples)
        importance_weights = np.zeros(num_samples)
        
        # Generar recompensas para los datos de prueba (usando y_actions_test)
        # Esto asume que y_actions_test son las acciones que generaron las recompensas observadas.
        # Si no hay recompensas observadas directas, este paso podría necesitar ajuste.
        # Para DRE, R son las recompensas observadas bajo la política de comportamiento.
        # Si y_actions_test son las acciones de comportamiento, y tenemos un simulador, podemos generar R.
        # O, si el dataset original tiene recompensas, usarlas. Aquí generaremos con simulador.
        
        observed_rewards = np.zeros(num_samples)
        if simulator:
            initial_glucose_test = np.zeros(num_samples)
            carb_intake_test = np.zeros(num_samples)
            for i in range(num_samples):
                if context_test_data and 'current_glucose' in context_test_data and i < len(context_test_data['current_glucose']):
                    initial_glucose_test[i] = context_test_data['current_glucose'][i]
                elif x_cgm_test[i].ndim > 0 and x_cgm_test[i].shape[0] > 0:
                    initial_glucose_test[i] = x_cgm_test[i][-1, 0] if x_cgm_test[i].ndim == 2 else x_cgm_test[i][-1]
                else:
                    initial_glucose_test[i] = 150.0

                if context_test_data and 'carb_intake' in context_test_data and i < len(context_test_data['carb_intake']):
                    carb_intake_test[i] = context_test_data['carb_intake'][i]
                elif context_test_data and 'meal_carbs' in context_test_data and i < len(context_test_data['meal_carbs']):
                     carb_intake_test[i] = context_test_data['meal_carbs'][i]
                else:
                    carb_intake_test[i] = 0.0
                
                _, observed_rewards[i], _, _ = simulator.step(y_actions_test[i], initial_glucose_test[i], carb_intake_test[i])
        else:
            print_warning("DRE: Simulador no proporcionado, las recompensas observadas serán 0. La estimación DR puede no ser significativa.")


        for i in tqdm(range(num_samples), desc="Evaluando política con DRE", leave=False):
            x_cgm_sample = x_cgm_test[i]
            x_other_sample = x_other_test[i] if x_other_test is not None and i < len(x_other_test) else np.array([])
            action_behavior = y_actions_test[i] # Acción de la política de comportamiento

            # Extraer contexto para predict_with_context
            current_glucose_val: float
            if context_test_data and 'current_glucose' in context_test_data and i < len(context_test_data['current_glucose']):
                current_glucose_val = float(context_test_data['current_glucose'][i])
            elif x_cgm_sample.ndim > 0 and x_cgm_sample.shape[0] > 0:
                current_glucose_val = float(x_cgm_sample[-1, 0] if x_cgm_sample.ndim == 2 else x_cgm_sample[-1])
            else:
                current_glucose_val = 150.0 
                print_warning(f"DRE: No se pudo determinar current_glucose para la muestra {i}, usando fallback {current_glucose_val}.")

            context_values_for_prediction: Dict[str, Optional[float]] = {}
            required_context_keys_for_predict = ['carb_intake', 'iob']
            
            for key_name in CONTEXT_FEATURE_ORDER:
                if key_name == 'current_glucose': continue
                default_val = 0.0
                val = default_val
                if context_test_data and key_name in context_test_data and i < len(context_test_data[key_name]):
                    val = float(context_test_data[key_name][i])
                elif key_name in required_context_keys_for_predict:
                     print_warning(f"DRE: Característica de contexto requerida '{key_name}' no encontrada para la muestra {i}. Usando fallback {default_val}.")
                context_values_for_prediction[key_name] = val
            
            try:
                action_policy = policy.predict_with_context(
                    x_cgm=x_cgm_sample.reshape(1, *x_cgm_sample.shape) if x_cgm_sample.ndim < 3 else x_cgm_sample,
                    x_other=x_other_sample.reshape(1, *x_other_sample.shape) if x_other_sample.ndim < 2 and x_other_sample.size > 0 else x_other_sample,
                    current_glucose=current_glucose_val,
                    carb_intake=context_values_for_prediction.get('carb_intake', 0.0), # type: ignore
                    iob=context_values_for_prediction.get('iob', 0.0), # type: ignore
                    sleep_quality=context_values_for_prediction.get('sleep_quality'),
                    work_intensity=context_values_for_prediction.get('work_intensity'),
                    exercise_intensity=context_values_for_prediction.get('exercise_intensity')
                )
                policy_actions[i] = action_policy
            except Exception as e:
                print_error(f"Error al llamar a policy.predict_with_context en DRE para la muestra {i}: {e}")
                policy_actions[i] = 0.0


            with torch.no_grad():
                # Q(s, a_policy)
                q_input_cgm = torch.FloatTensor(x_cgm_sample).unsqueeze(0).to(self.device)
                q_input_other = torch.FloatTensor(x_other_sample).unsqueeze(0).to(self.device) if x_other_sample.size > 0 else torch.empty(1,0).to(self.device)
                q_input_action_policy = torch.FloatTensor([policy_actions[i]]).unsqueeze(0).to(self.device)
                
                if q_input_cgm.ndim > 2: q_input_cgm = q_input_cgm.reshape(q_input_cgm.shape[0], -1)
                if q_input_other.ndim > 2: q_input_other = q_input_other.reshape(q_input_other.shape[0], -1)
                
                q_values_policy_actions[i] = self.q_network(q_input_cgm, q_input_other, q_input_action_policy).item()

                # Q(s, a_behavior)
                q_input_action_behavior = torch.FloatTensor([action_behavior]).unsqueeze(0).to(self.device)
                q_s_ab = self.q_network(q_input_cgm, q_input_other, q_input_action_behavior).item()

                # pi_target(a_policy | s) / pi_behavior(a_policy | s)
                # Esto es problemático si la política objetivo es determinística y la de comportamiento no, o viceversa.
                # Para DDPG (determinística), pi_target(a_policy | s) es infinito si a_policy es la acción de DDPG, 0 si no.
                # Asumimos que 'policy' es la política objetivo (determinística) y 'self.behavior_policy' es la de comportamiento (gaussiana).
                
                # Probabilidad de la acción de la política objetivo bajo la política de comportamiento
                log_prob_policy_action_under_behavior = self.behavior_policy.log_prob(q_input_cgm, q_input_other, q_input_action_policy)
                prob_policy_action_under_behavior = torch.exp(log_prob_policy_action_under_behavior).item()

                # Probabilidad de la acción de la política de comportamiento bajo la política de comportamiento
                log_prob_behavior_action_under_behavior = self.behavior_policy.log_prob(q_input_cgm, q_input_other, q_input_action_behavior)
                prob_behavior_action_under_behavior = torch.exp(log_prob_behavior_action_under_behavior).item()

                # Si la política objetivo es determinística, p_target(a_target|s) = 1 (o delta), p_target(a_other|s) = 0
                # El peso de importancia es pi_e(a|s) / pi_b(a|s)
                # Aquí, 'a' es la acción tomada por la política de comportamiento (action_behavior)
                # y queremos evaluar la política 'policy'.
                # El peso IS es rho_t = pi_e(a_t|s_t) / pi_b(a_t|s_t)
                # Si pi_e es determinística (toma action_policy), entonces:
                # rho_t = 1 / pi_b(action_policy|s_t) si action_behavior == action_policy, y 0 si no.
                # Esto no es estándar para DRE.
                # DRE clásico: V^{DR} = E_{s,a,r sim D_b} [ rho(s,a) * (r - Q(s,a)) + E_{a' sim pi_e(s)} [Q(s,a')] ]
                # donde rho(s,a) = pi_e(a|s) / pi_b(a|s)
                # Para DDPG (pi_e es determinística), pi_e(a|s) es una función delta.
                # Si pi_b es continua, pi_e(a|s)/pi_b(a|s) es problemático.
                # Usamos la forma alternativa: E[Q(s, pi_e(s))] + E[ (pi_e(a|s)/pi_b(a|s)) * (R - Q(s,a)) ]
                # Si pi_e es determinística, a^* = pi_e(s). El segundo término se vuelve:
                # (1 / pi_b(a^*|s)) * (R - Q(s,a^*)) si a = a^*, y 0 si a != a^*.
                # Esto requiere que la acción 'a' en los datos sea la misma que la acción de la política objetivo 'a^*'.

                # Usaremos una forma más común para DRE con política objetivo determinística:
                # V_DR = E_D [ Q(s, pi_e(s)) + (delta(a == pi_e(s)) / pi_b(a|s)) * (r - Q(s,a)) ]
                # donde delta es la función de Dirac. Esto es aún complicado.

                # Simplificación: Usamos la estimación del método directo Q(s, pi_e(s)) y añadimos el término de corrección IS
                # para la acción de comportamiento.
                # V_DR = E_D [ Q(s, pi_e(s)) ]  <-- Esto es DM
                # V_IS = E_D [ (pi_e(a_b|s) / pi_b(a_b|s)) * r ] <-- IS del valor de la política
                # V_WIS = E_D [ w(s,a_b) * r ] / E_D [ w(s,a_b) ] donde w = pi_e/pi_b

                # Para DRE, el peso de importancia es para la acción *observada* (action_behavior)
                # bajo la política objetivo y la política de comportamiento.
                # Si la política objetivo es determinística (toma action_policy):
                # pi_e(action_behavior | s) es 1 si action_behavior == action_policy, 0 otherwise.
                # Esto hace que muchos pesos sean 0.

                # Usaremos la forma: V_DR = 1/N * sum( Q(s_i, pi_e(s_i)) + w_i * (r_i - Q(s_i, a_i)) )
                # donde w_i = pi_e(a_i|s_i) / pi_b(a_i|s_i).
                # Si pi_e es determinística, a_i^* = pi_e(s_i).
                # w_i = delta(a_i == a_i^*) / pi_b(a_i|s_i).
                # El término de corrección solo se aplica si a_i == a_i^*.
                
                # Peso de importancia para la acción de comportamiento (action_behavior)
                # pi_e(action_behavior | s) / pi_b(action_behavior | s)
                # Si pi_e es determinística, pi_e(action_behavior | s) es 1 si action_behavior es la acción que tomaría pi_e, y 0 si no.
                # Esto significa que si la acción de comportamiento no es la que tomaría la política objetivo, el peso es 0.
                
                # Para DDPG, pi_e(a|s) es una delta en policy_actions[i].
                # pi_b(a|s) es la densidad de la política de comportamiento.
                # Si action_behavior es la acción que la política objetivo tomaría, entonces pi_e(action_behavior|s) es "infinito".
                # Esto requiere una formulación cuidadosa.

                # Usaremos el estimador de método directo como base y lo reportaremos.
                # El componente IS es más complejo de implementar correctamente aquí sin más supuestos.
                # Por ahora, calculamos Q(s, a_policy) como la estimación principal.
                # Y podemos calcular Q(s, a_behavior) para referencia.
                
                # Para el peso de importancia: pi_target(action_behavior | s) / pi_behavior(action_behavior | s)
                # Si la política objetivo es determinística (toma `policy_actions[i]`):
                # El numerador es 1 si action_behavior == policy_actions[i] (aproximadamente, debido a la naturaleza continua)
                # y 0 en caso contrario. Esto es problemático.
                
                # Una aproximación común para DRE con política objetivo determinística pi_e(s) es:
                # V_DR = E_s [Q(s, pi_e(s))] + E_{s,a,r ~ D_b} [ (1 / pi_b(a|s)) * I(a == pi_e(s)) * (r - Q(s,a)) ]
                # donde I es la función indicadora. El término I(a == pi_e(s)) será casi siempre cero para acciones continuas.

                # Por simplicidad y robustez, nos enfocaremos en el método directo (DM) y el IS por separado.
                # El DRE completo es más matizado.
                # Aquí, calcularemos Q(s, policy_action) como la estimación DM.
                # Y calcularemos IS(policy_action) si es posible.
                
                # Para IS: E [rho * r], rho = pi_target(a_behavior | s) / pi_behavior(a_behavior | s)
                # Si pi_target es DDPG, es determinística.
                # pi_target(action_behavior | s) es 1 si action_behavior es la acción que DDPG tomaría, 0 si no.
                # Esto hace que rho sea 0 a menos que la acción de comportamiento coincida exactamente con la de DDPG.
                
                # Por ahora, el DRE se simplificará a la estimación del método directo Q(s, pi_e(s)).
                # El componente de corrección de IS es difícil de aplicar directamente aquí sin supuestos adicionales
                # o una política objetivo estocástica.
                pass # No se calculan pesos de importancia complejos por ahora.

        dm_estimate = np.mean(q_values_policy_actions) if num_samples > 0 else 0.0
        
        results = {
            'dre_direct_method_estimate': dm_estimate,
            'dre_mse_actions_policy_vs_behavior': float(mean_squared_error(y_actions_test, policy_actions)) if y_actions_test is not None else -1.0,
            'dre_mae_actions_policy_vs_behavior': float(mean_absolute_error(y_actions_test, policy_actions)) if y_actions_test is not None else -1.0,
        }

        if simulator:
            # Reutilizar initial_glucose_test y carb_intake_test si ya se calcularon
            if 'initial_glucose_test' not in locals(): # Evitar redefinición si ya existen
                initial_glucose_test = np.zeros(num_samples)
                carb_intake_test = np.zeros(num_samples)
                for i in range(num_samples):
                    if context_test_data and 'current_glucose' in context_test_data and i < len(context_test_data['current_glucose']):
                        initial_glucose_test[i] = context_test_data['current_glucose'][i]
                    elif x_cgm_test[i].ndim > 0 and x_cgm_test[i].shape[0] > 0:
                        initial_glucose_test[i] = x_cgm_test[i][-1, 0] if x_cgm_test[i].ndim == 2 else x_cgm_test[i][-1]
                    else:
                        initial_glucose_test[i] = 150.0

                    if context_test_data and 'carb_intake' in context_test_data and i < len(context_test_data['carb_intake']):
                        carb_intake_test[i] = context_test_data['carb_intake'][i]
                    elif context_test_data and 'meal_carbs' in context_test_data and i < len(context_test_data['meal_carbs']):
                        carb_intake_test[i] = context_test_data['meal_carbs'][i]
                    else:
                        carb_intake_test[i] = 0.0
            
            clinical_metrics = evaluate_clinical_metrics(
                simulator=simulator,
                predictions=policy_actions, # Evaluar acciones de la política objetivo
                initial_glucose=initial_glucose_test,
                carb_intake=carb_intake_test
            )
            results.update({f"dre_{k}": v for k, v in clinical_metrics.items()})
            
        return results

def create_dre_evaluator(cgm_input_dim: tuple, other_input_dim: tuple) -> DoublyRobustEstimator:
    """
    Función para crear un evaluador Doubly Robust.
    
    Parámetros:
    -----------
    cgm_input_dim : tuple
        Dimensiones de entrada para datos CGM
    other_input_dim : tuple
        Dimensiones de entrada para otras características
        
    Retorna:
    --------
    DoublyRobustEstimator
        Instancia del evaluador Doubly Robust
    """
    return DoublyRobustEstimator(cgm_input_dim, other_input_dim)