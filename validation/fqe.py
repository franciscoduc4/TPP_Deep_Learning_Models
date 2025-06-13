"""
Fitted Q Evaluation (FQE)

1. Q-Function Learning: FQE ajusta iterativamente una función Q para estimar el valor de las acciones basándose en datos históricos de interacciones con el entorno.
    - Por cada par estado-acción (s,a) en el dataset, se computa un valor objetivo Q(s,a) como la recompensa inmediata más el valor descontado de la acción futura.
    objetivo = recompensa + gamma * Q(s', pi(s'))
    donde pi es la política que se está evaluando y s' es el siguiente estado.
    - Ajusta a un modelo de regresión para predecir estos objetivos a partir de los pares estado-acción.
    - Se repite hasta lograr una convergencia.
2. Policy Evaluation: Una vez que la función Q está "aprendida", el valor de la política se estima como el valor esperado de la función Q bajo la política.
    - Se evalúa la política al calcular el valor esperado de la función Q para las acciones tomadas por la política en los estados observados.
3. Bootstrap para Intervalos de Confianza: FQE utiliza técnicas de bootstrap para calcular intervalos de confianza sobre las estimaciones de valor Q.
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
    CONST_DEFAULT_BATCH_SIZE, CONTEXT_FEATURE_ORDER, OFFLINE_GAMMA, CONST_DEFAULT_EPOCHS,
    CONST_CONFIDENCE_LEVEL, SEVERE_HYPOGLYCEMIA_THRESHOLD,
    HYPOGLYCEMIA_THRESHOLD, HYPERGLYCEMIA_THRESHOLD, SEVERE_HYPERGLYCEMIA_THRESHOLD
)
from config.models_config import EARLY_STOPPING_POLICY
from custom.printer import print_error, print_warning
from training.common import evaluate_clinical_metrics
from validation.networks.QNetwork import QNetwork
from validation.simulator import GlucoseSimulator

class FittedQEvaluation:
    """
    Evaluador de políticas usando Fitted Q Evaluation (FQE).
    
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
        
        # Inicializar red Q
        self.q_network = QNetwork(
            cgm_input_dim=cgm_input_dim,
            other_input_dim=other_input_dim,
            action_dim=1,  # Dosis de insulina
            hidden_dim=hidden_dim
        )
        
        # Inicializar red Q objetivo (para estabilidad)
        self.target_q_network = QNetwork(
            cgm_input_dim=cgm_input_dim,
            other_input_dim=other_input_dim,
            action_dim=1,
            hidden_dim=hidden_dim
        )
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizador
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr, weight_decay=1e-5)
        
        # Criterio de pérdida
        self.criterion = nn.MSELoss()
        
        # Dispositivo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network.to(self.device)
        self.target_q_network.to(self.device)
        
        # Bootstrap para intervalos de confianza
        self.bootstrap_estimates = []
    
    def _generate_rewards(self, x_cgm: np.ndarray, x_other: np.ndarray, 
                    actions: np.ndarray) -> np.ndarray:
        """
        Genera recompensas basadas en el mantenimiento de glucosa en rango.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM
        x_other : np.ndarray
            Otras características
        actions : np.ndarray
            Acciones (dosis de insulina)
            
        Retorna:
        --------
        np.ndarray
            Recompensas calculadas
        """
        # Inicializar recompensas
        rewards = np.zeros_like(actions, dtype=np.float32)
        
        # Extraer valores de glucosa actuales de manera segura según la forma de los datos
        if len(x_cgm.shape) == 3:  # (samples, time_steps, features)
            current_glucose = x_cgm[:, -1, 0]  # Último paso de tiempo, primera característica
        elif len(x_cgm.shape) == 2:  # (samples, time_steps) o (samples, features)
            if x_cgm.shape[0] == actions.shape[0]:  # Verificar si las muestras coinciden
                if x_cgm.shape[1] > 1:
                    current_glucose = x_cgm[:, -1]  # Último paso de tiempo
                else:
                    current_glucose = x_cgm[:, 0]  # Primera característica
            else:
                raise ValueError(f"Número de muestras no coincide: {x_cgm.shape[0]} vs {actions.shape[0]}")
        else:
            raise ValueError(f"Forma inesperada para x_cgm: {x_cgm.shape}")
        
        # Verificar que las dimensiones coincidan
        if len(current_glucose) != len(rewards):
            raise ValueError(f"Dimensiones no coinciden: {len(current_glucose)} valores de glucosa pero {len(rewards)} recompensas")
        
        # Hipoglucemia severa (<54 mg/dL) - penalización muy severa
        severe_hypo = current_glucose < SEVERE_HYPOGLYCEMIA_THRESHOLD
        rewards[severe_hypo] = -4.0
        
        # Hipoglucemia moderada (54-70 mg/dL) - penalización severa
        hypo = np.logical_and(
            current_glucose >= SEVERE_HYPOGLYCEMIA_THRESHOLD,
            current_glucose < HYPOGLYCEMIA_THRESHOLD
        )
        rewards[hypo] = -2.0
        
        # En rango (70-180 mg/dL) - recompensa positiva
        in_range = np.logical_and(
            current_glucose >= HYPOGLYCEMIA_THRESHOLD, 
            current_glucose <= HYPERGLYCEMIA_THRESHOLD
        )
        rewards[in_range] = 1.0
        
        # Hiperglucemia moderada (180-250 mg/dL) - penalización moderada
        hyper = np.logical_and(
            current_glucose > HYPERGLYCEMIA_THRESHOLD,
            current_glucose <= SEVERE_HYPERGLYCEMIA_THRESHOLD
        )
        rewards[hyper] = -1.0
        
        # Hiperglucemia severa (>250 mg/dL) - penalización severa
        severe_hyper = current_glucose > SEVERE_HYPERGLYCEMIA_THRESHOLD
        rewards[severe_hyper] = -3.0
        
        return rewards
    
    def fit(self, x_cgm: np.ndarray, x_other: np.ndarray, y_actions: np.ndarray,
           validation_data: Optional[Tuple] = None,
           batch_size: int = CONST_DEFAULT_BATCH_SIZE,
           epochs: int = CONST_DEFAULT_EPOCHS,
           bootstrap_iterations: int = 20,
           patience: int = EARLY_STOPPING_POLICY['early_stopping_patience'],
           min_delta: float = EARLY_STOPPING_POLICY['early_stopping_min_delta'],
           restore_best_weights: bool = EARLY_STOPPING_POLICY['early_stopping_restore_best_weights']) -> Dict[str, List[float]]:
        """
        Entrena el modelo FQE con los datos proporcionados.
        
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
        patience : int, opcional
            Número de épocas sin mejora antes de detener el entrenamiento
        min_delta : float, opcional
            Mínima mejora para considerar progreso
        restore_best_weights : bool, opcional
            Si restaurar los mejores pesos al finalizar
        
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
            'loss': [],
            'val_loss': []
        }
        
        # Variables para early stopping
        best_val_loss = float('inf')
        best_weights = None
        early_stop_counter = 0
        
        # Crear barras de progreso
        epoch_progress = tqdm(range(epochs), desc="Entrenamiento", position=0)
        
        # Línea de métricas que se actualizará dinámicamente
        metrics_line = ""
        
        for epoch in epoch_progress:
            epoch_loss = 0.0
            self.q_network.train()
            
            # Barra de progreso para los batches (no mostrará progreso individual)
            batch_progress = tqdm(dataloader, desc=f"Época {epoch+1}/{epochs}", 
                           leave=False, position=1)
            
            for batch_cgm, batch_other, batch_actions, batch_rewards in batch_progress:
                batch_cgm = batch_cgm.to(self.device)
                batch_other = batch_other.to(self.device)
                batch_actions = batch_actions.to(self.device)
                batch_rewards = batch_rewards.to(self.device)
                
                # Forward pass
                q_values = self.q_network(batch_cgm, batch_other, batch_actions)
                
                # Para el estado siguiente, usamos la misma acción (simplificación para FQE)
                with torch.no_grad():
                    target_q_values = batch_rewards + self.gamma * self.target_q_network(
                        batch_cgm, batch_other, batch_actions
                    )
                
                # Calcular pérdida
                loss = self.criterion(q_values, target_q_values)
                
                # Backward pass y optimización
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            # Actualizar red objetivo periódicamente
            if epoch % 5 == 0:
                self.target_q_network.load_state_dict(self.q_network.state_dict())
            
            # Registrar pérdida de entrenamiento
            avg_loss = epoch_loss / len(dataloader)
            history['loss'].append(avg_loss)
            
            # Validación si hay datos disponibles
            val_loss = None
            if validation_data:
                val_loss = self._validate(validation_data[0][0], validation_data[0][1], validation_data[1])
                history['val_loss'].append(val_loss)
                metrics_line = f"Pérdida: {avg_loss:.4f} - Pérdida val: {val_loss:.4f}"
                
                # Verificación para early stopping con datos de validación
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    if restore_best_weights:
                        best_weights = {k: v.cpu().clone() for k, v in self.q_network.state_dict().items()}
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
            else:
                # Sin datos de validación, usar pérdida de entrenamiento
                metrics_line = f"Pérdida: {avg_loss:.4f}"
                
                # Verificación para early stopping con pérdida de entrenamiento
                if avg_loss < best_val_loss - min_delta:
                    best_val_loss = avg_loss
                    if restore_best_weights:
                        best_weights = {k: v.cpu().clone() for k, v in self.q_network.state_dict().items()}
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
            
            # Actualizar descripción de la barra de progreso con las métricas
            epoch_progress.set_description(f"Entrenamiento: {metrics_line}")
            
            # Verificar si se debe activar early stopping
            if early_stop_counter >= patience:
                epoch_progress.write(f"Early stopping en época {epoch+1}")
                break
        
        # Restaurar los mejores pesos si corresponde
        if restore_best_weights and best_weights is not None:
            self.q_network.load_state_dict(best_weights)
            if val_loss is not None:
                epoch_progress.write(f"Restaurados mejores pesos con pérdida de validación: {best_val_loss:.4f}")
            else:
                epoch_progress.write(f"Restaurados mejores pesos con pérdida de entrenamiento: {best_val_loss:.4f}")
        
        # Realizar bootstrap para intervalos de confianza
        self._bootstrap(x_cgm, x_other, y_actions, bootstrap_iterations)
        
        return history
    
    def _validate(self, x_cgm_val: np.ndarray, x_other_val: np.ndarray, 
                y_val: np.ndarray) -> float:
        """
        Valida el modelo con datos de validación.
        
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
        float
            Pérdida de validación
        """
        self.q_network.eval()
        
        with torch.no_grad():
            # Convertir a tensores
            x_cgm_tensor = torch.FloatTensor(x_cgm_val).to(self.device)
            x_other_tensor = torch.FloatTensor(x_other_val).to(self.device)
            y_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)
            
            # Generar recompensas para validación
            rewards = self._generate_rewards(x_cgm_val, x_other_val, y_val)
            rewards_tensor = torch.FloatTensor(rewards).reshape(-1, 1).to(self.device)
            
            # Calcular Q-values
            q_values = self.q_network(x_cgm_tensor, x_other_tensor, y_tensor)
            target_q_values = rewards_tensor + self.gamma * self.target_q_network(
                x_cgm_tensor, x_other_tensor, y_tensor
            )
            
            # Calcular pérdida
            val_loss = self.criterion(q_values, target_q_values).item()
        
        return val_loss
    
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
            
            # Evaluar en esta muestra
            with torch.no_grad():
                x_cgm_tensor = torch.FloatTensor(x_cgm_bootstrap).to(self.device)
                x_other_tensor = torch.FloatTensor(x_other_bootstrap).to(self.device)
                actions_tensor = torch.FloatTensor(actions_bootstrap).reshape(-1, 1).to(self.device)
                
                q_values = self.q_network(x_cgm_tensor, x_other_tensor, actions_tensor)
                mean_q_value = q_values.mean().item()
                
                self.bootstrap_estimates.append(mean_q_value)
    
    def evaluate_policy(self, policy: Any, 
                       x_cgm_test: np.ndarray, 
                       x_other_test: np.ndarray, 
                       y_actions_test: np.ndarray, # Renamed from y_test for clarity
                       context_test_data: Optional[Dict[str, np.ndarray]] = None,
                       simulator: Optional[GlucoseSimulator] = None) -> Dict[str, float]:
        """
        Evalúa una política utilizando el modelo Q aprendido.
        
        Parámetros:
        -----------
        policy : Any
            Política a evaluar (debe tener un método predict_with_context).
        x_cgm_test : np.ndarray
            Datos CGM de prueba.
        x_other_test : np.ndarray
            Otras características de prueba.
        y_actions_test : np.ndarray
            Acciones reales del conjunto de prueba (usadas para referencia o métricas adicionales).
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
        num_samples = len(x_cgm_test)
        
        if num_samples == 0:
            print_warning("No hay datos de prueba para evaluar la política en FQE.")
            return {}

        predicted_actions_for_policy = np.zeros(num_samples)
        q_values_sum = 0.0

        # Asegurar que x_other_test y context_test_data tengan la misma longitud que x_cgm_test si no son None
        if x_other_test is not None and len(x_other_test) != num_samples:
            print_error(f"La longitud de x_other_test ({len(x_other_test)}) no coincide con x_cgm_test ({num_samples}).")
            # Considerar lanzar un error o ajustar
            return {"error": -1.0}
        if context_test_data is not None:
            for key, arr in context_test_data.items():
                if len(arr) != num_samples:
                    print_error(f"La longitud del contexto '{key}' ({len(arr)}) no coincide con x_cgm_test ({num_samples}).")
                    # Considerar lanzar un error o ajustar
                    return {"error": -1.0}

        for i in tqdm(range(num_samples), desc="Evaluando política con FQE", leave=False):
            x_cgm_sample = x_cgm_test[i]
            # Asegurar que x_other_sample sea un array aunque x_other_test sea None o vacío para este índice
            x_other_sample = x_other_test[i] if x_other_test is not None and i < len(x_other_test) else np.array([])

            # Extraer contexto para predict_with_context
            # CONTEXT_FEATURE_ORDER = ['current_glucose', 'carb_intake', 'iob', 'sleep_quality', 'work_intensity', 'exercise_intensity']
            
            # current_glucose: Usar el último valor de la ventana CGM como proxy si no está en context_test_data
            # Asumiendo que x_cgm_sample tiene forma (timesteps, features_cgm) y la glucosa es la primera característica
            current_glucose_val: float
            if context_test_data and 'current_glucose' in context_test_data and i < len(context_test_data['current_glucose']):
                current_glucose_val = float(context_test_data['current_glucose'][i])
            elif x_cgm_sample.ndim > 0 and x_cgm_sample.shape[0] > 0:
                 # Si x_cgm_sample es (timesteps,) o (timesteps, 1)
                current_glucose_val = float(x_cgm_sample[-1, 0] if x_cgm_sample.ndim == 2 else x_cgm_sample[-1])
            else:
                current_glucose_val = 150.0 # Fallback
                print_warning(f"FQE: No se pudo determinar current_glucose para la muestra {i}, usando fallback {current_glucose_val}.")

            context_values_for_prediction: Dict[str, Optional[float]] = {}
            required_context_keys_for_predict = ['carb_intake', 'iob'] # Mínimos requeridos por DRLModelWrapper
            
            for key_idx, key_name in enumerate(CONTEXT_FEATURE_ORDER):
                if key_name == 'current_glucose': # Ya obtenido
                    continue

                default_val = 0.0
                val = default_val
                if context_test_data and key_name in context_test_data and i < len(context_test_data[key_name]):
                    val = float(context_test_data[key_name][i])
                elif key_name in required_context_keys_for_predict:
                    print_warning(f"FQE: Característica de contexto requerida '{key_name}' no encontrada para la muestra {i}. Usando fallback {default_val}.")
                context_values_for_prediction[key_name] = val
            
            # Llamada a predict_with_context
            try:
                action_pred = policy.predict_with_context(
                    x_cgm=x_cgm_sample.reshape(1, *x_cgm_sample.shape) if x_cgm_sample.ndim < 3 else x_cgm_sample, # Asegurar forma (1, timesteps, features) o (timesteps, features)
                    x_other=x_other_sample.reshape(1, *x_other_sample.shape) if x_other_sample.ndim < 2 and x_other_sample.size > 0 else x_other_sample, # Asegurar forma (1, features)
                    current_glucose=current_glucose_val,
                    carb_intake=context_values_for_prediction.get('carb_intake', 0.0), # type: ignore
                    iob=context_values_for_prediction.get('iob', 0.0), # type: ignore
                    sleep_quality=context_values_for_prediction.get('sleep_quality'),
                    work_intensity=context_values_for_prediction.get('work_intensity'),
                    exercise_intensity=context_values_for_prediction.get('exercise_intensity')
                )
                predicted_actions_for_policy[i] = action_pred
            except Exception as e:
                print_error(f"Error al llamar a policy.predict_with_context en FQE para la muestra {i}: {e}")
                predicted_actions_for_policy[i] = 0.0 # Fallback action
                # Considerar continuar o detener la evaluación

            # Calcular valor Q para la acción predicha por la política
            with torch.no_grad():
                q_input_cgm = torch.FloatTensor(x_cgm_sample).unsqueeze(0).to(self.device)
                q_input_other = torch.FloatTensor(x_other_sample).unsqueeze(0).to(self.device) if x_other_sample.size > 0 else torch.empty(1,0).to(self.device)
                q_input_action = torch.FloatTensor([predicted_actions_for_policy[i]]).unsqueeze(0).to(self.device)
                
                # Aplanar CGM si es necesario para QNetwork
                if q_input_cgm.ndim > 2:
                     q_input_cgm = q_input_cgm.reshape(q_input_cgm.shape[0], -1)
                if q_input_other.ndim > 2:
                     q_input_other = q_input_other.reshape(q_input_other.shape[0], -1)

                q_value = self.q_network(q_input_cgm, q_input_other, q_input_action)
                q_values_sum += q_value.item()
        
        avg_q_value = q_values_sum / num_samples if num_samples > 0 else 0.0
        
        results = {
            'fqe_average_q_value': avg_q_value,
            # Podrías añadir MSE o MAE si y_actions_test son las acciones óptimas o de referencia
            'fqe_mse_actions': float(mean_squared_error(y_actions_test, predicted_actions_for_policy)) if y_actions_test is not None else -1.0,
            'fqe_mae_actions': float(mean_absolute_error(y_actions_test, predicted_actions_for_policy)) if y_actions_test is not None else -1.0,
        }

        # Métricas clínicas si se proporciona simulador
        if simulator:
            # Necesitamos initial_glucose y carb_intake para el simulador
            # Estos deberían venir de context_test_data o ser inferidos
            initial_glucose_test = np.zeros(num_samples)
            carb_intake_test = np.zeros(num_samples)

            for i in range(num_samples):
                if context_test_data and 'current_glucose' in context_test_data and i < len(context_test_data['current_glucose']):
                    initial_glucose_test[i] = context_test_data['current_glucose'][i]
                elif x_cgm_test[i].ndim > 0 and x_cgm_test[i].shape[0] > 0:
                    initial_glucose_test[i] = x_cgm_test[i][-1, 0] if x_cgm_test[i].ndim == 2 else x_cgm_test[i][-1]
                else:
                    initial_glucose_test[i] = 150.0 # Fallback

                if context_test_data and 'carb_intake' in context_test_data and i < len(context_test_data['carb_intake']):
                    carb_intake_test[i] = context_test_data['carb_intake'][i]
                elif context_test_data and 'meal_carbs' in context_test_data and i < len(context_test_data['meal_carbs']): # Alias
                    carb_intake_test[i] = context_test_data['meal_carbs'][i]
                else:
                    carb_intake_test[i] = 0.0 # Fallback
            
            clinical_metrics = evaluate_clinical_metrics(
                simulator=simulator,
                predictions=predicted_actions_for_policy,
                initial_glucose=initial_glucose_test,
                carb_intake=carb_intake_test
            )
            results.update({f"fqe_{k}": v for k, v in clinical_metrics.items()})

        return results

def create_fqe_evaluator(cgm_input_dim: tuple, other_input_dim: tuple):
    """
    Función para crear un evaluador FQE.
    
    Parámetros:
    -----------
    cgm_input_dim : tuple
        Dimensiones de entrada para datos CGM
    other_input_dim : tuple
        Dimensiones de entrada para otras características
        
    Retorna:
    --------
    FittedQEvaluation
        Instancia del evaluador FQE
    """
    return FittedQEvaluation(cgm_input_dim, other_input_dim)