import os
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from config.models_config import BUFFER_CONFIG, EARLY_STOPPING_POLICY
from custom.DeepLearning.dl_pt import DLModelWrapperPyTorch
from custom.DeepReinforcementLearning.drl_pt import DRLModelWrapperPyTorch
from custom.ReinforcementLearning.rl_pt import RLModelWrapperPyTorch
from custom.printer import print_critical, print_error, print_header, print_info, print_debug, print_success, print_warning
from torch.utils.data import Dataset, DataLoader, TensorDataset
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
# from joblib import Parallel, delayed
# from scipy.optimize import minimize
from typing import Dict, List, Tuple, Callable, Optional, Any, Union
from config.params import DEBUG, TRAINING_CONFIG
from custom.early_stopping import ClinicalEarlyStopping
from models.utils.replay_buffer import ReplayBuffer
from training.common import (
    calculate_metrics, evaluate_clinical_metrics, optimize_ensemble_weights_clinical, get_model_type, enhance_features
)
from training.utils import calculate_iob, compute_reward
from constants.constants import (
    CONST_ACTOR_LOSS, CONST_CRITIC_LOSS, CONST_EPSILON, CONST_VAL_LOSS, CONST_LOSS, CONST_METRIC_MAE, CONST_METRIC_RMSE, CONST_METRIC_R2,
    CONST_MODELS, CONST_BEST_PREFIX, CONST_LOGS_DIR, CONST_DEFAULT_EPOCHS, 
    CONST_DEFAULT_BATCH_SIZE, CONST_DEFAULT_SEED, CONST_FIGURES_DIR, CONST_MODEL_TYPES, CONST_DURATION_HOURS, CONTEXT_FEATURE_ORDER, SUBJECT_ID_COL, TIMESTAMP_COL
)
from tqdm.auto import tqdm

from validation.simulator import GlucoseSimulator

# Usar menos épocas en modo debug
CONST_EPOCHS = 2 if DEBUG else CONST_DEFAULT_EPOCHS

# Configurar dispositivo GPU si está disponible
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AllData = Dict[str, Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]]
AllDataFrames = Dict[str, Optional[pl.DataFrame]]

class CGMDataset(Dataset):
    """
    Dataset personalizado para datos CGM y otras características.
    
    Parámetros:
    -----------
    x_cgm : np.ndarray
        Datos CGM con forma (muestras, pasos_tiempo, características)
    x_other : np.ndarray
        Otras características con forma (muestras, características)
    y : np.ndarray
        Valores objetivo con forma (muestras,)
    """
    
    def __init__(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray) -> None:
        self.x_cgm = torch.FloatTensor(x_cgm)
        self.x_other = torch.FloatTensor(x_other)
        self.y = torch.FloatTensor(y).reshape(-1, 1)
        
    def __len__(self) -> int:
        """
        Obtiene la longitud del dataset.
        
        Retorna:
        --------
        int
            Número de muestras en el dataset
        """
        return len(self.y)
        
    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Obtiene un ítem específico del dataset.
        
        Parámetros:
        -----------
        idx : int
            Índice del ítem a obtener
            
        Retorna:
        --------
        Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]
            Tupla con ((x_cgm, x_other), y) para el ítem solicitado
        """
        return ((self.x_cgm[idx], self.x_other[idx]), self.y[idx])


def create_dataloaders(x_cgm: np.ndarray,
                     x_other: Optional[np.ndarray], # x_other can be optional
                     y: np.ndarray,
                     batch_size: int = CONST_DEFAULT_BATCH_SIZE,
                     shuffle: bool = True) -> DataLoader: # Keep if used by DL part
    # ...existing code...
    # Ensure this handles x_other being None if that's a possibility for DL models
    if x_other is None:
        # Create a dummy tensor or adjust TensorDataset if x_other is essential but missing
        # For now, assume if x_other is None, it's not used or handled by the model
        print_debug("x_other is None in create_dataloaders. CGM-only dataset assumed.")
        x_cgm_tensor = torch.tensor(x_cgm, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(x_cgm_tensor, y_tensor) # Adjusted for x_other is None
    else:
        x_cgm_tensor = torch.tensor(x_cgm, dtype=torch.float32)
        x_other_tensor = torch.tensor(x_other, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(x_cgm_tensor, x_other_tensor, y_tensor)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

def _prepare_model(model_wrapper: Union[nn.Module, Any],
                 x_cgm_train: np.ndarray,
                 x_other_train: Optional[np.ndarray],
                 y_train: np.ndarray) -> nn.Module:
    """
    Prepara el modelo para entrenamiento, manejando wrappers si es necesario.
    
    Parámetros:
    -----------
    model_wrapper : Union[nn.Module, Any]
        Modelo a preparar (puede ser un wrapper)
    x_cgm_train : np.ndarray
        Datos CGM de entrenamiento
    x_other_train : np.ndarray
        Otras características de entrenamiento
    y_train : np.ndarray
        Valores objetivo de entrenamiento
        
    Retorna:
    --------
    nn.Module
        Modelo preparado para entrenamiento
    """
    # Verificar si es cualquier tipo de wrapper con método start y atributo model
    if hasattr(model_wrapper, 'start') and hasattr(model_wrapper, 'model'):
        # Si model es un wrapper, inicializar y usar el modelo interno
        model_wrapper.start(x_cgm_train, x_other_train, y_train)
        actual_model = model_wrapper.model
    else:
        # Si es un modelo PyTorch normal
        actual_model = model_wrapper
    
    # Transferir modelo al dispositivo
    return actual_model.to(DEVICE)


def _run_train_epoch(model_wrapper: nn.Module,
                    train_loader: DataLoader,
                    optimizer: optim.Optimizer,
                    criterion: nn.Module) -> float:
    """
    Ejecuta una época de entrenamiento y devuelve la pérdida promedio.
    
    Parámetros:
    -----------
    model_wrapper : nn.Module
        Modelo a entrenar
    train_loader : DataLoader
        DataLoader para el conjunto de entrenamiento
    optimizer : optim.Optimizer
        Optimizador para el modelo
    criterion : nn.Module
        Función de pérdida para el entrenamiento
        
    Retorna:
    --------
    float
        Pérdida promedio de la época
    """
    model_wrapper.model.train()
    train_loss = 0.0
    
    # Añadir barra de progreso para los batches
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), 
                       desc="Batches", leave=False)
    
    for batch_idx, ((x_cgm_batch, x_other_batch), y_batch) in progress_bar:
        # Transferir datos al dispositivo
        x_cgm_batch = x_cgm_batch.to(DEVICE)
        x_other_batch = x_other_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        
        # Poner gradientes a cero
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model_wrapper(x_cgm_batch, x_other_batch)
        
        # Asegurar que outputs tiene forma [batch_size, 1]
        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(1)
            
        loss = criterion(outputs, y_batch)
        
        # Backward pass y optimización
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_wrapper.parameters(), 1.0)
        optimizer.step()
        
        # Acumular pérdida
        train_loss += loss.item()
        
        # Actualizar barra de progreso con la pérdida actual
        if batch_idx % 5 == 0:  # Actualizar cada 5 batches para no sobrecargar
            progress_bar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
    
    return train_loss / len(train_loader)


def _run_validation(model_wrapper: nn.Module,
                  val_loader: DataLoader,
                  criterion: nn.Module) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Ejecuta validación y devuelve pérdida, predicciones y valores reales.
    
    Parámetros:
    -----------
    model_wrapper: nn.Module
        Modelo a validar
    val_loader : DataLoader
        DataLoader para el conjunto de validación
    criterion : nn.Module
        Función de pérdida para la validación
        
    Retorna:
    --------
    Tuple[float, np.ndarray, np.ndarray]
        (pérdida promedio, predicciones, valores reales)
    """
    model_wrapper.model.eval()
    val_loss = 0.0
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for (x_cgm_batch, x_other_batch), y_batch in val_loader:
            # Transferir datos al dispositivo
            x_cgm_batch = x_cgm_batch.to(DEVICE)
            x_other_batch = x_other_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            # Forward pass
            outputs = model_wrapper(x_cgm_batch, x_other_batch)
            
            # Asegurar que outputs tiene forma [batch_size, 1]
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(1)  # Convertir [batch_size] a [batch_size, 1]
            
            loss = criterion(outputs, y_batch)
            
            # Acumular pérdida y predicciones
            val_loss += loss.item()
            val_preds.append(outputs.cpu().numpy())
            val_targets.append(y_batch.cpu().numpy())

    # Calcular pérdida promedio de validación
    avg_val_loss = val_loss / len(val_loader)
    
    # Preparar arrays para métricas
    val_preds_np = np.vstack([pred.reshape(-1, 1) if pred.ndim == 1 else pred for pred in val_preds]).flatten()
    val_targets_np = np.vstack([targ.reshape(-1, 1) if targ.ndim == 1 else targ for targ in val_targets]).flatten()
    
    return avg_val_loss, val_preds_np, val_targets_np


def _predict_in_batches(model_wrapper: nn.Module,
                      x_cgm: np.ndarray,
                      x_other: Optional[np.ndarray],
                      batch_size: int = 64) -> np.ndarray:
    """
    Realiza predicciones en lotes para evitar problemas de memoria.
    
    Parámetros:
    -----------
    model_wrapper : nn.Module
        Modelo a usar para predicciones
    x_cgm : np.ndarray
        Datos CGM para predicción
    x_other : np.ndarray
        Otras características para predicción
    batch_size : int, opcional
        Tamaño del batch para predicción (default: 64)
        
    Retorna:
    --------
    np.ndarray
        Predicciones del modelo
    """
    model_wrapper.model.eval()
    with torch.no_grad():
        # Convertir datos a tensores
        x_cgm_tensor = torch.FloatTensor(x_cgm).to(DEVICE)
        x_other_tensor = torch.FloatTensor(x_other).to(DEVICE)
        
        preds = []
        
        for i in range(0, len(x_cgm), batch_size):
            end_idx = min(i + batch_size, len(x_cgm))
            batch_cgm = x_cgm_tensor[i:end_idx]
            batch_other = x_other_tensor[i:end_idx]
            
            outputs = model_wrapper(batch_cgm, batch_other)
            preds.append(outputs.cpu().numpy())
        
        return np.vstack(preds).flatten()

def estimate_prediction_uncertainty(model_wrapper: Union[nn.Module, DLModelWrapperPyTorch, RLModelWrapperPyTorch, DRLModelWrapperPyTorch],
                                  x_cgm: np.ndarray,
                                  x_other: np.ndarray,
                                  n_samples: int = 10) -> np.ndarray:
    """
    Estima la incertidumbre de las predicciones usando Monte Carlo Dropout.
    
    Parámetros:
    -----------
    model_wrapper : Union[nn.Module, DLModelWrapperPyTorch, RLModelWrapperPyTorch, DRLModelWrapperPyTorch]
        Modelo para estimar incertidumbre
    x_cgm : np.ndarray
        Datos CGM para predicción
    x_other : np.ndarray
        Otras características para predicción
    n_samples : int, opcional
        Número de muestras para Monte Carlo Dropout (default: 10)
        
    Retorna:
    --------
    np.ndarray
        Desviación estándar de las predicciones para cada muestra
    """
    # Preparar el modelo
    if hasattr(model_wrapper, 'model'):
        actual_model = model_wrapper.model
    else:
        actual_model = model_wrapper
    
    actual_model = actual_model.to(DEVICE)
    
    # Activar modo de evaluación pero mantener dropout activo
    actual_model.train()
    
    # Convertir datos a tensores
    x_cgm_tensor = torch.FloatTensor(x_cgm).to(DEVICE)
    x_other_tensor = torch.FloatTensor(x_other).to(DEVICE)
    
    # Realizar múltiples pases para obtener distribución de predicciones
    predictions = []
    batch_size = 64
    
    with torch.no_grad():
        for _ in range(n_samples):
            batch_preds = []
            for i in range(0, len(x_cgm), batch_size):
                end_idx = min(i + batch_size, len(x_cgm))
                batch_cgm = x_cgm_tensor[i:end_idx]
                batch_other = x_other_tensor[i:end_idx]
                
                outputs = actual_model(batch_cgm, batch_other)
                batch_preds.append(outputs.cpu().numpy())
            
            # Concatenar predicciones de todos los batches
            full_preds = np.vstack(batch_preds).flatten()
            predictions.append(full_preds)
    
    # Calcular desviación estándar a lo largo de las muestras
    predictions_array = np.array(predictions)
    uncertainty = np.std(predictions_array, axis=0)
    
    return uncertainty


def adjust_predictions_with_uncertainty(predictions: np.ndarray, 
                                       uncertainty: np.ndarray,
                                       safety_factor: float = 2.0,
                                       min_dose: float = 0.0,
                                       confidence_threshold: float = 0.3) -> np.ndarray:
    """
    Ajusta las predicciones en base a la incertidumbre estimada, aplicando restricciones clínicas.
    
    Parámetros:
    -----------
    predictions : np.ndarray
        Predicciones originales del modelo
    uncertainty : np.ndarray
        Estimación de incertidumbre para cada predicción
    safety_factor : float, opcional
        Factor de seguridad para el ajuste (default: 2.0)
    min_dose : float, opcional
        Dosis mínima permitida (default: 0.0)
    confidence_threshold : float, opcional
        Umbral de confianza para aplicar restricciones severas (default: 0.3)
        
    Retorna:
    --------
    np.ndarray
        Predicciones ajustadas con criterio de seguridad
    """
    # Normalizar incertidumbre para comparabilidad
    norm_uncertainty = uncertainty / (np.max(uncertainty) + 1e-6)
    
    # Ajustar predicciones basado en incertidumbre
    # Mayor incertidumbre = dosis más conservadora (menor)
    adjustment_factor = 1.0 - (norm_uncertainty * safety_factor)
    
    # Limitar factor de ajuste para evitar dosis negativas o excesivamente pequeñas
    adjustment_factor = np.clip(adjustment_factor, 0.1, 1.0)
    
    # Aplicar ajuste
    adjusted_predictions = predictions * adjustment_factor
    
    # Restricciones adicionales para alta incertidumbre
    high_uncertainty_mask = norm_uncertainty > confidence_threshold
    
    # Para casos de alta incertidumbre, usar dosis mínima segura
    adjusted_predictions[high_uncertainty_mask] = min_dose
    
    # Asegurar límites de seguridad (no dosis negativas)
    adjusted_predictions = np.maximum(adjusted_predictions, min_dose)
    
    return adjusted_predictions

def _setup_training_config(training_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Configura los parámetros de entrenamiento con valores por defecto."""
    if training_config is None:
        return {
            'epochs': CONST_EPOCHS,
            'batch_size': CONST_DEFAULT_BATCH_SIZE,
            'learning_rate': 0.001,
            'patience': 30,
            'monitor': 'time_in_range',
            'mode': 'max'
        }
    return training_config

def _extract_training_data(data: AllData) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str,np.ndarray]], Optional[Dict[str,np.ndarray]], Optional[Dict[str,np.ndarray]]]:
    """Extrae datos de entrenamiento, validación y prueba, incluyendo contexto."""
    train_split = data.get('train')
    val_split = data.get('val')
    test_split = data.get('test')

    x_cgm_train, x_other_train, y_train, context_train = None, None, None, None
    if train_split:
        x_cgm_train = train_split.get('x_cgm')
        x_other_train = train_split.get('x_other')
        y_train = train_split.get('y')
        context_train = train_split.get('context')

    x_cgm_val, x_other_val, y_val, context_val = None, None, None, None
    if val_split:
        x_cgm_val = val_split.get('x_cgm')
        x_other_val = val_split.get('x_other')
        y_val = val_split.get('y')
        context_val = val_split.get('context')

    x_cgm_test, x_other_test, y_test, context_test = None, None, None, None
    if test_split:
        x_cgm_test = test_split.get('x_cgm')
        x_other_test = test_split.get('x_other')
        y_test = test_split.get('y')
        context_test = test_split.get('context')
        
    return (x_cgm_train, x_other_train, y_train, 
            x_cgm_val, x_other_val, y_val, 
            x_cgm_test, x_other_test, y_test,
            context_train, context_val, context_test)

def _extract_carb_intake_data(data: AllData, x_other_val: np.ndarray, x_other_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extrae los datos de ingesta de carbohidratos del contexto o x_other."""
    context_val_data = data['val'].get('context', {})
    context_test_data = data['test'].get('context', {})

    carb_intake_val = np.array(context_val_data.get('carb_intake', 
                                                [x_other_val[i, 0] if x_other_val.shape[1] > 0 else 0.0 for i in range(len(x_other_val))]))
    carb_intake_test = np.array(context_test_data.get('carb_intake', 
                                                 [x_other_test[i, 0] if x_other_test.shape[1] > 0 else 0.0 for i in range(len(x_other_test))]))
    
    return carb_intake_val, carb_intake_test

def _setup_training_components(actual_model: nn.Module, learning_rate: float, patience: int) -> Tuple[optim.Optimizer, nn.Module, optim.lr_scheduler.ReduceLROnPlateau]:
    """Configura optimizador, función de pérdida y scheduler."""
    optimizer = optim.Adam(actual_model.parameters(), lr=learning_rate, weight_decay=1e-6)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=patience // 2, min_lr=1e-6
    )
    return optimizer, criterion, scheduler

def _process_validation_epoch(actual_model: nn.Module, val_loader: DataLoader, criterion: nn.Module,
                            x_cgm_val: np.ndarray, x_other_val: np.ndarray, simulator: GlucoseSimulator,
                            initial_glucose_val: np.ndarray, carb_intake_val: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """Procesa una época de validación y retorna métricas."""
    avg_val_loss, _val_preds_np, _val_targets_np = _run_validation(actual_model, val_loader, criterion)
    val_preds_full = _predict_in_batches(actual_model, x_cgm_val, x_other_val)
    
    clinical_metrics_val = evaluate_clinical_metrics(
        simulator=simulator,
        predictions=val_preds_full,
        initial_glucose=initial_glucose_val,
        carb_intake=carb_intake_val
    )
    
    return avg_val_loss, clinical_metrics_val

def _update_training_history(history: Dict[str, List[float]], avg_train_loss: float, 
                           avg_val_loss: float, clinical_metrics_val: Dict[str, float]) -> None:
    """Actualiza el historial de entrenamiento con las métricas."""
    history['loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['time_in_range'].append(clinical_metrics_val['time_in_range'])
    history['time_below_range'].append(clinical_metrics_val['time_below_range'])
    history['time_above_range'].append(clinical_metrics_val['time_above_range'])
    history['time_severe_below'].append(clinical_metrics_val['time_severe_below'])
    history['time_severe_above'].append(clinical_metrics_val['time_severe_above'])

def _get_monitor_value(monitor: str, clinical_metrics_val: Dict[str, float], avg_val_loss: float) -> float:
    """Determina el valor de la métrica a monitorear para early stopping."""
    if monitor == 'time_in_range':
        return clinical_metrics_val['time_in_range']
    elif monitor == 'val_loss':
        return -avg_val_loss
    else:
        return float(clinical_metrics_val.get(monitor, -avg_val_loss))

def _generate_final_predictions(actual_model: nn.Module, model_wrapper: Union[nn.Module, DLModelWrapperPyTorch, RLModelWrapperPyTorch, DRLModelWrapperPyTorch],
                              x_cgm_test: np.ndarray, x_other_test: np.ndarray) -> np.ndarray:
    """Genera predicciones finales con estimación de incertidumbre."""
    y_pred = _predict_in_batches(actual_model, x_cgm_test, x_other_test)
    
    uncertainty_estimates: np.ndarray = np.array([])
    if hasattr(model_wrapper, 'model') and model_wrapper.model is not None:
        uncertainty_estimates = estimate_prediction_uncertainty(model_wrapper.model, x_cgm_test, x_other_test)
    elif isinstance(model_wrapper, nn.Module):
        uncertainty_estimates = estimate_prediction_uncertainty(model_wrapper, x_cgm_test, x_other_test)

    safe_predictions = y_pred
    if uncertainty_estimates.size > 0:
        safe_predictions = adjust_predictions_with_uncertainty(y_pred, uncertainty_estimates)
    
    return safe_predictions

def train_and_evaluate_model_supervised(model_wrapper: Union[DLModelWrapperPyTorch, RLModelWrapperPyTorch], # Adjusted for wrappers
                          model_name: str,
                          data: AllData, models_dir: str = CONST_MODELS,
                          training_config: Optional[Dict[str, Any]] = None
                          ) -> Tuple[Dict[str, List[float]], Optional[np.ndarray], Dict[str, float], Union[DLModelWrapperPyTorch, RLModelWrapperPyTorch]]:
    """
    Entrena y evalúa un modelo supervisado (DL o RL clásico que se entrene supervisado).
    Esta función encapsula el bucle de entrenamiento por épocas para modelos supervisados.
    """
    print_header(f"Entrenando modelo supervisado: {model_name}")
    
    cfg = _setup_training_config(training_config)
    epochs = cfg['epochs']
    batch_size = cfg['batch_size']
    learning_rate = cfg['learning_rate']
    patience = cfg['patience']
    monitor_metric = cfg['monitor']
    # ... (extract other configs)

    (x_cgm_train, x_other_train, y_train,
     x_cgm_val, x_other_val, y_val,
     x_cgm_test, x_other_test, y_test,
     context_train, context_val, context_test) = _extract_training_data(data)

    if x_cgm_train is None or y_train is None:
        print_error(f"Datos de entrenamiento insuficientes para {model_name}.")
        return {}, None, {}, model_wrapper # Return the un-trained wrapper

    # Create DataLoaders
    train_loader = create_dataloaders(x_cgm_train, x_other_train, y_train, batch_size, shuffle=True)
    val_loader = None
    if x_cgm_val is not None and y_val is not None:
        val_loader = create_dataloaders(x_cgm_val, x_other_val, y_val, batch_size, shuffle=False)

    # Model, Optimizer, Criterion, Scheduler (assuming model_wrapper.model is the nn.Module)
    actual_model = model_wrapper.model if hasattr(model_wrapper, 'model') else model_wrapper
    if not isinstance(actual_model, nn.Module):
        print_error(f"El modelo envuelto en {model_name} no es un nn.Module.")
        return {}, None, {}, model_wrapper

    actual_model.to(DEVICE)
    optimizer, criterion, scheduler = _setup_training_components(actual_model, learning_rate, patience)
    
    # Early Stopping
    early_stopping = None
    if EARLY_STOPPING_POLICY.get('early_stopping', True): # Use global or from cfg
        early_stopping = ClinicalEarlyStopping( # Or standard EarlyStopping
            patience=cfg.get('early_stopping_patience', EARLY_STOPPING_POLICY['early_stopping_patience']),
            min_delta=cfg.get('early_stopping_min_delta', EARLY_STOPPING_POLICY['early_stopping_min_delta']),
            restore_best_weights=cfg.get('early_stopping_restore_best_weights', EARLY_STOPPING_POLICY['early_stopping_restore_best_weights']),
            monitor=monitor_metric, # e.g. 'val_loss' or a clinical one
            mode=cfg.get('mode', 'min' if 'loss' in monitor_metric else 'max')
        )
        print_info(f"Early stopping configurado para {model_name}: monitor='{monitor_metric}', patience={early_stopping.patience}")

    history: Dict[str, List[float]] = {CONST_LOSS: [], CONST_VAL_LOSS: [], monitor_metric: []}
    # Add other metrics if needed, e.g. clinical metrics from validation

    # Simulator for clinical metrics during validation (if applicable)
    simulator = GlucoseSimulator() # Default params
    
    # Training loop
    for epoch in range(epochs):
        print_info(f"Epoch {epoch+1}/{epochs} para {model_name}")
        actual_model.train()
        epoch_train_losses = []
        for batch_idx, batch_data in enumerate(tqdm(train_loader, desc="Training Batch")):
            # Adapt batch_data unpacking based on create_dataloaders output
            if x_other_train is not None:
                batch_x_cgm, batch_x_other, batch_y = batch_data
                batch_x_other = batch_x_other.to(DEVICE)
            else:
                batch_x_cgm, batch_y = batch_data
                batch_x_other = None

            batch_x_cgm, batch_y = batch_x_cgm.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            # Forward pass might differ if model_wrapper has specific forward
            if hasattr(model_wrapper, 'forward_train'): # custom forward for training
                 outputs = model_wrapper.forward_train(batch_x_cgm, batch_x_other)
            elif x_other_train is not None:
                 outputs = actual_model(batch_x_cgm, batch_x_other)
            else:
                 outputs = actual_model(batch_x_cgm) # Assumes model can handle x_other=None

            loss = criterion(outputs, batch_y.unsqueeze(1) if batch_y.ndim == 1 else batch_y)
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())
        
        avg_train_loss = np.mean(epoch_train_losses)
        history[CONST_LOSS].append(avg_train_loss)
        print_info(f"Epoch {epoch+1} - Training Loss: {avg_train_loss:.4f}")

        # Validation
        avg_val_loss = float('nan')
        current_monitor_value = float('-inf') if early_stopping and early_stopping.mode == 'max' else float('inf')

        if val_loader:
            actual_model.eval()
            epoch_val_losses = []
            all_val_preds, all_val_true = [], []
            with torch.no_grad():
                for batch_data in tqdm(val_loader, desc="Validation Batch"):
                    if x_other_val is not None:
                        batch_x_cgm_v, batch_x_other_v, batch_y_v = batch_data
                        batch_x_other_v = batch_x_other_v.to(DEVICE)
                    else:
                        batch_x_cgm_v, batch_y_v = batch_data
                        batch_x_other_v = None
                    
                    batch_x_cgm_v, batch_y_v = batch_x_cgm_v.to(DEVICE), batch_y_v.to(DEVICE)

                    if hasattr(model_wrapper, 'forward_eval'):
                        val_outputs = model_wrapper.forward_eval(batch_x_cgm_v, batch_x_other_v)
                    elif x_other_val is not None:
                        val_outputs = actual_model(batch_x_cgm_v, batch_x_other_v)
                    else:
                        val_outputs = actual_model(batch_x_cgm_v)
                        
                    val_loss = criterion(val_outputs, batch_y_v.unsqueeze(1) if batch_y_v.ndim == 1 else batch_y_v)
                    epoch_val_losses.append(val_loss.item())
                    all_val_preds.append(val_outputs.cpu().numpy())
                    all_val_true.append(batch_y_v.cpu().numpy())

            avg_val_loss = np.mean(epoch_val_losses)
            history[CONST_VAL_LOSS].append(avg_val_loss)
            print_info(f"Epoch {epoch+1} - Validation Loss: {avg_val_loss:.4f}")
            
            # Calculate clinical metrics on validation set if context is available
            val_preds_np = np.concatenate(all_val_preds)
            # y_val_np = np.concatenate(all_val_true) # Not used for clinical metrics directly here

            clinical_metrics_val = {}
            if context_val and 'glucose_last' in context_val and ('meal_carbs' in context_val or 'carb_intake' in context_val):
                initial_glucose_v = context_val['glucose_last']
                carb_intake_v = context_val.get('meal_carbs', context_val.get('carb_intake'))
                
                # Ensure lengths match for clinical eval
                num_val_samples = len(val_preds_np)
                if initial_glucose_v.shape[0] != num_val_samples or carb_intake_v.shape[0] != num_val_samples:
                    print_warning(f"Validation context data length mismatch for {model_name}. Skipping clinical metrics for validation.")
                else:
                    clinical_metrics_val = evaluate_clinical_metrics(
                        simulator, 
                        val_preds_np.ravel()[:num_val_samples], 
                        initial_glucose_v.ravel()[:num_val_samples], 
                        carb_intake_v.ravel()[:num_val_samples]
                    )
                    for m_name, m_val in clinical_metrics_val.items():
                        history.setdefault(f"val_{m_name}", []).append(m_val)
                    print_info(f"Epoch {epoch+1} - Validation Clinical Metrics: {clinical_metrics_val}")

            # Determine monitor value for early stopping
            if monitor_metric == CONST_VAL_LOSS:
                current_monitor_value = avg_val_loss
            elif f"val_{monitor_metric}" in history: # if monitor is a clinical metric
                current_monitor_value = history[f"val_{monitor_metric}"][-1]
            else: # Fallback if monitor metric not found
                current_monitor_value = avg_val_loss 
            
            history.setdefault(monitor_metric, []).append(current_monitor_value) # Store the actual value being monitored

            if early_stopping and early_stopping(actual_model, current_monitor_value):
                print_info(f"Early stopping activado en la época {epoch+1} para {model_name}.")
                break
            if scheduler: # Assuming ReduceLROnPlateau
                 scheduler.step(current_monitor_value)
        else: # No validation loader
            if early_stopping and early_stopping.monitor != CONST_LOSS : # Cannot do early stopping without val unless monitoring train loss
                 print_warning(f"Early stopping for {model_name} needs validation data or monitor='{CONST_LOSS}'.")
            elif early_stopping and early_stopping.monitor == CONST_LOSS:
                 if early_stopping(actual_model, avg_train_loss):
                    print_info(f"Early stopping activado en la época {epoch+1} para {model_name} (monitoreando loss de entrenamiento).")
                    break
            if scheduler: # Step scheduler with training loss if no validation
                 scheduler.step(avg_train_loss)


    # Test set evaluation (after training loop)
    predictions_test_np: Optional[np.ndarray] = None
    metrics_test: Dict[str, float] = {}
    if x_cgm_test is not None and y_test is not None:
        print_info(f"Evaluando {model_name} en el conjunto de prueba...")
        actual_model.eval() # Ensure model is in eval mode
        
        # Create test DataLoader
        test_loader = create_dataloaders(x_cgm_test, x_other_test, y_test, batch_size, shuffle=False)
        all_test_preds_list = []
        with torch.no_grad():
            for batch_data in tqdm(test_loader, desc="Test Batch"):
                if x_other_test is not None:
                    batch_x_cgm_t, batch_x_other_t, _ = batch_data # y_true from test_loader not used for preds
                    batch_x_other_t = batch_x_other_t.to(DEVICE)
                else:
                    batch_x_cgm_t, _ = batch_data
                    batch_x_other_t = None
                batch_x_cgm_t = batch_x_cgm_t.to(DEVICE)

                if hasattr(model_wrapper, 'forward_eval'):
                    test_outputs = model_wrapper.forward_eval(batch_x_cgm_t, batch_x_other_t)
                elif x_other_test is not None:
                    test_outputs = actual_model(batch_x_cgm_t, batch_x_other_t)
                else:
                    test_outputs = actual_model(batch_x_cgm_t)
                all_test_preds_list.append(test_outputs.cpu().numpy())
        
        predictions_test_np = np.concatenate(all_test_preds_list)

        # Standard regression metrics
        metrics_test = calculate_metrics(y_test.ravel(), predictions_test_np.ravel()) # from training.common
        
        # Clinical metrics on test set
        if context_test and 'glucose_last' in context_test and ('meal_carbs' in context_test or 'carb_intake' in context_test):
            initial_glucose_t = context_test['glucose_last']
            carb_intake_t = context_test.get('meal_carbs', context_test.get('carb_intake'))
            
            num_test_samples = len(y_test)
            if initial_glucose_t.shape[0] != num_test_samples or carb_intake_t.shape[0] != num_test_samples:
                 print_warning(f"Test context data length mismatch for {model_name}. Skipping clinical metrics for test set.")
            else:
                clinical_metrics_test = evaluate_clinical_metrics(
                    simulator, 
                    predictions_test_np.ravel()[:num_test_samples], 
                    initial_glucose_t.ravel()[:num_test_samples], 
                    carb_intake_t.ravel()[:num_test_samples]
                )
                metrics_test.update(clinical_metrics_test)
        print_info(f"Métricas de prueba finales para {model_name}: {metrics_test}")

    # Save model (usually done by train_model_sequential, but if this is the main DL train func, save here)
    # model_wrapper.save(os.path.join(models_dir, f"{model_name}.pt"))
    
    return history, predictions_test_np, metrics_test, model_wrapper

def train_model_sequential(
    model_creator_func: Callable[..., Union[DLModelWrapperPyTorch, RLModelWrapperPyTorch, DRLModelWrapperPyTorch]],
    model_name: str,
    data_dfs: AllDataFrames,
    models_dir: str = CONST_MODELS,
    training_config: Dict[str, Any] = TRAINING_CONFIG,
    feature_config: Optional[Dict[str, List[str]]] = None
) -> Tuple[Dict[str, List[float]], Optional[np.ndarray], Dict[str, float], Optional[Union[DLModelWrapperPyTorch, RLModelWrapperPyTorch, DRLModelWrapperPyTorch]]]:
    """
    Entrena y evalúa un único modelo secuencialmente (sin k-fold).
    Adaptado para trabajar con DataFrames y pasar feature_config.
    """
    print_info(f"Entrenando modelo secuencial: {model_name} con DataFrames.")
    
    # The model_creator_func (e.g., create_ddpg_model) must accept feature_config.
    # DRLModelWrapperPyTorch's __init__ takes feature_config.
    # The create_xxx_model functions (like create_ddpg_model in ddpg.py) are updated to pass it.
    model_wrapper = model_creator_func(feature_config=feature_config)

    train_df = data_dfs.get('train')
    val_df = data_dfs.get('val')
    test_df = data_dfs.get('test')

    if train_df is None or train_df.is_empty():
        print_error(f"DataFrame de entrenamiento está vacío o no proporcionado para {model_name}.")
        return {}, None, {}, None

    # Start/initialize model (DRL wrapper's start method)
    # This populates the replay buffer for DRL models.
    model_wrapper.start(train_df=train_df)

    # Fit model (DRL wrapper's fit method)
    history = model_wrapper.fit(
        train_df=train_df, # Passed for reference, DRL fit uses its buffer
        val_df=val_df,
        epochs=training_config.get("epochs", training_config.get("episodes", CONST_DEFAULT_EPOCHS)),
        batch_size=training_config.get("batch_size", CONST_DEFAULT_BATCH_SIZE),
        verbose=1
    )

    # Save model
    model_save_path = os.path.join(models_dir, f"{model_name}.pt")
    model_wrapper.save(model_save_path)
    print_success(f"Modelo {model_name} guardado en {model_save_path}")

    predictions_np: Optional[np.ndarray] = None
    clinical_metrics: Dict[str, float] = {}

    if test_df is not None and not test_df.is_empty():
        predictions_np = model_wrapper.predict(test_df)
        
        simulator = GlucoseSimulator(
            insulin_sensitivity=training_config.get("simulator_insulin_sensitivity", 50),
            carb_ratio=training_config.get("simulator_carb_ratio", 10)
        )
        
        clinical_metrics = model_wrapper.evaluate_clinical(
            df=test_df,
            simulator=simulator,
            simulation_hours=training_config.get("simulation_hours", CONST_DURATION_HOURS)
        )
        print_info(f"Métricas clínicas para {model_name} en datos de test: {clinical_metrics}")
    else:
        print_warning(f"No hay datos de test para evaluar {model_name} o están vacíos.")

    return history, predictions_np, clinical_metrics, model_wrapper

def cross_validate_model(create_model_fn: Callable, 
                       x_cgm: np.ndarray, 
                       x_other: np.ndarray, 
                       y: np.ndarray, 
                       n_splits: int = 5, 
                       models_dir: str = CONST_MODELS) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Realiza validación cruzada para un modelo.
    
    Parámetros:
    -----------
    create_model_fn : Callable
        Función que crea el modelo
    x_cgm : np.ndarray
        Datos CGM
    x_other : np.ndarray
        Otras características
    y : np.ndarray
        Valores objetivo
    n_splits : int, opcional
        Número de divisiones para validación cruzada (default: 5)
    models_dir : str, opcional
        Directorio para guardar modelos (default: "models")
        
    Retorna:
    --------
    Tuple[Dict[str, float], Dict[str, float]]
        (métricas_promedio, métricas_desviación)
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=CONST_DEFAULT_SEED)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(x_cgm)):
        print(f"\nEntrenando fold {fold + 1}/{n_splits}")
        
        # Dividir datos
        x_cgm_train_fold = x_cgm[train_idx]
        x_cgm_val_fold = x_cgm[val_idx]
        x_other_train_fold = x_other[train_idx]
        x_other_val_fold = x_other[val_idx]
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]
        
        # Crear modelo
        model = create_model_fn()
        
        # Organizar datos en estructura esperada
        data = {
            'train': {'x_cgm': x_cgm_train_fold, 'x_other': x_other_train_fold, 'y': y_train_fold},
            'val': {'x_cgm': x_cgm_val_fold, 'x_other': x_other_val_fold, 'y': y_val_fold},
            'test': {'x_cgm': x_cgm_val_fold, 'x_other': x_other_val_fold, 'y': y_val_fold}
        }
        
        # Entrenar y evaluar modelo
        _, _, metrics = train_and_evaluate_model_supervised(
            model_wrapper=model,
            model_name=f'fold_{fold}',
            data=data,
            models_dir=models_dir
        )
        
        scores.append(metrics)
        
        # Limpiar memoria
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Calcular estadísticas
    mean_scores = {
        metric: np.mean([s[metric] for s in scores])
        for metric in scores[0].keys()
    }
    std_scores = {
        metric: np.std([s[metric] for s in scores])
        for metric in scores[0].keys()
    }
    
    return mean_scores, std_scores


def predict_model(model_path: str, 
                model_creator: Callable, 
                x_cgm: np.ndarray, 
                x_other: np.ndarray, 
                input_shapes: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]] = None) -> np.ndarray:
    """
    Realiza predicciones con un modelo guardado.
    
    Parámetros:
    -----------
    model_path : str
        Ruta al modelo guardado
    model_creator : Callable
        Función que crea el modelo
    x_cgm : np.ndarray
        Datos CGM para predicción
    x_other : np.ndarray
        Otras características para predicción
    input_shapes : Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]], opcional
        Formas de las entradas. Si es None, se infieren (default: None)
        
    Retorna:
    --------
    np.ndarray
        Predicciones del modelo
    """
    # Determinar formas de entrada si no se proporcionan
    if input_shapes is None:
        input_shapes = ((x_cgm.shape[1:]), (x_other.shape[1:]))
    
    # Crear modelo
    model = model_creator(input_shapes[0], input_shapes[1])
    
    # Cargar pesos guardados
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    
    # Hacer predicciones en lotes para evitar problemas de memoria
    with torch.no_grad():
        x_cgm_tensor = torch.FloatTensor(x_cgm).to(DEVICE)
        x_other_tensor = torch.FloatTensor(x_other).to(DEVICE)
        
        batch_size = 64
        predictions = []
        
        for i in range(0, len(x_cgm), batch_size):
            end_idx = min(i + batch_size, len(x_cgm))
            batch_cgm = x_cgm_tensor[i:end_idx]
            batch_other = x_other_tensor[i:end_idx]
            
            outputs = model(batch_cgm, batch_other)
            predictions.append(outputs.cpu().numpy())
    
    # Limpiar memoria
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return np.vstack(predictions).flatten()

def _get_patient_specific_data(
    global_data_split: Optional[Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]],
    patient_id: Any,
    split_name: str
) -> Optional[Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]]:
    """Obtiene datos específicos del paciente, incluyendo el diccionario de contexto."""
    if global_data_split is None:
        print_warning(f"Datos globales para split '{split_name}' es None. No se pueden obtener datos para paciente {patient_id}.")
        return None

    if 'subject_id' not in global_data_split:
        print_warning(f"'subject_id' no encontrado en split '{split_name}'. No se pueden filtrar datos para paciente {patient_id}.")
        # Si no hay subject_id, podría ser un dataset no particionable por paciente o un error.
        # Dependiendo de la lógica, se podría devolver global_data_split si es un modelo general,
        # o None si se espera filtrado por paciente. Para _get_patient_specific_data, devolvemos None.
        return None
        
    subject_ids_array = global_data_split['subject_id']
    if not isinstance(subject_ids_array, np.ndarray):
        print_warning(f"'subject_id' en split '{split_name}' no es un np.ndarray. No se puede filtrar.")
        return None

    patient_mask = (subject_ids_array == patient_id)
    if not np.any(patient_mask):
        print_debug(f"Paciente {patient_id} no encontrado en split '{split_name}'.")
        return None

    patient_data: Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]] = {}
    for key, value in global_data_split.items():
        if key == 'subject_id': # Ya usado para la máscara, se puede re-añadir si es necesario
            filtered_value = value[patient_mask]
            if filtered_value.size > 0:
                 patient_data[key] = filtered_value
            else: # Should not happen if patient_mask has True values
                 print_debug(f"No subject_id data for patient {patient_id} in split {split_name} after filtering, though mask was positive.")
                 # patient_data[key] = np.array([]) # or skip
        elif key == 'context' and isinstance(value, dict):
            # Filtrar cada array dentro del diccionario de contexto
            filtered_context_dict: Dict[str, np.ndarray] = {}
            for ctx_key, ctx_array in value.items():
                if isinstance(ctx_array, np.ndarray) and ctx_array.shape[0] == len(subject_ids_array):
                    filtered_ctx_array = ctx_array[patient_mask]
                    if filtered_ctx_array.size > 0:
                        filtered_context_dict[ctx_key] = filtered_ctx_array
                    else:
                        # print_debug(f"Context key '{ctx_key}' for patient {patient_id} in split {split_name} resulted in empty array after filtering.")
                        # filtered_context_dict[ctx_key] = np.array([]) # Store empty or skip
                        pass # Skip if empty to avoid issues downstream if features are expected
                elif isinstance(ctx_array, np.ndarray): # Mismatch in length
                     print_warning(f"Context key '{ctx_key}' array length {ctx_array.shape[0]} does not match subject_ids length {len(subject_ids_array)} in split {split_name}. Skipping this context feature for patient {patient_id}.")
                else: # Not an ndarray
                     print_warning(f"Context key '{ctx_key}' is not an ndarray in split {split_name}. Skipping.")

            if filtered_context_dict: # Only add 'context' if it's not empty
                patient_data[key] = filtered_context_dict
            # else:
                # print_debug(f"No context data retained for patient {patient_id} in split {split_name} after filtering.")

        elif isinstance(value, np.ndarray) and value.shape[0] == len(subject_ids_array):
            # Filtrar otros arrays numpy
            filtered_value = value[patient_mask]
            if filtered_value.size > 0:
                patient_data[key] = filtered_value
            else:
                # print_debug(f"Key '{key}' for patient {patient_id} in split {split_name} resulted in empty array after filtering.")
                # patient_data[key] = np.array([]) # Store empty or skip
                pass # Skip if empty
        elif isinstance(value, np.ndarray): # Mismatch in length for a primary data field
            print_warning(f"Key '{key}' array length {value.shape[0]} does not match subject_ids length {len(subject_ids_array)} in split {split_name}. Cannot filter reliably for patient {patient_id}.")
            # This is problematic, might indicate data inconsistency.
            # Depending on strictness, could return None here or try to proceed.
            # For now, we skip this problematic key for this patient.
        else: # Non-ndarray, non-context dict values are copied as is (e.g., metadata strings)
            patient_data[key] = value 
            # print_debug(f"Key '{key}' is not an ndarray or context dict, copied as is for patient {patient_id} in split {split_name}.")

    if not patient_data or 'x_cgm' not in patient_data or patient_data['x_cgm'].size == 0 : # Check if essential data is missing
        print_warning(f"Datos insuficientes para paciente {patient_id} en split '{split_name}' después del filtrado.")
        return None
        
    return patient_data

def _validate_split_data(global_data_split: Optional[Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]], 
                        split_name: str, 
                        patient_id: Any) -> bool:
    """Valida que los datos del split sean válidos."""
    if global_data_split is None:
        return False
        
    if 'subject_id' not in global_data_split:
        print_warning(f"No hay 'subject_id' en el split '{split_name}' de datos globales. No se pueden filtrar datos para el paciente {patient_id}.")
        return False
    
    subject_ids_array = global_data_split['subject_id']
    if not isinstance(subject_ids_array, np.ndarray):
        print_warning(f"'subject_id' en el split '{split_name}' no es un np.ndarray. No se pueden filtrar datos para el paciente {patient_id}.")
        return False
    
    return True


def _get_patient_mask(global_data_split: Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]], 
                     patient_id: Any, 
                     _split_name: str) -> Optional[np.ndarray]:
    """Obtiene la máscara para filtrar datos del paciente específico."""
    subject_ids_array = global_data_split['subject_id']
    patient_mask = (subject_ids_array == patient_id)
    
    if not np.any(patient_mask):
        return None
    
    return patient_mask


def _filter_patient_data(global_data_split: Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]], 
                        patient_mask: np.ndarray, 
                        split_name: str) -> Optional[Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]]:
    """Filtra los datos usando la máscara del paciente."""
    patient_data: Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]] = {}
    valid_data_found = False
    # subject_ids_array se pasa a _filter_single_data_entry, no es necesario aquí directamente si no se usa para otra cosa.
    # No obstante, mantenerlo si otras partes de la función lo necesitaran directamente.
    subject_ids_array = global_data_split['subject_id'] 
    
    for key, value in global_data_split.items():
        filtered_result = _filter_single_data_entry(key, value, patient_mask, subject_ids_array, split_name)
        
        add_to_patient_data = False
        if filtered_result is not None:
            if isinstance(filtered_result, np.ndarray):
                if filtered_result.size > 0:
                    add_to_patient_data = True
            elif isinstance(filtered_result, dict):
                # Si filtered_result es un diccionario y no es None,
                # _filter_context_data asegura que no está vacío.
                # Un diccionario vacío se habría convertido en None.
                add_to_patient_data = True
        
        if add_to_patient_data:
            patient_data[key] = filtered_result
            valid_data_found = True
    
    return patient_data if valid_data_found else None

def _filter_single_data_entry(key: str, 
                             value: Union[np.ndarray, Dict[str, np.ndarray]], 
                             patient_mask: np.ndarray,
                             subject_ids_array: np.ndarray,
                             split_name: str) -> Optional[Union[np.ndarray, Dict[str, np.ndarray]]]:
    """Filtra una entrada individual de datos."""
    if key == 'context':
        return _filter_context_data(value, patient_mask, subject_ids_array, split_name)
    elif key == 'subject_id':
        return _filter_subject_id_data(subject_ids_array, patient_mask)
    elif isinstance(value, np.ndarray) and value.shape[0] == len(subject_ids_array):
        return _filter_array_data(value, patient_mask)
    return None


def _filter_subject_id_data(subject_ids_array: np.ndarray, patient_mask: np.ndarray) -> Optional[np.ndarray]:
    """Filtra los datos de subject_id."""
    filtered_data = subject_ids_array[patient_mask]
    return filtered_data if filtered_data.size > 0 else None


def _filter_array_data(value: np.ndarray, patient_mask: np.ndarray) -> Optional[np.ndarray]:
    """Filtra un array de datos usando la máscara del paciente."""
    filtered_value = value[patient_mask]
    return filtered_value if filtered_value.size > 0 else None


def _filter_context_data(context_value: Union[np.ndarray, Dict[str, np.ndarray]], 
                        patient_mask: np.ndarray, 
                        subject_ids_array: np.ndarray, 
                        split_name: str) -> Optional[Dict[str, np.ndarray]]:
    """Filtra los datos de contexto para el paciente específico."""
    if not isinstance(context_value, dict):
        print_warning(f"El valor de 'context' en el split '{split_name}' no es un diccionario como se esperaba.")
        return None
    
    filtered_context_data: Dict[str, np.ndarray] = {}
    for ctx_key, ctx_val_array in context_value.items():
        if isinstance(ctx_val_array, np.ndarray) and ctx_val_array.shape[0] == len(subject_ids_array):
            filtered_ctx_val = ctx_val_array[patient_mask]
            if filtered_ctx_val.size > 0:
                filtered_context_data[ctx_key] = filtered_ctx_val
    
    return filtered_context_data if filtered_context_data else None


def _validate_essential_data(patient_data: Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]], 
                           patient_id: Any, 
                           split_name: str) -> bool:
    """Valida que los datos esenciales estén presentes y no vacíos."""
    essential_keys = ['x_cgm', 'y']
    
    for key_check in essential_keys:
        if key_check in patient_data:
            if isinstance(patient_data[key_check], np.ndarray) and len(patient_data[key_check]) == 0:
                return False
        elif split_name == 'train':
            print_warning(f"Clave esencial '{key_check}' faltante para paciente {patient_id} en split de entrenamiento '{split_name}'.")
            return False
    
    return True

def _validate_training_data(data: AllData, train_per_patient: bool) -> bool:
    """Valida que los datos de entrenamiento sean válidos."""
    if train_per_patient:
        train_split = data.get('train')
        if not train_split or train_split.get('subject_id') is None:
            print_error("Para entrenar por paciente, 'data['train']['subject_id']' debe estar presente y 'data['train']' no debe ser None.")
            return False
        if train_split.get('x_cgm') is None or train_split.get('y') is None:
            print_error("Para entrenar por paciente, 'data['train']['x_cgm']' y 'data['train']['y']' no deben ser None.")
            return False
    else: # General model
        train_split = data.get('train')
        if not train_split or train_split.get('x_cgm') is None or train_split.get('y') is None:
            print_error("Para modelo general, 'data['train']['x_cgm']' y 'data['train']['y']' no deben ser None.")
            return False
            
    # Validar la presencia de 'context' si se usan modelos DRL que lo requieran (opcional, DRL wrappers should handle missing context if possible)
    return True

def _initialize_training_results() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Inicializa los diccionarios de resultados de entrenamiento."""
    return {}, {}, {}, {}

def _train_single_patient_model(
    model_name_key: str,
    patient_id: Any,
    data: AllData, # This is the global data
    model_creators: Dict[str, Callable[..., Union[DLModelWrapperPyTorch, RLModelWrapperPyTorch, DRLModelWrapperPyTorch]]],
    models_dir: str,
    training_config: Dict[str, Any]
) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray], Optional[Dict[str, float]], Optional[Union[DLModelWrapperPyTorch, RLModelWrapperPyTorch, DRLModelWrapperPyTorch]]]:
    """Entrena un modelo para un paciente específico."""
    print_info(f"Intentando entrenar modelo {model_name_key} para paciente {patient_id}")

    patient_train_data = _get_patient_specific_data(data.get('train'), patient_id, 'train')
    patient_val_data = _get_patient_specific_data(data.get('val'), patient_id, 'val')
    patient_test_data = _get_patient_specific_data(data.get('test'), patient_id, 'test')

    if patient_train_data is None or patient_train_data.get('x_cgm') is None or patient_train_data.get('y') is None:
        print_warning(f"No hay datos de entrenamiento CGM o Y para el paciente {patient_id} y modelo {model_name_key}. Saltando entrenamiento para este paciente.")
        return None, None, None, None
    
    data_patient: AllData = {'train': patient_train_data}
    
    if patient_val_data is not None and patient_val_data.get('x_cgm') is not None and patient_val_data.get('y') is not None:
        data_patient['val'] = patient_val_data
    else:
        print_warning(f"No hay datos de validación CGM o Y para paciente {patient_id} para {model_name_key}. La validación podría ser omitida o limitada.")
        data_patient['val'] = None # Explicitly set to None if not valid

    if patient_test_data is not None and patient_test_data.get('x_cgm') is not None and patient_test_data.get('y') is not None:
         data_patient['test'] = patient_test_data
    else:
        data_patient['test'] = None # Explicitly set to None if not valid
    
    model_creator_func = model_creators.get(model_name_key)
    if model_creator_func is None:
        print_warning(f"No se encontró creador para el modelo {model_name_key}. Saltando paciente {patient_id}.")
        return None, None, None, None
    
    patient_model_name_instance = f"{model_name_key}_patient_{patient_id}"
    # Create a subdirectory for each patient model under the main model_name_key directory
    current_model_patient_save_dir = os.path.join(models_dir, model_name_key, f"patient_{patient_id}")
    os.makedirs(current_model_patient_save_dir, exist_ok=True)

    try:
        return train_model_sequential(
            model_creator=model_creator_func,
            model_name=patient_model_name_instance,
            data=data_patient, 
            models_dir=current_model_patient_save_dir, # Save patient model in its own subdir
            training_config=training_config
        )
    except Exception as e:
        print_error(f"Error entrenando {patient_model_name_instance}: {e}")
        return None, None, None, None
    
def _initialize_model_results(model_name_key: str, 
                            results_dicts: Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]) -> None:
    """Inicializa los diccionarios de resultados para un modelo específico."""
    histories_all, predictions_all, metrics_all, trained_models_all = results_dicts
    histories_all[model_name_key] = {}
    predictions_all[model_name_key] = {}
    metrics_all[model_name_key] = {}
    trained_models_all[model_name_key] = {}

def _store_patient_results(model_name_key: str, 
                          patient_id: Any,
                          training_results: Tuple[Optional[Dict[str, Any]], Optional[np.ndarray], Optional[Dict[str, float]], Optional[Union[DLModelWrapperPyTorch, RLModelWrapperPyTorch, DRLModelWrapperPyTorch]]],
                          results_dicts: Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]) -> None:
    """Almacena los resultados de entrenamiento de un paciente específico."""
    history, y_pred_patient, metrics_patient, trained_model_instance = training_results
    histories_all, predictions_all, metrics_all, trained_models_all = results_dicts
    
    if history is not None:
        histories_all[model_name_key][patient_id] = history
        if y_pred_patient is not None:
            predictions_all[model_name_key][patient_id] = y_pred_patient
        if metrics_patient is not None:
            metrics_all[model_name_key][patient_id] = metrics_patient
        if trained_model_instance is not None:
            trained_models_all[model_name_key][patient_id] = trained_model_instance

def _train_patient_models(
    data: AllData,
    models_to_use: Dict[str, bool],
    model_creators: Dict[str, Callable[..., Union[DLModelWrapperPyTorch, RLModelWrapperPyTorch, DRLModelWrapperPyTorch]]],
    models_dir: str,
    training_config: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Entrena modelos específicos por paciente."""
    results_dicts = _initialize_training_results()
    
    unique_patient_ids = np.unique(data['train']['subject_id'])
    print_info(f"Entrenamiento por paciente activado. Se intentarán entrenar modelos para {len(unique_patient_ids)} pacientes.")

    for model_name_key, use_model_flag in models_to_use.items():
        if not use_model_flag:
            continue
            
        print_header(f"Procesando tipo de modelo: {model_name_key} (por paciente)")
        _initialize_model_results(model_name_key, results_dicts)

        for patient_id in tqdm(unique_patient_ids, desc=f"Pacientes para {model_name_key}", unit="paciente"):
            training_results = _train_single_patient_model(
                model_name_key, patient_id, data, model_creators, models_dir, training_config
            )
            
            _store_patient_results(model_name_key, patient_id, training_results, results_dicts)

    return results_dicts

def _train_global_models(
    data: AllData,
    models_to_use: Dict[str, bool],
    model_creators: Dict[str, Callable[..., Union[DLModelWrapperPyTorch, RLModelWrapperPyTorch, DRLModelWrapperPyTorch]]],
    models_dir: str,
    training_config: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Entrena modelos globales."""
    histories_all, predictions_all, metrics_all, trained_models_all = _initialize_training_results()
    
    print_info("Entrenamiento global activado.")
    
    for model_name_key, use_model_flag in models_to_use.items():
        if not use_model_flag:
            continue
            
        print_header(f"Procesando tipo de modelo: {model_name_key} (global)")
        model_creator_func = model_creators.get(model_name_key)
        if model_creator_func is None:
            print_warning(f"No se encontró creador para el modelo {model_name_key}. Saltando.")
            continue
        
        current_model_global_save_dir = os.path.join(models_dir, model_name_key, "global")
        os.makedirs(current_model_global_save_dir, exist_ok=True)

        try:
            history, y_pred, metrics, trained_model_instance = train_model_sequential(
                model_creator=model_creator_func,
                model_name=model_name_key, 
                data=data, 
                models_dir=current_model_global_save_dir,
                training_config=training_config
            )
            histories_all[model_name_key] = history
            if y_pred is not None:
                predictions_all[model_name_key] = y_pred
            metrics_all[model_name_key] = metrics
            trained_models_all[model_name_key] = trained_model_instance
        except Exception as e:
            print_error(f"Error entrenando {model_name_key} (global): {e}")

    return histories_all, predictions_all, metrics_all, trained_models_all

def train_multiple_models(
    data_dfs: AllDataFrames,
    models_to_use: Dict[str, bool],
    model_creators: Dict[str, Callable[..., Union[DLModelWrapperPyTorch, RLModelWrapperPyTorch, DRLModelWrapperPyTorch]]],
    models_dir: str,
    results_dir: str,
    figures_dir: str,
    training_config: Dict[str, Any],
    train_per_patient: bool = False,
    feature_config: Optional[Dict[str, List[str]]] = None
) -> Tuple[Dict[str, Any], Dict[str, pl.DataFrame], Dict[str, Any], Dict[str, Any]]:
    """
    Entrena múltiples modelos utilizando DataFrames y configuración de características.
    """
    histories: Dict[str, Any] = {}
    predictions_dfs_map: Dict[str, Optional[np.ndarray]] = {} # Store numpy arrays first
    all_clinical_metrics: Dict[str, Any] = {}
    trained_models_dict: Dict[str, Any] = {}

    main_train_df = data_dfs.get('train')
    main_val_df = data_dfs.get('val')
    main_test_df = data_dfs.get('test')

    if train_per_patient:
        if main_train_df is None or main_train_df.is_empty() or SUBJECT_ID_COL not in main_train_df.columns:
            print_critical(f"'{SUBJECT_ID_COL}' es requerido en train_df para train_per_patient=True o el DataFrame está vacío.")
            return histories, {}, all_clinical_metrics, trained_models_dict
        
        patient_ids = main_train_df[SUBJECT_ID_COL].unique().sort().to_list()
        print_info(f"Entrenamiento por paciente activado para {len(patient_ids)} pacientes.")

        for model_name, use_model in models_to_use.items():
            if use_model:
                model_type = get_model_type(model_name)
                print_header(f"Entrenando modelo por paciente: {model_name} (Tipo: {model_type})")
                
                patient_histories = {}
                patient_predictions = {}
                patient_clinical_metrics = {}
                patient_trained_models = {}

                for patient_id in tqdm(patient_ids, desc=f"Pacientes ({model_name})"):
                    patient_train_df = main_train_df.filter(pl.col(SUBJECT_ID_COL) == patient_id)
                    patient_val_df = main_val_df.filter(pl.col(SUBJECT_ID_COL) == patient_id) if main_val_df is not None else None
                    patient_test_df = main_test_df.filter(pl.col(SUBJECT_ID_COL) == patient_id) if main_test_df is not None else None

                    if patient_train_df.is_empty():
                        print_warning(f"No hay datos de entrenamiento para el paciente {patient_id} en el modelo {model_name}. Saltando.")
                        continue

                    patient_data_subset: AllDataFrames = {'train': patient_train_df, 'val': patient_val_df, 'test': patient_test_df}
                    
                    current_model_name_patient = f"{model_name}_patient_{patient_id}"
                    
                    # Ensure the creator can accept feature_config
                    # The DRLModelWrapperPyTorch and other wrappers should handle feature_config internally for dimensions.
                    history, predictions_np, clinical_metrics, trained_model_instance = train_model_sequential(
                        model_creator_func=model_creators[model_name],
                        model_name=current_model_name_patient,
                        data_dfs=patient_data_subset,
                        models_dir=models_dir,
                        training_config=training_config,
                        feature_config=feature_config
                    )
                    patient_histories[patient_id] = history
                    if predictions_np is not None:
                        patient_predictions[patient_id] = predictions_np
                    patient_clinical_metrics[patient_id] = clinical_metrics
                    patient_trained_models[patient_id] = trained_model_instance
                
                histories[model_name] = patient_histories
                predictions_dfs_map[model_name] = patient_predictions # This structure needs careful handling for pd.DataFrame conversion
                all_clinical_metrics[model_name] = patient_clinical_metrics
                trained_models_dict[model_name] = patient_trained_models
            else:
                print_info(f"Modelo {model_name} desactivado. Saltando entrenamiento.")
    else: # Global model training
        for model_name, use_model in models_to_use.items():
            if use_model:
                model_type = get_model_type(model_name)
                print_header(f"Entrenando modelo global: {model_name} (Tipo: {model_type})")

                history, predictions_np, clinical_metrics, trained_model_instance = train_model_sequential(
                    model_creator_func=model_creators[model_name],
                    model_name=model_name,
                    data_dfs=data_dfs,
                    models_dir=models_dir,
                    training_config=training_config,
                    feature_config=feature_config
                )
                histories[model_name] = history
                if predictions_np is not None:
                    predictions_dfs_map[model_name] = predictions_np
                all_clinical_metrics[model_name] = clinical_metrics
                trained_models_dict[model_name] = trained_model_instance
            else:
                print_info(f"Modelo {model_name} desactivado. Saltando entrenamiento.")

    # Convert predictions to DataFrames
    predictions_dfs_pd: Dict[str, pl.DataFrame] = {}
    if main_test_df is not None and not main_test_df.is_empty():
        # Assuming predictions align with main_test_df for global models
        # For per-patient, this needs more complex aggregation or separate DFs
        if not train_per_patient:
            test_subject_ids = main_test_df.select(pl.col(SUBJECT_ID_COL).cast(str)).to_series().to_list() if SUBJECT_ID_COL in main_test_df.columns else [str(i) for i in range(len(main_test_df))]
            test_timestamps = main_test_df.select(pl.col(TIMESTAMP_COL).cast(str)).to_series().to_list() if TIMESTAMP_COL in main_test_df.columns else [str(i) for i in range(len(main_test_df))]
            
            base_pred_df = pl.DataFrame({
                SUBJECT_ID_COL: test_subject_ids[:len(next(iter(predictions_dfs_map.values()), np.array([])))], # Ensure length match
                TIMESTAMP_COL: test_timestamps[:len(next(iter(predictions_dfs_map.values()), np.array([])))]
            })

            for model_name, preds_np in predictions_dfs_map.items():
                if preds_np is not None:
                    # Ensure preds_np matches the length of base_pred_df
                    preds_to_add = preds_np[:len(base_pred_df)]
                    temp_df = base_pred_df.copy()
                    temp_df[model_name] = preds_to_add
                    predictions_dfs_pd[model_name] = temp_df
        else:
            # For per-patient, predictions_dfs_map is Dict[str, Dict[patient_id, np.ndarray]]
            # This requires a different way to form DataFrames, perhaps one per model,
            # concatenating patient predictions.
            print_warning("La conversión de predicciones por paciente a DataFrame consolidado no está completamente implementada.")
            # Placeholder:
            for model_name, patient_preds_dict in predictions_dfs_map.items():
                if isinstance(patient_preds_dict, dict): # Check if it's per-patient
                    all_patient_preds_list = []
                    all_patient_subjects_list = []
                    all_patient_timestamps_list = [] # TODO: Get corresponding timestamps
                    
                    for patient_id, preds_np in patient_preds_dict.items():
                        if preds_np is not None:
                            patient_test_data_filter = main_test_df.filter(pl.col(SUBJECT_ID_COL) == patient_id)
                            if not patient_test_data_filter.is_empty():
                                timestamps_patient = patient_test_data_filter.select(pl.col(TIMESTAMP_COL).cast(str)).to_series().to_list()
                                
                                all_patient_preds_list.extend(preds_np[:len(timestamps_patient)])
                                all_patient_subjects_list.extend([patient_id] * len(preds_np[:len(timestamps_patient)]))
                                all_patient_timestamps_list.extend(timestamps_patient[:len(preds_np)])
                    
                    if all_patient_preds_list:
                        predictions_dfs_pd[model_name] = pl.DataFrame({
                            SUBJECT_ID_COL: all_patient_subjects_list,
                            TIMESTAMP_COL: all_patient_timestamps_list, # This needs correct alignment
                            model_name: all_patient_preds_list
                        })
                elif isinstance(patient_preds_dict, np.ndarray): # Global model case already handled
                     if patient_preds_dict is not None and len(base_pred_df) > 0:
                        preds_to_add = patient_preds_dict[:len(base_pred_df)]
                        temp_df = base_pred_df.copy()
                        temp_df[model_name] = preds_to_add
                        predictions_dfs_pd[model_name] = temp_df


    return histories, predictions_dfs_pd, all_clinical_metrics, trained_models_dict

def debug_tensor_info(tensor: torch.Tensor, name: str) -> None:
    """
    Imprime información de depuración sobre un tensor.
    
    Parámetros:
    -----------
    tensor : torch.Tensor
        Tensor a debugguear
    name : str
        Nombre del tensor para identificación
    """
    print_debug(f"{name} - Forma: {tensor.shape}, Tipo: {tensor.dtype}, Dispositivo: {tensor.device}")
    print_debug(f"{name} - Min: {tensor.min().item():.4f}, Max: {tensor.max().item():.4f}, Media: {tensor.mean().item():.4f}")
    if torch.isnan(tensor).any():
        print_warning(f"{name} contiene valores NaN")
    if torch.isinf(tensor).any():
        print_warning(f"{name} contiene valores Inf")

def _process_batch_data(simulator: GlucoseSimulator, 
                       model_wrapper: Union[DLModelWrapperPyTorch, RLModelWrapperPyTorch, DRLModelWrapperPyTorch], 
                       batch_x_cgm: np.ndarray,  # Forma: (processing_batch_size, timesteps, cgm_features)
                       batch_x_other: np.ndarray, # Forma: (processing_batch_size, other_features_len)
                       batch_context: Dict[str, np.ndarray], # Cada valor: (processing_batch_size,)
                       replay_buffer: ReplayBuffer,
                       device: torch.device) -> list:
    """
    Procesa un batch de datos, interactúa con el simulador y almacena experiencias.
    
    Parámetros:
    -----------
    simulator : GlucoseSimulator
        Simulador de glucosa.
    model_wrapper : Union[DLModelWrapperPyTorch, RLModelWrapperPyTorch, DRLModelWrapperPyTorch]
        Wrapper del modelo DRL.
    batch_x_cgm : np.ndarray
        Batch de datos CGM.
    batch_x_other : np.ndarray
        Batch de otras características.
    batch_context : Dict[str, np.ndarray]
        Batch de datos de contexto.
    replay_buffer : ReplayBuffer
        Buffer para almacenar experiencias.
    device : torch.device
        Dispositivo para tensores (aunque DDPG espera numpy para algunas partes).
        
    Retorna:
    --------
    list
        Lista de recompensas obtenidas en el batch.
    """
    batch_rewards_processed = []
    num_samples_in_batch = len(batch_x_cgm)

    # Asumimos que el modelo subyacente (ej. DDPG) tiene _build_state_representation
    # Deberíamos acceder a esto de una manera más genérica si es posible, o asegurar que todos los DRL lo tengan.
    # Por ahora, se accede directamente para DDPG.
    drl_agent = model_wrapper.model 
    if not hasattr(drl_agent, '_build_state_representation'):
        print_warning("El agente DRL subyacente no tiene '_build_state_representation'. El estado para el buffer puede ser incorrecto.")
        # Podría definirse un método de construcción de estado por defecto o lanzar un error.

    for i in range(num_samples_in_batch):
        current_cgm_sample_np = batch_x_cgm[i]     # Forma: (timesteps, cgm_features)
        current_other_sample_np = batch_x_other[i] # Forma: (other_features_len,)

        # Construir diccionario de contexto para esta muestra específica
        sample_context_features = {key: val[i] for key, val in batch_context.items()}
        
        # Crear el contexto completo para la selección de acción y construcción del estado
        # Asegurarse que las claves coincidan con CONTEXT_FEATURE_ORDER y las esperadas por _create_context_for_sample
        # y _build_state_representation del agente DDPG.
        
        # El `_create_context_for_sample` es más para predicción detallada.
        # Para `select_action` del DDPG, necesitamos un `context_dict` simple con las claves de `CONTEXT_FEATURE_ORDER`.
        context_dict_for_action = {
            key: float(sample_context_features.get(key, 0.0)) for key in CONTEXT_FEATURE_ORDER
        }
        # Asegurar que las claves mapeadas como 'carb_intake' (de 'meal_carbs') estén presentes
        # Esto se maneja en _extract_context_for_drl que llena context_train/val/test
        # y DDPG._build_state_representation usa CONTEXT_FEATURE_ORDER
        # Si batch_context ya tiene las claves de CONTEXT_FEATURE_ORDER, esto es más simple:
        # context_dict_for_action = {k: float(sample_context_features[k]) for k in CONTEXT_FEATURE_ORDER if k in sample_context_features}
        # Es crucial que `batch_context` (que viene de `context_train`) ya tenga las claves correctas
        # según `CONTEXT_FEATURE_ORDER` o un mapeo claro.
        # `_extract_context_for_drl` ya crea estas claves.

        # Seleccionar acción usando el método del wrapper (que delega al DDPG.select_action)
        # DDPG.select_action espera state_tuple=(np.ndarray, np.ndarray) y context_dict
        action_np = model_wrapper.select_action_for_rollout(
            current_cgm_sample_np, 
            current_other_sample_np,
            context_dict_for_action, # Pasar el diccionario de contexto simple
            add_noise=True
        )
        
        action_to_take = float(action_np[0]) # Asumiendo action_dim = 1

        # Simular un paso en el entorno
        # El simulador necesita la dosis, glucosa inicial, y carbohidratos
        # La glucosa inicial para el simulador es el último valor de la ventana CGM
        initial_glucose_for_sim = float(current_cgm_sample_np[-1, 0]) 
        # Los carbohidratos para el simulador vienen del contexto
        carbs_for_sim = float(context_dict_for_action.get('carb_intake', 0.0))

        _next_glucose_value, reward_value, done_flag, _ = simulator.step(
            action_to_take,
            current_glucose=initial_glucose_for_sim,
            carb_intake=carbs_for_sim
        )
        batch_rewards_processed.append(reward_value)

        # Construir representación completa del estado actual y siguiente para el buffer
        # Esto DEBE usar el método del agente DRL (o el que sea)
        if hasattr(drl_agent, '_build_state_representation'):
            current_full_state_tensor = drl_agent._build_state_representation(
                current_cgm_sample_np,
                current_other_sample_np,
                context_dict_for_action # Usar el mismo context_dict
            )
            current_full_state_np = current_full_state_tensor.squeeze(0).cpu().numpy()

            # Para el next_state, necesitamos el "siguiente" cgm, other, y context.
            # Esto es una simplificación en entornos offline. A menudo, next_state se basa en
            # la observación real siguiente del dataset, no en una simulación de un solo paso.
            # Aquí, estamos en un bucle sobre datos existentes. "next_state" sería
            # (batch_x_cgm[i+1], batch_x_other[i+1], context_dict_for_action_next) si no es el final.
            # Si el simulador diera un `next_observation` completo, eso se usaría.
            # Por ahora, asumimos que el `next_state` se construye a partir de la siguiente muestra en el batch.
            # Esto es típico en DRL offline donde (s, a, r, s') vienen del dataset.
            # La lógica actual de _run_episode itera sobre el dataset, así que s' es la siguiente entrada.
            
            # Si no es la última muestra del batch Y no es la última muestra del dataset total
            if i + 1 < num_samples_in_batch and i + replay_buffer.current_episode_step < replay_buffer.max_episode_steps -1 : # Heurística
                next_cgm_sample_np = batch_x_cgm[i+1]
                next_other_sample_np = batch_x_other[i+1]
                next_sample_context_features = {key: val[i+1] for key, val in batch_context.items()}
                next_context_dict = {
                    key: float(next_sample_context_features.get(key, 0.0)) for key in CONTEXT_FEATURE_ORDER
                }
                next_full_state_tensor = drl_agent._build_state_representation(
                    next_cgm_sample_np,
                    next_other_sample_np,
                    next_context_dict
                )
                next_full_state_np = next_full_state_tensor.squeeze(0).cpu().numpy()
            else: # Si es el final, el next_state puede ser una copia o un estado terminal especial
                next_full_state_np = current_full_state_np # Simplificación: o un estado terminal
                done_flag = True # Marcar como done si es el final del batch/dataset

            print_critical(f"VERIFYING BEFORE ADD: next_full_state_np shape: {next_full_state_np.shape}, dtype: {next_full_state_np.dtype}")
            if next_full_state_np.shape != (model_wrapper.model.state_dim,):
                print_error(f"CRITICAL SHAPE MISMATCH for next_full_state_np before add! Expected {(model_wrapper.model.state_dim,)}, Got {next_full_state_np.shape}")

            print_debug(f"[_process_batch_data] Adding to buffer: current_state shape: {current_full_state_np.shape}, next_state shape: {next_full_state_np.shape}")
            replay_buffer.add(current_full_state_np, action_np, reward_value, next_full_state_np, done_flag)
        
        replay_buffer.current_episode_step +=1


    return batch_rewards_processed

def _extract_batch_context(
    context_train: Dict[str, np.ndarray],
    start_idx: int,
    end_idx: int,
    num_samples_in_data: int
) -> Dict[str, np.ndarray]:
    """
    Extrae el contexto para el batch actual de datos.

    Parámetros:
    -----------
    context_train : Dict[str, np.ndarray]
        Diccionario completo con los datos de contexto del entrenamiento.
        Puede estar vacío si no hay datos de contexto.
    start_idx : int
        Índice de inicio para el batch actual.
    end_idx : int
        Índice de fin para el batch actual.
    num_samples_in_data : int
        Número total de muestras en los datos de entrenamiento.

    Retorna:
    --------
    Dict[str, np.ndarray]
        Diccionario con el contexto para el batch actual.
    """
    batch_context: Dict[str, np.ndarray] = {}
    if not context_train:
        return batch_context

    for key, full_array in context_train.items():
        if full_array is not None:
            if len(full_array) == num_samples_in_data:
                batch_context[key] = full_array[start_idx:end_idx]
            else:
                # Si el array de contexto no tiene la misma longitud, se advierte y se usa completo.
                # Esto podría indicar un problema en la preparación de datos.
                print_warning(
                    f"La longitud del array de contexto '{key}' ({len(full_array)}) no coincide con "
                    f"los datos de entrenamiento ({num_samples_in_data}). Se usará el array completo "
                    f"para cada batch, lo cual podría ser incorrecto."
                )
                batch_context[key] = full_array
        else:
            batch_context[key] = np.array([]) # Asignar array vacío si el original es None
    return batch_context

def _try_update_drl_model(
    model_wrapper: DRLModelWrapperPyTorch,
    replay_buffer: ReplayBuffer,
    min_buffer_samples: int,
    current_step_or_batch_idx: int,
    update_freq: int,
    training_batch_size: int
) -> Tuple[float, float, bool]:
    """
    Intenta actualizar el modelo DRL si se cumplen las condiciones.

    Parámetros:
    -----------
    model_wrapper : DRLModelWrapperPyTorch
        Wrapper del modelo DRL.
    replay_buffer : ReplayBuffer
        Buffer de repetición.
    min_buffer_samples : int
        Mínimo de muestras en el buffer para iniciar el entrenamiento.
    current_step_or_batch_idx : int
        Paso o índice de lote actual.
    update_freq : int
        Frecuencia con la que se debe intentar la actualización.
    training_batch_size : int
        Tamaño de lote para muestrear del buffer.
            
    Retorna:
    --------
    Tuple[float, float, bool]
        (pérdida_actor, pérdida_crítico, fue_actualizado).
    """
    actor_loss: float = 0.0
    critic_loss: float = 0.0
    model_updated: bool = False

    if len(replay_buffer) >= min_buffer_samples and current_step_or_batch_idx % update_freq == 0:
        # Asegurarse que model_wrapper.model es el agente DDPG/SAC etc.
        drl_agent = model_wrapper.model
        if drl_agent is not None and hasattr(drl_agent, 'run_training_step') and callable(drl_agent.run_training_step):
            try:
                loss_info = drl_agent.run_training_step(replay_buffer, training_batch_size)
                actor_loss = loss_info.get(CONST_ACTOR_LOSS, 0.0)
                critic_loss = loss_info.get(CONST_CRITIC_LOSS, 0.0)
                model_updated = True
            except Exception as e:
                print_error(f"Error durante DRL training step: {e}")
                # Considerar si relanzar o manejar de otra forma
        else:
            model_name = type(drl_agent).__name__ if drl_agent is not None else "None"
            print_warning(f"El modelo DRL {model_name} no tiene 'run_training_step' o no es llamable. No se puede actualizar.")
            
    return actor_loss, critic_loss, model_updated

def _get_next_state_representation(
    x_cgm_batch: np.ndarray, 
    x_other_batch: Optional[np.ndarray], 
    context_batch: Optional[Dict[str, np.ndarray]], 
    current_idx_in_batch: int,
    current_batch_actual_size: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, float]]]:
    """
    Obtiene la representación del siguiente estado del lote de datos.
    Retorna None para los componentes si no hay un siguiente estado (fin del lote).

    Parámetros:
    -----------
    x_cgm_batch : np.ndarray
        Lote de datos CGM.
    x_other_batch : Optional[np.ndarray]
        Lote de otras características.
    context_batch : Optional[Dict[str, np.ndarray]]
        Lote de datos de contexto.
    current_idx_in_batch : int
        Índice actual dentro del lote.
    current_batch_actual_size : int
        Tamaño real del lote actual que se está procesando.
            
    Retorna:
    --------
    Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, float]]]
        (x_cgm_next_sample, x_other_next_sample, context_dict_next_sample) o (None, None, None).
    """
    next_idx_in_batch = current_idx_in_batch + 1
    if next_idx_in_batch < current_batch_actual_size:
        x_cgm_next_sample = x_cgm_batch[next_idx_in_batch]
        x_other_next_sample = x_other_batch[next_idx_in_batch] if x_other_batch is not None and len(x_other_batch) > next_idx_in_batch else np.array([])
        
        context_dict_next_sample: Dict[str, float] = {}
        if context_batch:
            for key, values in context_batch.items():
                if next_idx_in_batch < len(values):
                    context_dict_next_sample[key] = float(values[next_idx_in_batch])
                else: 
                    print_warning(f"Índice {next_idx_in_batch} fuera de rango para la clave de contexto '{key}' (longitud {len(values)}). Usando 0.0.")
                    context_dict_next_sample[key] = 0.0
        return x_cgm_next_sample, x_other_next_sample, context_dict_next_sample
    return None, None, None


def _validate_drl_agent(drl_agent: Any) -> None:
    """Valida que el agente DRL tenga los atributos y métodos necesarios."""
    required_attrs = ['_build_state_representation', 'state_dim', 'action_dim']
    missing_attrs = [attr for attr in required_attrs if not hasattr(drl_agent, attr)]
    
    if drl_agent is None or missing_attrs:
        msg = f"El modelo DRL (agente) no está inicializado correctamente o le faltan atributos/métodos necesarios: {missing_attrs}"
        print_error(msg)
        raise ValueError(msg)

def _extract_sample_data(x_cgm_batch: np.ndarray, x_other_batch: Optional[np.ndarray], 
                        context_batch: Optional[Dict[str, np.ndarray]], 
                        index: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Extrae los datos de una muestra específica del lote."""
    x_cgm_sample = x_cgm_batch[index]
    x_other_sample = (x_other_batch[index] 
                     if x_other_batch is not None and len(x_other_batch) > index 
                     else np.array([]))
    
    context_dict_sample: Dict[str, float] = {}
    if context_batch:
        for key, values_array in context_batch.items():
            context_dict_sample[key] = (float(values_array[index]) 
                                       if index < len(values_array) 
                                       else 0.0)
    
    return x_cgm_sample, x_other_sample, context_dict_sample

def _get_simulation_params(context_dict: Dict[str, float], action_np: np.ndarray) -> Tuple[float, float, float]:
    """Extrae los parámetros necesarios para la simulación."""
    current_glucose = context_dict.get('current_glucose', 150.0)
    carb_intake = context_dict.get('carb_intake', 0.0)
    action_value = (float(action_np[0]) 
                   if action_np.ndim > 0 and action_np.size > 0 
                   else float(action_np))
    return current_glucose, carb_intake, action_value

def _build_next_state(drl_agent: Any, x_cgm_next_sample: Optional[np.ndarray], 
                     x_other_next_sample: Optional[np.ndarray], 
                     context_dict_next_sample: Optional[Dict[str, float]]) -> Tuple[np.ndarray, bool]:
    """Construye el siguiente estado para el buffer de repetición."""
    if x_cgm_next_sample is not None and context_dict_next_sample is not None:
        x_other_next_val = x_other_next_sample if x_other_next_sample is not None else np.array([])
        next_unified_state_tensor = drl_agent._build_state_representation(
            x_cgm_next_sample, x_other_next_val, context_dict_next_sample
        )
        next_unified_state_np = next_unified_state_tensor.squeeze(0).cpu().numpy()
        done = False
    else:
        next_unified_state_np = np.zeros(drl_agent.state_dim, dtype=np.float32)
        done = True
    
    return next_unified_state_np, done

def _process_single_sample(model_wrapper: DRLModelWrapperPyTorch, simulator: GlucoseSimulator,
                          drl_agent: Any, x_cgm_sample: np.ndarray, x_other_sample: np.ndarray,
                          context_dict_sample: Dict[str, float], x_cgm_batch: np.ndarray,
                          x_other_batch: Optional[np.ndarray], context_batch: Optional[Dict[str, np.ndarray]],
                          index: int, batch_size: int, replay_buffer: ReplayBuffer) -> float:
    """Procesa una única muestra del lote."""
    # Construir estado actual
    current_unified_state_tensor = drl_agent._build_state_representation(
        x_cgm_sample, x_other_sample, context_dict_sample
    )
    current_unified_state_np = current_unified_state_tensor.squeeze(0).cpu().numpy()

    # Seleccionar acción
    action_np = model_wrapper.select_action_for_rollout(
        x_cgm_sample, x_other_sample, context_dict_sample, add_noise=True
    )

    # Simular paso
    current_glucose, carb_intake, action_value = _get_simulation_params(context_dict_sample, action_np)
    _next_glucose_level_sim, reward, done, _ = simulator.step(
        action_insulin=action_value,
        current_glucose=current_glucose,
        carb_intake=carb_intake
    )

    # Obtener siguiente estado
    x_cgm_next_sample, x_other_next_sample, context_dict_next_sample = _get_next_state_representation(
        x_cgm_batch, x_other_batch, context_batch, index, batch_size
    )

    next_unified_state_np, done = _build_next_state(
        drl_agent, x_cgm_next_sample, x_other_next_sample, context_dict_next_sample
    )

    # Almacenar experiencia
    action_to_store = action_np.reshape(drl_agent.action_dim)
    replay_buffer.add(current_unified_state_np, action_to_store, reward, next_unified_state_np, done)
    
    return reward

def _process_batch_for_drl(
    model_wrapper: DRLModelWrapperPyTorch,
    simulator: GlucoseSimulator,
    x_cgm_batch: np.ndarray,
    x_other_batch: Optional[np.ndarray],
    context_batch: Optional[Dict[str, np.ndarray]],
    replay_buffer: ReplayBuffer
) -> List[float]:
    """
    Procesa un lote de datos para DRL, interactuando con el simulador y almacenando transiciones.

    Parámetros:
    -----------
    model_wrapper : DRLModelWrapperPyTorch
        Wrapper del modelo DRL.
    simulator : GlucoseSimulator
        Simulador de glucosa.
    x_cgm_batch : np.ndarray
        Lote de datos CGM.
    x_other_batch : Optional[np.ndarray]
        Lote de otras características.
    context_batch : Optional[Dict[str, np.ndarray]]
        Lote de datos de contexto.
    replay_buffer : ReplayBuffer
        Buffer de repetición para almacenar experiencias.
            
    Retorna:
    --------
    List[float]
        Lista de recompensas obtenidas en el lote.
    """
    drl_agent = model_wrapper.model
    _validate_drl_agent(drl_agent)

    batch_rewards_list: List[float] = []
    current_batch_actual_size = len(x_cgm_batch)

    for i in range(current_batch_actual_size):
        x_cgm_sample, x_other_sample, context_dict_sample = _extract_sample_data(
            x_cgm_batch, x_other_batch, context_batch, i
        )
        
        reward = _process_single_sample(
            model_wrapper, simulator, drl_agent, x_cgm_sample, x_other_sample,
            context_dict_sample, x_cgm_batch, x_other_batch, context_batch,
            i, current_batch_actual_size, replay_buffer
        )
        
        batch_rewards_list.append(reward)
        
    return batch_rewards_list

def _run_episode(
    simulator: GlucoseSimulator,
    model_wrapper: DRLModelWrapperPyTorch,
    x_cgm_train: np.ndarray,
    x_other_train: Optional[np.ndarray],
    context_train: Optional[Dict[str, np.ndarray]],
    replay_buffer: ReplayBuffer,
    min_buffer_samples: int,
    training_batch_size: int, 
    update_freq: int,
    processing_batch_size: int
) -> Tuple[float, float, float, int]:
    """
    Ejecuta un episodio de entrenamiento para un modelo DRL.

    Parámetros:
    -----------
    simulator : GlucoseSimulator
        Simulador de glucosa.
    model_wrapper : DRLModelWrapperPyTorch
        Wrapper del modelo DRL.
    x_cgm_train : np.ndarray
        Datos CGM de entrenamiento.
    x_other_train : Optional[np.ndarray]
        Otras características de entrenamiento.
    context_train : Optional[Dict[str, np.ndarray]]
        Datos de contexto para el entrenamiento.
    replay_buffer : ReplayBuffer
        Buffer de repetición.
    min_buffer_samples : int
        Número mínimo de muestras en el buffer antes de empezar a entrenar.
    training_batch_size : int
        Tamaño de lote para muestrear del buffer y actualizar el agente DRL.
    update_freq : int
        Frecuencia de actualización del modelo DRL (ej: cada N pasos/batches procesados).
    processing_batch_size : int
        Tamaño de lote para iterar sobre los datos de entrenamiento.
            
    Retorna:
    --------
    Tuple[float, float, float, int]
        (recompensa_total_episodio, perdida_actor_promedio, perdida_critico_promedio, numero_actualizaciones_modelo)
    """
    episode_total_rewards: float = 0.0
    cumulative_actor_loss: float = 0.0
    cumulative_critic_loss: float = 0.0
    num_model_updates: int = 0
    
    num_processing_batches = (len(x_cgm_train) + processing_batch_size - 1) // processing_batch_size

    for batch_idx in range(num_processing_batches):
        start = batch_idx * processing_batch_size
        end = min((batch_idx + 1) * processing_batch_size, len(x_cgm_train))
        
        x_cgm_batch = x_cgm_train[start:end]
        x_other_batch_slice = x_other_train[start:end] if x_other_train is not None else None
        
        context_batch_slice: Optional[Dict[str, np.ndarray]] = None
        if context_train:
            context_batch_slice = {}
            for key, values in context_train.items():
                context_batch_slice[key] = values[start:end]
        
        batch_rewards_list = _process_batch_for_drl(
            model_wrapper, simulator, x_cgm_batch, x_other_batch_slice,
            context_batch_slice, replay_buffer
        )
        episode_total_rewards += sum(batch_rewards_list)
        
        # Se asume que current_step_or_batch_idx para _try_update_drl_model es el índice del lote de procesamiento
        actor_loss_increment, critic_loss_increment, model_was_updated = _try_update_drl_model(
            model_wrapper, replay_buffer, min_buffer_samples,
            batch_idx, update_freq, training_batch_size
        )
        
        if model_was_updated:
            cumulative_actor_loss += actor_loss_increment
            cumulative_critic_loss += critic_loss_increment
            num_model_updates += 1
            
    avg_actor_loss = cumulative_actor_loss / num_model_updates if num_model_updates > 0 else 0.0
    avg_critic_loss = cumulative_critic_loss / num_model_updates if num_model_updates > 0 else 0.0
    
    return episode_total_rewards, avg_actor_loss, avg_critic_loss, num_model_updates

def _create_context_for_sample(current_cgm_value: float, 
                               sample_context_features: Dict[str, Union[float, np.ndarray]], 
                               cgm_history_for_iob: np.ndarray) -> Dict[str, Any]:
    """
    Crea el diccionario de contexto para una muestra específica durante la predicción.
    
    Parámetros:
    -----------
    current_cgm_value : float
        Valor actual de glucosa CGM.
    sample_context_features : Dict[str, Union[float, np.ndarray]]
        Diccionario con las características contextuales para esta muestra 
        (ej: {'carb_intake': valor, 'sleep_quality': valor, ...}).
    cgm_history_for_iob : np.ndarray
        Historial CGM necesario para calcular IOB (usualmente el array [1, timesteps, 1] de la muestra actual).
        
    Retorna:
    --------
    dict
        Contexto completo para la predicción.
    """
    context = {}
    context['current_glucose'] = float(current_cgm_value)
    
    # Extraer características del diccionario sample_context_features
    context['carb_intake'] = float(sample_context_features.get('carb_intake', 0.0))
    context['sleep_quality'] = float(sample_context_features.get('sleep_quality', 0.0)) # 0 si no está disponible
    context['work_intensity'] = float(sample_context_features.get('work_intensity', 0.0)) # 0 si no está disponible
    context['exercise_intensity'] = float(sample_context_features.get('exercise_intensity', 0.0)) # 0 si no está disponible
    
    # IOB puede venir precalculado en sample_context_features o calcularse aquí
    if 'iob' in sample_context_features:
        context['iob'] = float(sample_context_features['iob'])
    else:
        # Asegurarse que cgm_history_for_iob tiene la forma correcta para calculate_iob
        # calculate_iob espera (muestras, timesteps, características) o similar.
        # Si cgm_history_for_iob es (1, 24, 1) para la muestra actual:
        context['iob'] = calculate_iob(cgm_history_for_iob, context['carb_intake']) 
        # Asegurar que calculate_iob puede manejar un solo carb_intake
        # o pasar [context['carb_intake']] si espera una lista.

    # Añadir otras características contextuales necesarias por el modelo
    # context['stress_level'] = float(sample_context_features.get('stress_level', 0.0))
    # context['target_glucose'] = float(sample_context_features.get('target_glucose', 110.0)) # Ejemplo de target

    return context


def _predict_test_samples(model_wrapper: Union[DLModelWrapperPyTorch, RLModelWrapperPyTorch, DRLModelWrapperPyTorch], 
                         x_cgm_test: np.ndarray, 
                         x_other_test: np.ndarray,
                         context_test: Dict[str, np.ndarray]
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Realiza predicciones sobre el conjunto de prueba utilizando el modelo DRL entrenado.
    Utiliza el contexto completo para cada predicción.

    Parámetros:
    -----------
    model_wrapper : Union[DLModelWrapperPyTorch, RLModelWrapperPyTorch, DRLModelWrapperPyTorch]
        Wrapper del modelo DRL entrenado.
    x_cgm_test : np.ndarray
        Datos CGM del conjunto de prueba.
    x_other_test : np.ndarray
        Otras características del conjunto de prueba.
    context_test : Dict[str, np.ndarray]
        Contexto completo para el conjunto de prueba.

    Retorna:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (Predicciones y_pred, glucosa inicial, ingesta de carbohidratos para evaluación)
    """
    num_samples = len(x_cgm_test)
    y_pred = np.zeros(num_samples)
    
    initial_glucose_test = np.zeros(num_samples)
    carb_intake_test = np.zeros(num_samples)
    
    pred_bar = tqdm(range(num_samples), desc="Predicciones finales DRL", unit="pred", leave=True) # leave=True para DRL

    for i in pred_bar:
        # Extraer la muestra actual como NumPy array (sin añadir una dimensión de batch)
        # model_wrapper.predict_with_context espera una única muestra (timesteps, features)
        current_cgm_sample_np = x_cgm_test[i] 
        current_other_sample_np = x_other_test[i]
        
        # Construir el diccionario de contexto para esta muestra específica
        sample_context_features = {key: val[i] for key, val in context_test.items()}
        
        # Guardar glucosa inicial y carbohidratos para este punto de prueba
        # La glucosa inicial es el último valor de la ventana CGM actual
        if current_cgm_sample_np.ndim == 2 and current_cgm_sample_np.shape[0] > 0 and current_cgm_sample_np.shape[1] > 0:
            initial_glucose_test[i] = current_cgm_sample_np[-1, 0]
        elif current_cgm_sample_np.ndim == 1 and current_cgm_sample_np.shape[0] > 0: # Si CGM es 1D (solo una característica)
            initial_glucose_test[i] = current_cgm_sample_np[-1]
        else: # Fallback si la forma no es la esperada
            initial_glucose_test[i] = sample_context_features.get('current_glucose', 0.0)
            if initial_glucose_test[i] < CONST_EPSILON: # Uso de CONST_EPSILON para evitar problemas de precisión de punto flotante.
                 print_warning(f"No se pudo determinar la glucosa inicial para la muestra {i} desde CGM. Usando valor de contexto o 0.0.")


        carb_intake_test[i] = float(sample_context_features.get('carb_intake', 0.0))

        # Crear el contexto completo para la predicción
        full_context_for_prediction = _create_context_for_sample(
            current_cgm_value=initial_glucose_test[i], # Usar la glucosa actual ya extraída
            sample_context_features=sample_context_features,
            cgm_history_for_iob=current_cgm_sample_np # Pasar la ventana cgm actual para IOB
        )
        
        # Realizar la predicción con el contexto llamando al método del WRAPPER
        # El wrapper se encarga de la conversión a tensores y de llamar al modelo subyacente.
        with torch.no_grad():
            prediction_value = model_wrapper.predict_with_context(
                x_cgm=current_cgm_sample_np,         # Pasar NumPy array
                x_other=current_other_sample_np,     # Pasar NumPy array
                **full_context_for_prediction        # Kwargs como current_glucose, carb_intake, etc.
            )
        # model_wrapper.predict_with_context devuelve un float directamente
        y_pred[i] = prediction_value 
        
        if i % 10 == 0:
            pred_bar.set_description(f"Predicción {i+1}/{num_samples}: Dosis={y_pred[i]:.2f}")
    
    pred_bar.close()
    
    return y_pred, initial_glucose_test, carb_intake_test

def train_and_evaluate_model_drl(model_wrapper: Union[DLModelWrapperPyTorch, RLModelWrapperPyTorch, DRLModelWrapperPyTorch],
                        model_name: str,
                        data: Dict[str, Dict[str, np.ndarray]],
                        models_dir: str = CONST_MODELS,
                        training_config: Dict[str, Any] = TRAINING_CONFIG) -> Tuple[Dict[str, List[float]], np.ndarray, Dict[str, float], Union[DLModelWrapperPyTorch, RLModelWrapperPyTorch, DRLModelWrapperPyTorch]]:
    """
    Entrena y evalúa un modelo de aprendizaje por refuerzo para la predicción de glucosa.
    
    Parámetros:
    -----------
    model_wrapper : Union[DLModelWrapperPyTorch, RLModelWrapperPyTorch, DRLModelWrapperPyTorch]
        Modelo a entrenar y evaluar
    model_name : str
        Nombre del modelo para guardado y registro
    data : Dict[str, Dict[str, np.ndarray]]
        Diccionario con datos de entrenamiento, validación y prueba
    models_dir : str, opcional
        Directorio para guardar modelos (default: "models") 
    training_config : Dict[str, Any], opcional
        Configuración de entrenamiento, incluyendo número de episodios, tamaño de batch, etc.
    
    Retorna:
    --------
    Tuple[Dict[str, List[float]], np.ndarray, Dict[str, float], Union[DLModelWrapperPyTorch, RLModelWrapperPyTorch, DRLModelWrapperPyTorch]]
        (historial de entrenamiento, predicciones, métricas clínicas, modelo entrenado)
    """
    # Inicializar simulador
    simulator = GlucoseSimulator()
    
    episodes = training_config.get('episodes', CONST_EPOCHS * 5) # Ajustado para DRL
    batch_size = training_config.get('batch_size', CONST_DEFAULT_BATCH_SIZE * 2)
    update_freq = training_config.get('update_freq', 10) 
    processing_batch_size = training_config.get('processing_batch_size', 128)
    
    _device = model_wrapper.model.device # Asumiendo que el modelo wrapper tiene el modelo real en .model
    
    # Obtener dimensiones del estado y acción del modelo DRL subyacente
    if not hasattr(model_wrapper.model, 'state_dim') or not hasattr(model_wrapper.model, 'action_dim'):
        print_error("El modelo DRL subyacente debe tener atributos 'state_dim' y 'action_dim'.")
        raise AttributeError("Atributos 'state_dim' o 'action_dim' faltantes en el modelo DRL.")
    
    state_dim = model_wrapper.model.state_dim
    action_dim = model_wrapper.model.action_dim

    # Obtener configuración del buffer del modelo o de la configuración global
    # DDPG_CONFIG y otros deben tener 'buffer_size' y 'seed'
    model_specific_config = model_wrapper.model.config if hasattr(model_wrapper.model, 'config') else {}
    buffer_size = model_specific_config.get('buffer_size', BUFFER_CONFIG.get('buffer_size'))
    seed = model_specific_config.get('seed', CONST_DEFAULT_SEED)
    print_debug(f"[train_and_evaluate_model_drl] Initializing ReplayBuffer with state_dim: {state_dim}, action_dim: {action_dim}")

    replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        max_size=buffer_size,
        # device=_device, # El ReplayBuffer ahora maneja NumPy, el agente se encarga del dispositivo
        seed=seed
    )
    # Asegurar que min_buffer_samples sea un porcentaje del tamaño real del buffer o un mínimo absoluto
    min_buffer_fill_ratio = training_config.get('min_buffer_fill_ratio', 0.1) # e.g., 10%
    min_absolute_samples = training_config.get('min_absolute_samples_for_training', 1000)
    min_buffer_samples = max(int(buffer_size * min_buffer_fill_ratio), min_absolute_samples, batch_size)


    episode_metrics_history = {'actor_loss': [], 'critic_loss': [], 'rewards': [], 'buffer_size': []} # Para guardar historial
    
    episode_bar = tqdm(range(episodes), desc="Episodios DRL", position=0, leave=True)
    
    # Extraer datos de entrenamiento y contexto
    x_cgm_train = data['train']['x_cgm']
    x_other_train = data['train']['x_other'] # Estas son las características generales
    context_train = data['train']['context']   # Este es el diccionario de contexto
    
    for episode_num in episode_bar:
        # _run_episode necesita x_cgm, x_other, y el contexto asociado a estos datos
        episode_rewards, episode_actor_loss, episode_critic_loss, updates_count = _run_episode(
            simulator, model_wrapper, 
            x_cgm_train, x_other_train, context_train, # Pasar contexto aquí
            replay_buffer,
            min_buffer_samples, batch_size, update_freq, processing_batch_size
        )
        
        avg_episode_reward = np.mean(episode_rewards) if episode_rewards else 0
        avg_actor_loss = episode_actor_loss / max(1, updates_count)
        avg_critic_loss = episode_critic_loss / max(1, updates_count)
        
        episode_bar.set_description(
            f"Episodio {episode_num+1}/{episodes}: "
            f"Recompensa={avg_episode_reward:.2f}, "
            f"Actor Loss={avg_actor_loss:.4f}, "
            f"Critic Loss={avg_critic_loss:.4f}, "
            f"Buffer={len(replay_buffer)}"
        )
        
        episode_metrics_history['rewards'].append(avg_episode_reward)
        episode_metrics_history['actor_loss'].append(avg_actor_loss)
        episode_metrics_history['critic_loss'].append(avg_critic_loss)
        episode_metrics_history['buffer_size'].append(len(replay_buffer))
    
    episode_bar.close()
    print_info(f"\nEntrenamiento DRL completado: {episodes} episodios, {len(replay_buffer)} experiencias en buffer.")
    if episode_metrics_history['rewards']:
        print_info(f"Recompensa media final (últimos 10 episodios): {np.mean(episode_metrics_history['rewards'][-10:]):.2f}")
    
    # Evaluar modelo en datos de prueba, pasando el contexto de prueba
    x_cgm_test = data['test']['x_cgm']
    x_other_test = data['test']['x_other'] # Características generales de test
    context_test = data['test']['context']   # Contexto de test

    y_pred, initial_glucose_test, carb_intake_test = _predict_test_samples(
        model_wrapper, x_cgm_test, x_other_test, context_test # Pasar context_test
    )
    
    print_debug(f"Predicciones del modelo {model_name} con contexto: {y_pred[:10]}") # Mostrar solo algunas
    non_zero_predictions = np.count_nonzero(y_pred)
    if len(y_pred) > 0:
        print_info(f"Número de predicciones no cero: {non_zero_predictions}/{len(y_pred)} ({non_zero_predictions/len(y_pred)*100:.2f}%)")
    else:
        print_info("No se generaron predicciones.")

    clinical_metrics = evaluate_clinical_metrics(
        simulator=simulator,
        predictions=y_pred,
        initial_glucose=initial_glucose_test,
        carb_intake=carb_intake_test # carb_intake_test viene de _predict_test_samples, que lo saca de context_test
    )
    
    # Guardar el modelo DRL (estado del actor y crítico, etc.)
    # model_wrapper.save_model(os.path.join(models_dir, f"{model_name}_drl.pt")) # Asumiendo que el wrapper tiene un método save_model

    return episode_metrics_history, y_pred, clinical_metrics, model_wrapper

