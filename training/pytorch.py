import os
import numpy as np
from config.models_config import BUFFER_CONFIG
from custom.DeepLearning.dl_pt import DLModelWrapperPyTorch
from custom.DeepReinforcementLearning.drl_pt import DRLModelWrapperPyTorch
from custom.ReinforcementLearning.rl_pt import RLModelWrapperPyTorch
from custom.printer import print_error, print_header, print_info, print_debug, print_warning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Callable, Optional, Any, Union
from config.params import DEBUG, TRAINING_CONFIG
from custom.early_stopping import ClinicalEarlyStopping
from models.utils.replay_buffer import ReplayBuffer
from training.common import (
    calculate_metrics, evaluate_clinical_metrics, optimize_ensemble_weights_clinical, get_model_type, enhance_features
)
from training.utils import calculate_iob, compute_reward
from constants.constants import (
    CONST_EPSILON, CONST_VAL_LOSS, CONST_LOSS, CONST_METRIC_MAE, CONST_METRIC_RMSE, CONST_METRIC_R2,
    CONST_MODELS, CONST_BEST_PREFIX, CONST_LOGS_DIR, CONST_DEFAULT_EPOCHS, 
    CONST_DEFAULT_BATCH_SIZE, CONST_DEFAULT_SEED, CONST_FIGURES_DIR, CONST_MODEL_TYPES, CONST_DURATION_HOURS, CONTEXT_FEATURE_ORDER
)
from tqdm.auto import tqdm

from validation.simulator import GlucoseSimulator

# Usar menos épocas en modo debug
CONST_EPOCHS = 2 if DEBUG else CONST_DEFAULT_EPOCHS

# Configurar dispositivo GPU si está disponible
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AllData = Dict[str, Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]]

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
                     x_other: np.ndarray, 
                     y: np.ndarray, 
                     batch_size: int = CONST_DEFAULT_BATCH_SIZE,
                     shuffle: bool = True) -> DataLoader:
    """
    Crea DataLoaders para el entrenamiento PyTorch.
    
    Parámetros:
    -----------
    x_cgm : np.ndarray
        Datos CGM con forma (muestras, pasos_tiempo, características)
    x_other : np.ndarray
        Otras características con forma (muestras, características)
    y : np.ndarray
        Valores objetivo con forma (muestras,)
    batch_size : int, opcional
        Tamaño del batch para entrenamiento (default: 32)
    shuffle : bool, opcional
        Si se deben mezclar los datos (default: True)
        
    Retorna:
    --------
    DataLoader
        DataLoader de PyTorch para el entrenamiento
    """
    dataset = CGMDataset(x_cgm, x_other, y)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available()  # Mejora el rendimiento con GPU
    )

def _prepare_model(model_wrapper: Union[nn.Module, Any],
                 x_cgm_train: np.ndarray,
                 x_other_train: np.ndarray,
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
                      x_other: np.ndarray,
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

def _extract_training_data(data: AllData) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extrae los datos de entrenamiento, validación y prueba."""
    x_cgm_train = data['train']['x_cgm']
    x_other_train = data['train']['x_other']
    y_train = data['train']['y']
    
    x_cgm_val = data['val']['x_cgm']
    x_other_val = data['val']['x_other']
    y_val = data['val']['y']
    
    x_cgm_test = data['test']['x_cgm']
    x_other_test = data['test']['x_other']
    
    return x_cgm_train, x_other_train, y_train, x_cgm_val, x_other_val, y_val, x_cgm_test, x_other_test

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

def train_and_evaluate_model_supervised(model_wrapper: Union[nn.Module, DLModelWrapperPyTorch, RLModelWrapperPyTorch, DRLModelWrapperPyTorch], 
                          model_name: str, 
                          data: AllData, # Modificado para usar AllData
                          models_dir: str = CONST_MODELS,
                          training_config: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, List[float]], np.ndarray, Dict[str, float], nn.Module]: # Asegurar que devuelve el modelo
    """
    Entrena y evalúa un modelo con énfasis en métricas clínicas.
    
    Parámetros:
    -----------
    model_wrapper : Union[nn.Module, DLModelWrapperPyTorch, RLModelWrapperPyTorch, DRLModelWrapperPyTorch]
        Modelo a entrenar
    model_name : str
        Nombre del modelo para guardado y registro
    data : AllData
        Diccionario con datos de entrenamiento, validación y prueba.
        Cada clave ('train', 'val', 'test') contiene un diccionario con
        'x_cgm', 'x_other', 'y', y opcionalmente 'context'.
    models_dir : str, opcional
        Directorio para guardar modelos (default: "models")
    training_config : Dict[str, Any], opcional
        Configuración de entrenamiento
        
    Retorna:
    --------
    Tuple[Dict[str, List[float]], np.ndarray, Dict[str, float], nn.Module]
        (history, predictions, metrics clínicas, modelo_entrenado)
    """
    # Configurar parámetros de entrenamiento
    current_training_config = _setup_training_config(training_config)
    
    # Extraer datos
    x_cgm_train, x_other_train, y_train, x_cgm_val, x_other_val, y_val, x_cgm_test, x_other_test = _extract_training_data(data)
    _y_test = data['test']['y']  # y_test no se usa directamente aquí para el entrenamiento supervisado
    
    # Extraer datos de carbohidratos
    carb_intake_val, carb_intake_test = _extract_carb_intake_data(data, x_other_val, x_other_test)
    
    # Extraer parámetros de configuración
    epochs = current_training_config.get('epochs', CONST_EPOCHS)
    batch_size = current_training_config.get('batch_size', CONST_DEFAULT_BATCH_SIZE)
    learning_rate = current_training_config.get('learning_rate', 0.001)
    patience = current_training_config.get('patience', 30)
    monitor = current_training_config.get('monitor', 'time_in_range')
    mode = current_training_config.get('mode', 'max')
    
    # Crear directorios necesarios
    os.makedirs(models_dir, exist_ok=True)
    log_dir = os.path.join(models_dir, CONST_LOGS_DIR, model_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # Preparar el modelo
    actual_model = _prepare_model(model_wrapper, x_cgm_train, x_other_train, y_train)
    
    # Configurar componentes de entrenamiento
    optimizer, criterion, scheduler = _setup_training_components(actual_model, learning_rate, patience)
    
    # Crear dataloaders
    train_loader = create_dataloaders(x_cgm_train, x_other_train, y_train, batch_size)
    val_loader = create_dataloaders(x_cgm_val, x_other_val, y_val, batch_size, shuffle=False)
    
    # Inicializar simulador para métricas clínicas
    simulator = GlucoseSimulator()
    
    # Inicializar early stopping basado en métricas clínicas
    early_stopping = ClinicalEarlyStopping(
        patience=patience, 
        restore_best_weights=True,
        monitor=monitor,
        mode=mode
    )
    
    # Historial de entrenamiento (solo métricas clínicas)
    history: Dict[str, List[float]] = {
        'loss': [],
        'val_loss': [],
        'time_in_range': [],
        'time_below_range': [],
        'time_above_range': [],
        'time_severe_below': [],
        'time_severe_above': []
    }
    
    # Bucle de entrenamiento
    print(f"\nEntrenando modelo {model_name}...")
    print_info(f"Configuración de entrenamiento: {epochs} épocas, batch size: {batch_size}, lr: {learning_rate}")
    print_info(f"Datos: {len(x_cgm_train)} ejemplos de entrenamiento, {len(x_cgm_val) if x_cgm_val is not None else 0} ejemplos de validación")
    
    # Extraer valores iniciales de glucosa para simulación clínica
    initial_glucose_val = np.array([x_cgm_val[i, -1, 0] for i in range(len(x_cgm_val))])
    initial_glucose_test = np.array([x_cgm_test[i, -1, 0] for i in range(len(x_cgm_test))])
    
    progress_bar = tqdm(range(epochs), desc="Entrenamiento")
    for epoch in progress_bar:
        # Ejecutar época de entrenamiento
        avg_train_loss = _run_train_epoch(actual_model, train_loader, optimizer, criterion) # type: ignore
        
        # Fase de validación
        if x_cgm_val is not None:
            avg_val_loss, clinical_metrics_val = _process_validation_epoch(
                actual_model, val_loader, criterion, x_cgm_val, x_other_val, 
                simulator, initial_glucose_val, carb_intake_val
            )
            
            # Actualizar historial con las métricas
            _update_training_history(history, avg_train_loss, avg_val_loss, clinical_metrics_val)
            
            # Determinar la métrica a monitorear para early stopping
            monitor_value = _get_monitor_value(monitor, clinical_metrics_val, avg_val_loss)
            
            # Descripción mejorada de la barra de progreso con métricas clínicas
            progress_bar.set_description(
                f"Entrenamiento: TSBR: {clinical_metrics_val['time_severe_below']:.2f}%; TBR: {clinical_metrics_val['time_below_range']:.2f}%; TIR: {clinical_metrics_val['time_in_range']:.2f}%; TAR: {clinical_metrics_val['time_above_range']:.2f}%; TSAR: {clinical_metrics_val['time_severe_above']:.2f}%; loss: {avg_train_loss:.4f}; val_loss: {avg_val_loss:.4f}"
            )
            
            # Comprobar early stopping basado en métricas clínicas
            if early_stopping(actual_model, monitor_value): # type: ignore
                print_info(f"Early stopping en época {epoch+1} - Mejor {monitor}: {early_stopping.best_score:.2f}")
                break
            
            # Paso del scheduler basado en pérdida de validación
            scheduler.step(avg_val_loss)
        else:
            # Sin validación, solo mostrar pérdida de entrenamiento
            progress_bar.set_description(f"Entrenamiento: loss: {avg_train_loss:.4f}")
            history['loss'].append(avg_train_loss)
    
    # Guardar modelo final
    torch.save(actual_model.state_dict(), os.path.join(models_dir, f'{model_name}.pt'))
    
    # Generar predicciones finales
    safe_predictions = _generate_final_predictions(actual_model, model_wrapper, x_cgm_test, x_other_test)
    
    # Calcular métricas clínicas en datos de prueba
    clinical_metrics_test = evaluate_clinical_metrics(
        simulator=simulator,
        predictions=safe_predictions,
        initial_glucose=initial_glucose_test,
        carb_intake=carb_intake_test
    )
    
    # Mostrar métricas clínicas al finalizar
    print_info(f"Métricas clínicas para {model_name}:")
    print_info(f"  Tiempo en Rango: {clinical_metrics_test['time_in_range']:.2f}%")
    print_info(f"  Tiempo Bajo Rango: {clinical_metrics_test['time_below_range']:.2f}%")
    print_info(f"  Tiempo Sobre Rango: {clinical_metrics_test['time_above_range']:.2f}%")
    
    return history, safe_predictions, clinical_metrics_test, actual_model # Devuelve el modelo entrenado

def train_model_sequential(model_creator: Callable, 
                         name: str, 
                         input_shapes: Tuple[Tuple[int, ...], Tuple[int, ...]], 
                         all_data: AllData,
                         models_dir: str = CONST_MODELS) -> Dict[str, Any]:
    """
    Entrena un modelo secuencialmente con los datos proporcionados.
    
    Parámetros:
    -----------
    model_creator : Callable
        Función creadora del modelo.
    name : str
        Nombre del modelo.
    input_shapes : Tuple[Tuple[int, ...], Tuple[int, ...]]
        Formas de entrada para el modelo (cgm_shape, other_shape).
    all_data : AllData
        Diccionario que contiene los conjuntos de datos 'train', 'val', y 'test'.
        Cada conjunto es un diccionario con 'x_cgm', 'x_other', 'y', y 'context'.
    models_dir : str, opcional
        Directorio para guardar modelos (default: CONST_MODELS).
    
    Retorna:
    --------
    Dict[str, Any]
        Diccionario con resultados del entrenamiento, incluyendo nombre, historial,
        predicciones, métricas y la instancia del modelo entrenado.
    """
    # Crear modelo
    model_wrapper = model_creator(input_shapes[0], input_shapes[1])
        
    # Configuración de entrenamiento general (puede ser específica por tipo de modelo luego)
    # Se pasa 'all_data' directamente a las funciones de entrenamiento específicas.
    base_training_config = {
        'epochs': CONST_EPOCHS,
        'batch_size': CONST_DEFAULT_BATCH_SIZE,
        'learning_rate': 1e-4, # Común para DL, DRL puede tener otra
        'patience': 30,
        'min_delta': 0.0001,
        'restore_best_weights': True,
        # DRL specific, pero pueden estar aquí y ser ignoradas por DL
        'episodes': CONST_EPOCHS * 5, 
        'update_freq': 10,
        'processing_batch_size': 128,
        'gamma': TRAINING_CONFIG.get('gamma', 0.99),
        'tau': TRAINING_CONFIG.get('tau', 0.01),
        'buffer_size': TRAINING_CONFIG.get('buffer_size', 100000),
    }
    
    history: Dict[str, List[float]]
    y_pred: np.ndarray
    metrics: Dict[str, float]
    trained_model_instance: Union[DLModelWrapperPyTorch, RLModelWrapperPyTorch, DRLModelWrapperPyTorch, nn.Module]

    if isinstance(model_wrapper, (DRLModelWrapperPyTorch, RLModelWrapperPyTorch)):
        # Configuración específica para DRL si es necesario, o usar la base_training_config
        drl_config = base_training_config.copy() 
        # Ajustar LR para DRL si es diferente, ej: drl_config['learning_rate'] = 3e-4
        
        history, y_pred, metrics, trained_model_instance = train_and_evaluate_model_drl(
            model_wrapper=model_wrapper,
            model_name=name,
            data=all_data, 
            models_dir=models_dir,
            training_config=drl_config
        )
    elif isinstance(model_wrapper, DLModelWrapperPyTorch) or isinstance(model_wrapper, nn.Module): # nn.Module para modelos no wrappeados
        # Configuración específica para DL supervisado
        supervised_config = base_training_config.copy()
        supervised_config['monitor'] = 'time_in_range' # Ejemplo de métrica para DL
        supervised_config['mode'] = 'max'
        # supervised_config['learning_rate'] = 1e-3 # Ejemplo de LR diferente para DL

        history, y_pred, metrics, trained_model_instance = train_and_evaluate_model_supervised(
            model_wrapper=model_wrapper, 
            model_name=name,
            data=all_data, 
            models_dir=models_dir,
            training_config=supervised_config 
        )
    else:
        raise ValueError(f"Tipo de modelo desconocido para {name}: {type(model_wrapper)}")
    
    # Limpiar memoria
    # trained_model_instance ya es el modelo, no necesitamos borrar model_wrapper si es el mismo
    if model_wrapper is not trained_model_instance:
        del model_wrapper
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return {
        'name': name,
        'history': history,
        'predictions': y_pred.tolist() if isinstance(y_pred, np.ndarray) else y_pred,
        'metrics': metrics,
        'model': trained_model_instance 
    }


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

def train_multiple_models(model_creators: Dict[str, Callable], 
                        input_shapes: Tuple[Tuple[int, ...], Tuple[int, ...]],
                        all_data: AllData,
                        models_dir: str = CONST_MODELS
                        ) -> Tuple[Dict[str, Dict[str, List[float]]], Dict[str, np.ndarray], Dict[str, Dict[str, float]], Dict[str, Any]]:
    """
    Entrena múltiples modelos y recopila sus resultados.
    
    Parámetros:
    -----------
    model_creators : Dict[str, Callable]
        Diccionario de funciones creadoras de modelos indexadas por nombre.
    input_shapes : Tuple[Tuple[int, ...], Tuple[int, ...]]
        Formas de las entradas (CGM, otras).
    all_data : AllData
        Diccionario que contiene los conjuntos de datos 'train', 'val', y 'test'.
        Cada conjunto es un diccionario con 'x_cgm', 'x_other', 'y', y 'context'.
    models_dir : str, opcional
        Directorio para guardar modelos (default: "models").
        
    Retorna:
    --------
    Tuple[Dict[str, Dict[str, List[float]]], Dict[str, np.ndarray], Dict[str, Dict[str, float]], Dict[str, Any]]
        (historiales, predicciones, métricas clínicas, modelos entrenados).
    """
    models_names = list(model_creators.keys())
    
    model_results: List[Dict[str, Any]] = []
    # Tipado más específico para los diccionarios de resultados
    histories: Dict[str, Dict[str, List[float]]] = {}
    predictions: Dict[str, np.ndarray] = {}
    metrics: Dict[str, Dict[str, float]] = {}
    trained_models: Dict[str, Any] = {}
    
    for name in models_names:
        print_header(f"Entrenando modelo: {name}")
        # Llamada a train_model_sequential con la estructura de datos agrupada
        result = train_model_sequential(
            model_creators[name], 
            name, 
            input_shapes,
            all_data, # Pasar el diccionario all_data directamente
            models_dir
        )
        model_results.append(result)
    
    # Guardar resultados
    for result_item in model_results:
        model_name_key = result_item['name']
        histories[model_name_key] = result_item['history']
        # Asegurar que las predicciones son np.ndarray
        preds_list = result_item['predictions']
        predictions[model_name_key] = np.array(preds_list) if isinstance(preds_list, list) else preds_list
        metrics[model_name_key] = result_item['metrics']
        trained_models[model_name_key] = result_item['model']
    
    return histories, predictions, metrics, trained_models

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
    model_wrapper: Union[DLModelWrapperPyTorch, RLModelWrapperPyTorch, DRLModelWrapperPyTorch],
    replay_buffer: ReplayBuffer,
    min_buffer_samples: int,
    current_batch_idx: int,
    update_freq: int,
    training_batch_size: int
) -> Tuple[float, float, bool]:
    """
    Intenta actualizar el modelo DRL si se cumplen las condiciones (buffer suficiente y frecuencia de actualización).

    Parámetros:
    -----------
    model_wrapper : Union[DLModelWrapperPyTorch, RLModelWrapperPyTorch, DRLModelWrapperPyTorch]
        Wrapper del modelo DRL que contiene el agente (ej. DDPG).
    replay_buffer : ReplayBuffer
        Buffer de repetición con las experiencias recolectadas.
    min_buffer_samples : int
        Número mínimo de muestras requeridas en el buffer para comenzar el entrenamiento.
    current_batch_idx : int
        Índice del batch actual dentro del episodio.
    update_freq : int
        Frecuencia (en número de batches procesados) con la que se debe intentar actualizar el modelo.
    training_batch_size : int
        Tamaño del batch de experiencias a muestrear del buffer para el paso de entrenamiento del modelo.

    Retorna:
    --------
    Tuple[float, float, bool]
        Una tupla conteniendo:
        - actor_loss_update (float): La pérdida del actor si el modelo fue actualizado, sino 0.0.
        - critic_loss_update (float): La pérdida del crítico si el modelo fue actualizado, sino 0.0.
        - was_updated (bool): True si el modelo fue actualizado, False en caso contrario.
    """
    actor_loss_update = 0.0
    critic_loss_update = 0.0
    was_updated = False

    if len(replay_buffer) >= min_buffer_samples and (current_batch_idx + 1) % update_freq == 0:
        # El método run_training_step del agente DRL maneja el muestreo internamente.
        loss_info = model_wrapper.model.run_training_step(replay_buffer, training_batch_size)
        if loss_info: # loss_info puede ser None si el entrenamiento no ocurrió por alguna razón interna
            actor_loss_update = loss_info.get('actor_loss', 0.0)
            critic_loss_update = loss_info.get('critic_loss', 0.0)
            was_updated = True
    return actor_loss_update, critic_loss_update, was_updated

def _run_episode(
    simulator: GlucoseSimulator, 
    model_wrapper: Union[DLModelWrapperPyTorch, RLModelWrapperPyTorch, DRLModelWrapperPyTorch], 
    x_cgm_train: np.ndarray, 
    x_other_train: np.ndarray, 
    context_train: Dict[str, np.ndarray], 
    replay_buffer: ReplayBuffer, 
    min_buffer_samples: int, 
    training_batch_size: int, # Renombrado desde batch_size para claridad
    update_freq: int, 
    processing_batch_size: int
) -> Tuple[float, float, float, int]:
    """
    Ejecuta un episodio completo de interacción con el entorno (simulador),
    recolecta experiencias y actualiza el modelo DRL periódicamente.

    Parámetros:
    -----------
    simulator : GlucoseSimulator
        Instancia del simulador de glucosa.
    model_wrapper : Union[DLModelWrapperPyTorch, RLModelWrapperPyTorch, DRLModelWrapperPyTorch]
        Wrapper del modelo DRL.
    x_cgm_train : np.ndarray
        Datos CGM de entrenamiento para el episodio.
    x_other_train : np.ndarray
        Otras características de entrenamiento para el episodio.
    context_train : Dict[str, np.ndarray]
        Datos de contexto asociados a los datos de entrenamiento.
    replay_buffer : ReplayBuffer
        Buffer para almacenar las transiciones (estado, acción, recompensa, siguiente_estado, done).
    min_buffer_samples : int
        Número mínimo de muestras en el replay_buffer antes de iniciar las actualizaciones del modelo.
    training_batch_size : int
        Tamaño del batch a muestrear del replay_buffer para cada paso de entrenamiento del modelo.
    update_freq : int
        Frecuencia (en batches de procesamiento) con la que se actualiza el modelo DRL.
    processing_batch_size : int
        Tamaño del batch para procesar los datos de entrada (x_cgm_train, x_other_train).

    Retorna:
    --------
    Tuple[float, float, float, int]
        Una tupla conteniendo:
        - episode_total_rewards (float): Suma total de recompensas obtenidas en el episodio.
        - avg_actor_loss (float): Pérdida promedio del actor durante el episodio.
        - avg_critic_loss (float): Pérdida promedio del crítico durante el episodio.
        - updates_count (int): Número de veces que el modelo fue actualizado durante el episodio.
    """
    episode_total_rewards = 0.0
    cumulative_actor_loss = 0.0
    cumulative_critic_loss = 0.0
    updates_count = 0
    
    num_samples_in_data = len(x_cgm_train)
    if num_samples_in_data == 0:
        print_warning("No hay datos de entrenamiento para ejecutar el episodio.")
        return 0.0, 0.0, 0.0, 0
        
    max_processing_steps = (num_samples_in_data + processing_batch_size - 1) // processing_batch_size
    
    print_debug(
        f"Iniciando episodio. Máx. pasos de procesamiento: {max_processing_steps}, "
        f"Buffer actual: {len(replay_buffer)}, Mínimo para entrenar: {min_buffer_samples}"
    )

    for batch_idx in range(max_processing_steps):
        start_idx = batch_idx * processing_batch_size
        end_idx = min((batch_idx + 1) * processing_batch_size, num_samples_in_data)
        
        if start_idx >= end_idx: # No debería ocurrir si max_processing_steps está bien calculado
            break

        current_batch_x_cgm = x_cgm_train[start_idx:end_idx]
        current_batch_x_other = x_other_train[start_idx:end_idx]
        
        current_batch_context = _extract_batch_context(
            context_train, start_idx, end_idx, num_samples_in_data
        )

        # Simular interacciones y añadir a replay buffer
        batch_rewards_list = _process_batch_data(
            simulator, model_wrapper, current_batch_x_cgm, current_batch_x_other, 
            current_batch_context, replay_buffer, model_wrapper.model.device
        )
        episode_total_rewards += sum(batch_rewards_list)
        
        # Intentar actualizar el modelo DRL
        actor_loss_increment, critic_loss_increment, model_was_updated = _try_update_drl_model(
            model_wrapper, replay_buffer, min_buffer_samples,
            batch_idx, update_freq, training_batch_size
        )
        
        if model_was_updated:
            cumulative_actor_loss += actor_loss_increment
            cumulative_critic_loss += critic_loss_increment
            updates_count += 1
        
        # Nota: La condición de finalización temprana del episodio (done_flag de _process_batch_data)
        # no se está utilizando actualmente para detener el episodio. El episodio corre por todos los datos.

    avg_actor_loss = cumulative_actor_loss / updates_count if updates_count > 0 else 0.0
    avg_critic_loss = cumulative_critic_loss / updates_count if updates_count > 0 else 0.0
    
    print_debug(
        f"Episodio finalizado. Recompensa total: {episode_total_rewards}, Actualizaciones: {updates_count}, "
        f"Pérdida Actor (promedio): {avg_actor_loss:.4f}, Pérdida Crítico (promedio): {avg_critic_loss:.4f}"
    )
    
    return episode_total_rewards, avg_actor_loss, avg_critic_loss, updates_count
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

