import numpy as np
from typing import Union
from constants.constants import HYPER_PENALTY_BASE, HYPO_PENALTY_BASE, MAX_REWARD, SEVERE_HYPER_PENALTY, SEVERE_HYPO_PENALTY, SEVERE_HYPOGLYCEMIA_THRESHOLD, HYPOGLYCEMIA_THRESHOLD, HYPERGLYCEMIA_THRESHOLD, SEVERE_HYPERGLYCEMIA_THRESHOLD, IDEAL_LOWER_BOUND, IDEAL_UPPER_BOUND

# Función de Recompensa
def compute_reward(glucose_level: Union[float, np.ndarray]) -> float:
    """
    Calcula la recompensa basada en el nivel actual de glucosa en sangre.
    
    Parámetros:
    -----------
    glucose_level : float or np.ndarray
        Nivel de glucosa en mg/dL (scalar o trayectoria)
        
    Retorna:
    --------
    float
        Valor de recompensa promedio
    """
    # Convertir a array si es escalar
    glucose_level = np.atleast_1d(glucose_level)
    
    rewards = np.zeros_like(glucose_level, dtype=float)
    
    for i, gl in enumerate(glucose_level):
        # Hipoglucemia severa
        if gl < SEVERE_HYPOGLYCEMIA_THRESHOLD:
            rewards[i] = SEVERE_HYPO_PENALTY
        # Hipoglucemia
        elif gl < HYPOGLYCEMIA_THRESHOLD:
            severity_factor = (HYPOGLYCEMIA_THRESHOLD - gl) / (HYPOGLYCEMIA_THRESHOLD - SEVERE_HYPOGLYCEMIA_THRESHOLD)
            rewards[i] = HYPO_PENALTY_BASE * (1 + severity_factor)
        # Rango saludable por debajo del ideal
        elif gl < IDEAL_LOWER_BOUND:
            rewards[i] = MAX_REWARD * (1 - (IDEAL_LOWER_BOUND - gl) / (IDEAL_LOWER_BOUND - HYPOGLYCEMIA_THRESHOLD))
        # Rango objetivo
        elif IDEAL_LOWER_BOUND <= gl <= IDEAL_UPPER_BOUND:
            rewards[i] = MAX_REWARD
        # Rango saludable por encima del ideal
        elif gl < HYPERGLYCEMIA_THRESHOLD:
            rewards[i] = MAX_REWARD * (1 - (gl - IDEAL_UPPER_BOUND) / (HYPERGLYCEMIA_THRESHOLD - IDEAL_UPPER_BOUND))
        # Hiperglucemia
        elif gl <= SEVERE_HYPERGLYCEMIA_THRESHOLD:
            severity_factor = (gl - HYPERGLYCEMIA_THRESHOLD) / (SEVERE_HYPERGLYCEMIA_THRESHOLD - HYPERGLYCEMIA_THRESHOLD)
            rewards[i] = HYPER_PENALTY_BASE * (1 + severity_factor)
        # Hiperglucemia severa
        else:
            rewards[i] = SEVERE_HYPER_PENALTY
    
    # Retornar recompensa promedio
    return float(np.mean(rewards))

def calculate_iob(x_cgm: np.ndarray, carb_intake: float, 
                 current_iob: float = None) -> float:
    """
    Estima la cantidad de Insulina a Bordo (IOB) basado en tendencias de glucosa y carbohidratos.
    
    Parámetros:
    -----------
    x_cgm : np.ndarray
        Datos CGM para estimación de tendencia
    carb_intake : float
        Ingesta de carbohidratos en gramos
    current_iob : float, opcional
        Valor actual de IOB si está disponible
        
    Retorna:
    --------
    float
        Insulina a Bordo estimada en unidades
    """
    # Si ya se proporciona un valor de IOB, usarlo
    if current_iob is not None:
        return current_iob
    
    # Detectar si la glucosa está bajando (indicador de insulina activa)
    glucose_dropping = False
    
    if len(x_cgm.shape) > 2 and x_cgm.shape[1] > 2:
        # Formato: [muestras, tiempo, características]
        glucose_trend = x_cgm[0, -1, 0] - x_cgm[0, -3, 0]
        glucose_dropping = glucose_trend < -10  # Bajando más de 10 mg/dL
    elif len(x_cgm.shape) == 2 and x_cgm.shape[1] > 2:
        # Formato: [tiempo, características] o [muestras, tiempo]
        glucose_trend = x_cgm[-1, 0] - x_cgm[-3, 0]
        glucose_dropping = glucose_trend < -10
    elif len(x_cgm.shape) == 1 and x_cgm.shape[0] > 2:
        # Formato: [tiempo]
        glucose_trend = x_cgm[-1] - x_cgm[-3]
        glucose_dropping = glucose_trend < -10
    
    # Estimar IOB basado en carbohidratos y tendencia de glucosa
    if glucose_dropping:
        # Si la glucosa está bajando, probablemente hay insulina activa
        iob_estimate = max(1.0, carb_intake / 20.0)
    else:
        # Estimación conservadora basada en carbohidratos
        iob_estimate = carb_intake / 30.0
    
    # Limitar a un máximo razonable
    return min(iob_estimate, 5.0)
