import numpy as np
from typing import Tuple, List, Dict, Any, Optional

from constants.constants import SEVERE_HYPERGLYCEMIA_THRESHOLD, SEVERE_HYPOGLYCEMIA_THRESHOLD
from training.utils import compute_reward

class GlucoseSimulator:
    """
    Simulador de dinámica de glucosa para validar dosis de insulina.
    
    Implementa un modelo fisiológico simplificado para predecir el efecto
    de la insulina y los carbohidratos en la glucosa sanguínea.
    
    Parámetros:
    -----------
    insulin_sensitivity : float, opcional
        Factor de sensibilidad a la insulina (mg/dL por unidad) (default: 50)
    carb_ratio : float, opcional
        Ratio insulina:carbohidratos (gramos por unidad) (default: 10)
    basal_glucose_impact : float, opcional
        Aumento de glucosa basal por hora sin insulina (default: 20)
    insulin_duration_hours : float, opcional
        Duración de acción de la insulina en horas (default: 4)
    """
    def __init__(
        self,
        insulin_sensitivity: float = 50,
        carb_ratio: float = 10,
        basal_glucose_impact: float = 20,
        insulin_duration_hours: float = 4
    ) -> None:
        self.insulin_sensitivity = insulin_sensitivity  # mg/dL por unidad
        self.carb_ratio = carb_ratio  # gramos por unidad
        self.basal_glucose_impact = basal_glucose_impact  # mg/dL por hora
        self.insulin_duration_hours = insulin_duration_hours  # horas
        
        # Parámetros internos para modelado
        self.insulin_decay = np.log(2) / (insulin_duration_hours / 2)  # Vida media
        
    def predict_glucose_trajectory(
        self,
        initial_glucose: float,
        insulin_doses: List[float],
        carb_intakes: List[float],
        timestamps: List[float],
        prediction_horizon: int = 12
    ) -> np.ndarray:
        """
        Predice la trayectoria de glucosa basada en dosis de insulina y carbohidratos.
        
        Parámetros:
        -----------
        initial_glucose : float
            Glucosa inicial en mg/dL
        insulin_doses : List[float]
            Lista de dosis de insulina en unidades
        carb_intakes : List[float]
            Lista de ingestas de carbohidratos en gramos
        timestamps : List[float]
            Tiempos relativos en horas (0 = inicio)
        prediction_horizon : int, opcional
            Horas a predecir después del último evento (default: 12)
            
        Retorna:
        --------
        np.ndarray
            Trayectoria de glucosa predicha cada 5 minutos
        """
        # Tiempo total en horas
        total_duration = max(timestamps) + prediction_horizon
        
        # Puntos de tiempo para predicción (cada 5 minutos)
        time_points = np.arange(0, total_duration, 5/60)
        
        # Inicializar trayectoria de glucosa
        glucose_trajectory = np.zeros_like(time_points)
        glucose_trajectory[0] = initial_glucose
        
        # Para cada punto de tiempo
        for i in range(1, len(time_points)):
            t = time_points[i]
            dt = time_points[i] - time_points[i-1]  # Diferencia de tiempo en horas
            
            # Calcular efectos de insulina, carbohidratos y basal
            insulin_effect = self._calculate_insulin_effect(insulin_doses, timestamps, t, dt)
            carb_effect = self._calculate_carb_effect(carb_intakes, timestamps, t, dt)
            basal_effect = self.basal_glucose_impact * dt
            
            # Actualizar nivel de glucosa
            glucose_trajectory[i] = glucose_trajectory[i-1] + carb_effect - insulin_effect + basal_effect
            
            # Limitar valores mínimos (no puede ser menor a 40 mg/dL fisiológicamente)
            glucose_trajectory[i] = max(40, glucose_trajectory[i])
        
        return glucose_trajectory

    def _calculate_insulin_effect(self, insulin_doses: List[float], timestamps: List[float], 
                                  current_time: float, dt: float) -> float:
        """Calcula el efecto de insulina activa (IOB) en el tiempo actual."""
        insulin_effect = 0
        for dose, dose_time in zip(insulin_doses, timestamps):
            if current_time > dose_time:
                time_since_dose = current_time - dose_time
                if time_since_dose < self.insulin_duration_hours:
                    effect_fraction = self._get_insulin_effect_fraction(time_since_dose)
                    effect = dose * self.insulin_sensitivity * effect_fraction * dt
                    insulin_effect += effect
        return insulin_effect

    def _get_insulin_effect_fraction(self, time_since_dose: float) -> float:
        """Calcula la fracción de efecto de insulina basada en el tiempo transcurrido."""
        if time_since_dose < 2:
            # Fase creciente (0-2 horas)
            return time_since_dose / 2
        else:
            # Fase decreciente (2-4 horas)
            return 1 - (time_since_dose - 2) / (self.insulin_duration_hours - 2)

    def _calculate_carb_effect(self, carb_intakes: List[float], timestamps: List[float], 
                               current_time: float, dt: float) -> float:
        """Calcula el efecto de carbohidratos activos (COB) en el tiempo actual."""
        carb_effect = 0
        for carbs, carb_time in zip(carb_intakes, timestamps):
            if current_time > carb_time:
                time_since_intake = current_time - carb_time
                if time_since_intake < 3:
                    effect_fraction = self._get_carb_effect_fraction(time_since_intake)
                    # Conversión carbohidratos a glucosa (mg/dL)
                    # Aproximadamente 1g de carbohidratos eleva 5 mg/dL para un adulto promedio
                    effect = carbs * 5 * effect_fraction * dt
                    carb_effect += effect
        return carb_effect

    def _get_carb_effect_fraction(self, time_since_intake: float) -> float:
        """Calcula la fracción de efecto de carbohidratos basada en el tiempo transcurrido."""
        if time_since_intake < 1:
            # Fase creciente (0-1 hora)
            return time_since_intake
        else:
            # Fase decreciente (1-3 horas)
            return 1 - (time_since_intake - 1) / 2
    
    def step(self, 
             action_insulin: float, 
             current_glucose: float, 
             carb_intake: float) -> Tuple[float, float, bool, Dict[str, Any]]:
        """
        Simula un paso de tiempo (5 minutos) en el entorno de glucosa.

        Parámetros:
        -----------
        action_insulin : float
            Dosis de insulina administrada en este paso (unidades).
        current_glucose : float
            Nivel de glucosa actual (mg/dL).
        carb_intake : float
            Ingesta de carbohidratos en este paso (gramos).

        Retorna:
        --------
        Tuple[float, float, bool, Dict[str, Any]]
            (siguiente_glucosa, recompensa, finalizado, info)
        """
        dt = 5 / 60  # Paso de tiempo de 5 minutos en horas

        # Efecto de la insulina administrada en este paso
        # Usando la lógica de la fase creciente de predict_glucose_trajectory:
        # effect_fraction = time_since_dose / 2 (donde time_since_dose = dt)
        # insulin_effect_calc = action_insulin * self.insulin_sensitivity * effect_fraction * dt
        insulin_effect_calc = action_insulin * self.insulin_sensitivity * (dt / 2.0) * dt

        # Efecto de los carbohidratos ingeridos en este paso
        # Usando la lógica de la fase creciente de predict_glucose_trajectory:
        # effect_fraction = time_since_intake / 1 (donde time_since_intake = dt, pico a 1h)
        # carb_effect_calc = carb_intake * 5 * effect_fraction * dt (5 es factor de conversión g a mg/dL)
        carb_effect_calc = carb_intake * 5 * (dt / 1.0) * dt
        
        # Efecto de producción de glucosa basal
        basal_effect_calc = self.basal_glucose_impact * dt
        
        # Calcular siguiente nivel de glucosa
        next_glucose = current_glucose + carb_effect_calc - insulin_effect_calc + basal_effect_calc
        
        # Limitar valores mínimos y máximos fisiológicos
        next_glucose = max(20.0, min(next_glucose, 600.0)) # Rango fisiológico amplio

        # Calcular recompensa
        reward = compute_reward(next_glucose)
        
        # Determinar si el episodio ha terminado
        # Termina si la glucosa alcanza niveles extremadamente peligrosos
        done = bool(next_glucose <= SEVERE_HYPOGLYCEMIA_THRESHOLD - 10 or \
                    next_glucose >= SEVERE_HYPERGLYCEMIA_THRESHOLD + 100) # Umbrales más amplios para 'done'
        
        info: Dict[str, Any] = {}
        
        return next_glucose, reward, done, info