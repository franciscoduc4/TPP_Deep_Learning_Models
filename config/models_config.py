"""Model configuration parameters"""
from constants.constants import CONST_DEFAULT_SEED

###########################################################
###                Global Configuration                 ###
###########################################################
EARLY_STOPPING_POLICY = {
    'early_stopping': True,                         # Activar/desactivar early stopping
    'early_stopping_patience': 30,                  # Épocas a esperar antes de detener
    'early_stopping_min_delta': 0.0001,              # Mejora mínima considerada significativa
    'early_stopping_restore_best_weights': True,    # Restaurar mejores pesos al finalizar
    'early_stopping_best_val_loss': float('inf'),   # Mejor pérdida de validación
    'early_stopping_best_loss': float('inf'),       # Mejor pérdida de entrenamiento
    'early_stopping_counter': 0,                    # Contador de épocas sin mejora
    'early_stopping_best_epoch': 0,                 # Época con mejor pérdida de validación
    'early_stopping_best_weights': None,            # Mejores pesos del modelo
}

BUFFER_CONFIG = {
    'buffer_size': 100000,  # Tamaño máximo del buffer de experiencia
    'batch_size': 64,       # Tamaño del batch para entrenamiento
    'action_dim': 1,      # Dimensión de la acción (dosis de insulina)
    'hidden_dim': 256,     # Dimensión de las capas ocultas en la red neuronal
    'seed': CONST_DEFAULT_SEED  # Semilla para reproducibilidad
}

###########################################################
###         Deep Reinforcement Learning Models          ###
###########################################################
DDPG_CONFIG = {
    # Parámetros básicos de la red
    "action_dim": 1,                # Dimensión de la acción (dosis de insulina)
    "hidden_dim": 256,              # Dimensión de las capas ocultas
    
    # Parámetros de aprendizaje
    "actor_lr": 1e-4,               # Tasa de aprendizaje para el actor
    "critic_lr": 1e-3,              # Tasa de aprendizaje para el crítico
    "gamma": 0.99,                  # Factor de descuento para recompensas futuras
    "tau": 0.001,                   # Parámetro de actualización suave para redes objetivo
    "weight_decay": 1e-5,          # Decaimiento de peso para regularización
    
    # Parámetros del buffer de experiencia
    "buffer_size": 50000,          # Capacidad máxima del buffer
    
    # Límites de acción
    "max_action": 20.0,             # Valor máximo de acción (dosis máxima de insulina)
    "min_action": 0.0,              # Valor mínimo de acción (dosis mínima de insulina)
    
    # Parámetros de exploración
    "exploration_noise": 0.2,       # Desviación estándar del ruido de exploración
    
    # Otros parámetros
    "seed": CONST_DEFAULT_SEED                      # Semilla aleatoria para reproducibilidad
}

DQN_CONFIG = {
    # Discretización del espacio de acción
    'action_bins': 20,  # Número de valores discretos de dosis
    'max_action': 10.0,  # Dosis máxima de insulina
    'min_action': 0.0,   # Dosis mínima de insulina
    
    # Parámetros del algoritmo
    'gamma': 0.99,       # Factor de descuento
    'tau': 0.005,        # Tasa de actualización suave
    'learning_rate': 3e-4,  # Tasa de aprendizaje
    'weight_decay': 1e-5,  # Decaimiento de peso para regularización
    
    # Parámetros para exploración
    'epsilon_start': 1.0,  # Epsilon inicial para exploración
    'epsilon_decay': 0.995,  # Tasa de decaimiento de epsilon
    'epsilon_min': 0.01,  # Epsilon mínimo
    
    # Parámetros del entrenamiento
    'buffer_size': 100000,  # Tamaño del buffer de experiencia
    'batch_size': 64,  # Tamaño del batch
    'target_update_freq': 10,  # Frecuencia de actualización de la red target
    
    # Reproducibilidad
    'seed': CONST_DEFAULT_SEED  # Semilla para generación de números aleatorios
}

SAC_CONFIG = {
    # Dimensiones de entrada y salida
    'action_dim': 1,  # Dosis de insulina (escalar)
    'hidden_dim': 256,  # Dimensión de capas ocultas
    
    # Parámetros del algoritmo
    'gamma': 0.99,  # Factor de descuento
    'tau': 0.005,  # Tasa de actualización suave para redes objetivo
    'alpha_lr': 3e-4,  # Tasa de aprendizaje para parámetro de temperatura
    'actor_lr': 3e-4,  # Tasa de aprendizaje para actor
    'critic_lr': 3e-4,  # Tasa de aprendizaje para crítico
    'weight_decay': 1e-5,  # Decaimiento de peso para regularización
    'initial_alpha': 0.2,  # Valor inicial de la temperatura (coef. de entropía)
    
    # Limitaciones de acción
    'max_action': 10.0,  # Dosis máxima de insulina
    'min_action': 0.0,  # Dosis mínima de insulina
    
    # Parámetros del buffer
    'buffer_size': 100000,  # Tamaño del buffer de experiencia
    
    # Reproducibilidad
    'seed': CONST_DEFAULT_SEED  # Semilla para generación de números aleatorios
}

TD3_BC_CONFIG = {
    # Dimensiones de entrada y salida
    'action_dim': 1,  # Dosis de insulina (escalar)
    'hidden_dim': 256,  # Dimensión de capas ocultas
    
    # Parámetros del algoritmo
    'gamma': 0.99,  # Factor de descuento
    'tau': 0.005,  # Tasa de actualización suave para redes objetivo
    'policy_noise': 0.2,  # Ruido de la política para target policy smoothing
    'noise_clip': 0.5,  # Recorte de ruido para target policy smoothing
    'policy_delay': 2,  # Frecuencia de actualización del actor (2 = cada 2 actualizaciones)
    'alpha': 2.5,  # Peso para balancear la pérdida de BC vs RL
    
    # Limitaciones de acción
    'max_action': 10.0,  # Dosis máxima de insulina
    'min_action': 0.0,  # Dosis mínima de insulina
    'exploration_noise': 0.1,  # Ruido de exploración durante entrenamiento
    
    # Optimización
    'actor_lr': 3e-4,  # Tasa de aprendizaje para actor
    'critic_lr': 3e-4,  # Tasa de aprendizaje para crítico
    'weight_decay': 1e-5,  # Decaimiento de peso para regularización
    
    # Parámetros del buffer
    'buffer_size': 100000,  # Tamaño del buffer de experiencia
    
    # Reproducibilidad
    'seed': CONST_DEFAULT_SEED  # Semilla para generación de números aleatorios
}