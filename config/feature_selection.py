import polars as pl
import numpy as np
from typing import Dict, List, Optional, Tuple

from custom.printer import coloured, print_error, print_info, print_warning

# Constantes para selección de características
CORE_CGM_FEATURES = [
    'cgm_0', 'cgm_1', 'cgm_2', 'cgm_3', 'cgm_4', 'cgm_5', 
    'cgm_6', 'cgm_7', 'cgm_8', 'cgm_9', 'cgm_10', 'cgm_11',
    'cgm_12', 'cgm_13', 'cgm_14', 'cgm_15', 'cgm_16', 'cgm_17',
    'cgm_18', 'cgm_19', 'cgm_20', 'cgm_21', 'cgm_22', 'cgm_23'
]

GLUCOSE_AGGREGATED_FEATURES = [
    'glucose_last',      # Valor más reciente de glucosa
    'glucose_mean',      # Media de la ventana actual
    'glucose_std',       # Variabilidad actual
    'glucose_slope',     # Tendencia (subiendo/bajando)
    'glucose_range',     # Rango de variación
    'glucose_min',       # Mínimo en ventana
    'glucose_max'        # Máximo en ventana
]

GLUCOSE_24H_FEATURES = [
    'cgm_mean_24h',      # Media de 24h
    'cgm_std_24h',       # Variabilidad de 24h
    'mage_24h',          # Mean Amplitude of Glycemic Excursions
    'time_in_range_24h', # TIR histórico
    'hypo_percentage_24h',   # % hipoglucemia histórica
    'hyper_percentage_24h'   # % hiperglucemia histórica
]

MEAL_INSULIN_FEATURES = [
    'meal_carbs',                # Carbohidratos de la comida
    'meal_carbs_log1p',         # Transformación log de carbohidratos
    'time_since_last_meal',     # Tiempo desde última comida
    'insulin_on_board',         # Insulina activa en el cuerpo
    'insulin_on_board_log1p',   # Transformación log de IOB
    'insulin_carb_ratio',       # Ratio insulina/carbohidratos
    'insulin_sensitivity_factor' # Factor de sensibilidad
]

CONTEXTUAL_FEATURES = [
    'sleep_quality',        # Calidad del sueño (1-4, 0=sin datos)
    'work_intensity',       # Intensidad del trabajo (1-10, 0=no trabaja)
    'exercise_intensity'    # Intensidad del ejercicio (1-10, 0=sin ejercicio)
]

TEMPORAL_FEATURES = [
    'hour_of_day',         # Hora del día
    'hour_cos',            # Codificación cíclica hora
    'hour_sin',            # Codificación cíclica hora
    'day_of_week_normalized', # Día de la semana normalizado
    'day_cos',             # Codificación cíclica día
    'day_sin'              # Codificación cíclica día
]

RISK_FEATURES = [
    'hypo_risk',           # Riesgo de hipoglucemia
    'hyper_risk',          # Riesgo de hiperglucemia
    'variability_risk',    # Riesgo por variabilidad
    'overall_risk'         # Riesgo general
]

# Características principales recomendadas
RECOMMENDED_FEATURES = (
    CORE_CGM_FEATURES + 
    GLUCOSE_AGGREGATED_FEATURES + 
    GLUCOSE_24H_FEATURES +
    MEAL_INSULIN_FEATURES +
    CONTEXTUAL_FEATURES +
    TEMPORAL_FEATURES +
    RISK_FEATURES
)

# Características mínimas esenciales para el modelo
ESSENTIAL_FEATURES = [
    # CGM sequence (últimos 12 puntos más críticos)
    'cgm_12', 'cgm_13', 'cgm_14', 'cgm_15', 'cgm_16', 'cgm_17',
    'cgm_18', 'cgm_19', 'cgm_20', 'cgm_21', 'cgm_22', 'cgm_23',
    
    # Estado actual de glucosa
    'glucose_last', 'glucose_slope', 'glucose_std',
    
    # Contexto nutricional
    'meal_carbs', 'time_since_last_meal', 'insulin_on_board',
    
    # Contexto externo
    'sleep_quality', 'work_intensity', 'exercise_intensity',
    
    # Tiempo
    'hour_of_day', 'day_of_week_normalized',
    
    # Riesgos
    'hypo_risk', 'hyper_risk', 'overall_risk'
]

def get_feature_groups() -> Dict[str, List[str]]:
    """
    Retorna grupos de características organizados por tipo.
    
    Retorna:
    --------
    Dict[str, List[str]]
        Diccionario con grupos de características.
    """
    return {
        'cgm_sequence': CORE_CGM_FEATURES,
        'glucose_current': GLUCOSE_AGGREGATED_FEATURES,
        'glucose_historical': GLUCOSE_24H_FEATURES,
        'nutrition_insulin': MEAL_INSULIN_FEATURES,
        'context': CONTEXTUAL_FEATURES,
        'temporal': TEMPORAL_FEATURES,
        'risk': RISK_FEATURES
    }

def validate_features_availability(df_columns: List[str], required_features: List[str]) -> Tuple[List[str], List[str]]:
    """
    Valida qué características están disponibles en el DataFrame.
    
    Parámetros:
    -----------
    df_columns : List[str]
        Columnas disponibles en el DataFrame.
    required_features : List[str]
        Características requeridas.
        
    Retorna:
    --------
    Tuple[List[str], List[str]]
        Tupla con (características_disponibles, características_faltantes).
    """
    available_features = [feat for feat in required_features if feat in df_columns]
    missing_features = [feat for feat in required_features if feat not in df_columns]
    
    return available_features, missing_features

def _separate_feature_types(available_features: List[str]) -> Tuple[List[str], List[str]]:
    """
    Separa las características en CGM y otras características.
    """
    cgm_feature_names = [feat for feat in available_features if feat.startswith('cgm_') and feat[4:].isdigit()]
    other_feature_names = [feat for feat in available_features if not (feat.startswith('cgm_') and feat[4:].isdigit())]
    
    if cgm_feature_names:
        cgm_feature_names.sort(key=lambda x: int(x.split('_')[1]))
    
    return cgm_feature_names, other_feature_names

def _impute_cgm_features(df: pl.DataFrame, cgm_feature_names: List[str]) -> np.ndarray:
    """
    Imputa características CGM con forward fill, backward fill y relleno con 0.0.
    """
    if not cgm_feature_names:
        return np.empty((len(df), 0))
    
    cgm_select_exprs = [
        pl.col(cgm_col).forward_fill().backward_fill().fill_null(0.0).alias(cgm_col)
        for cgm_col in cgm_feature_names
    ]
    
    cgm_imputed_df = df.select(cgm_select_exprs)
    return cgm_imputed_df.to_numpy()

def _impute_other_features(df: pl.DataFrame, other_feature_names: List[str]) -> np.ndarray:
    """
    Imputa otras características rellenando NaNs con 0.0.
    """
    if not other_feature_names:
        return np.empty((len(df), 0))
    
    other_select_exprs = [
        pl.col(other_col).fill_null(0.0).alias(other_col)
        for other_col in other_feature_names
    ]
    
    other_imputed_df = df.select(other_select_exprs)
    return other_imputed_df.to_numpy()

def _reshape_cgm_data(cgm_data_np: np.ndarray, cgm_feature_names: List[str], df_length: int) -> np.ndarray:
    """
    Reforma los datos CGM al formato [samples, timesteps, features].
    """
    if cgm_data_np.shape[1] > 0:
        return cgm_data_np.reshape(cgm_data_np.shape[0], len(cgm_feature_names), 1)
    elif df_length > 0:
        return np.empty((df_length, 0, 1))
    else:
        return np.empty((0, 0, 1))

def _validate_imputation(cgm_data_np: np.ndarray, other_data_np: np.ndarray) -> None:
    """
    Valida que no queden NaNs después de la imputación.
    """
    if cgm_data_np.size > 0 and np.isnan(cgm_data_np).any():
        print_error("NaNs detectados en cgm_data_np DESPUÉS de la imputación.")
    if other_data_np.size > 0 and np.isnan(other_data_np).any():
        print_error("NaNs detectados en other_data_np DESPUÉS de la imputación.")

def _impute_and_indicate_missing(df: pl.DataFrame, column_name: str, new_indicator_col_name: str) -> Tuple[pl.Series, pl.Series]:
    """
    Imputa una columna con 0.0 y crea una columna indicadora para los NaNs originales.
    """
    indicator = df[column_name].is_null().cast(pl.Float32).alias(new_indicator_col_name)
    imputed_column = df[column_name].fill_null(0.0).alias(column_name)
    return imputed_column, indicator

def prepare_features_for_drl(df: pl.DataFrame, use_essential_only: bool = False) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepara las características para modelos DRL, asegurando la imputación de NaNs
    e introduciendo variables indicadoras para ciertas características críticas.
    Para las características CGM, se aplica forward fill, luego backward fill y finalmente se rellenan los NaNs restantes con 0.0.
    Para otras características, los NaNs se rellenan con 0.0, y para 'insulin_on_board' y 'meal_carbs',
    se añaden columnas indicadoras de valores faltantes.

    Parámetros:
    -----------
    df : pl.DataFrame
        DataFrame con datos procesados que pueden contener NaNs.
    use_essential_only : bool, opcional
        Si se deben usar solo las características esenciales (default: False).
        
    Retorna:
    --------
    Tuple[np.ndarray, np.ndarray, List[str]]
        Tupla con (cgm_sequence, other_features, other_feature_names), donde
        cgm_sequence y other_features están imputados y no contienen NaNs.
    """
    features_to_use = ESSENTIAL_FEATURES if use_essential_only else RECOMMENDED_FEATURES
    
    available_features, missing_features = validate_features_availability(
        df.columns, features_to_use
    )
    
    if missing_features:
        print_warning(f"Características faltantes que no se usarán: {missing_features}")

    # Remover características constantes conocidas antes de la separación
    constant_features_to_remove = ['activity_hypo_risk', 'sleep_hypo_risk', 'stress_hyper_risk']
    available_features = [feat for feat in available_features if feat not in constant_features_to_remove]
    if any(feat in df.columns for feat in constant_features_to_remove):
        print_info(f"Removiendo características constantes: {constant_features_to_remove}")
    
    cgm_feature_names, other_feature_names_original = _separate_feature_types(available_features)
    
    # Imputación de características CGM
    cgm_data_np = _impute_cgm_features(df, cgm_feature_names)
    
    # Imputación de otras características y creación de indicadores
    select_expressions = []
    final_other_feature_names = []

    # Características con estrategia de indicador + imputación
    features_for_indicator = {
        'insulin_on_board': 'insulin_on_board_is_missing',
        'meal_carbs': 'meal_carbs_is_missing',
        # Añadir aquí otras características si es necesario, e.g.:
        # 'insulin_carb_ratio': 'insulin_carb_ratio_is_missing',
        # 'insulin_sensitivity_factor': 'insulin_sensitivity_factor_is_missing'
    }

    for original_col, indicator_col in features_for_indicator.items():
        if original_col in other_feature_names_original or original_col in available_features:
            imputed_expr, indicator_expr = _impute_and_indicate_missing(df, original_col, indicator_col)
            select_expressions.append(imputed_expr)
            select_expressions.append(indicator_expr)
            final_other_feature_names.append(original_col)
            final_other_feature_names.append(indicator_col)

    # Resto de las "otras" características con imputación simple a 0.0
    for other_col in other_feature_names_original:
        if other_col not in features_for_indicator:
            select_expressions.append(df[other_col].fill_null(0.0).alias(other_col))
            final_other_feature_names.append(other_col)
    
    if select_expressions:
        other_imputed_df = df.select(select_expressions)
        other_data_np = other_imputed_df.to_numpy()
    elif df.height > 0 : # Si no hay otras características pero sí filas
        other_data_np = np.empty((df.height, 0), dtype=np.float32)
        print_warning("No se seleccionaron 'otras características' o todas fueron manejadas con indicadores y no quedaron más.")
    else: # No hay filas ni otras características
        other_data_np = np.empty((0, 0), dtype=np.float32)


    # Reshape CGM para formato [samples, timesteps, features]
    cgm_data_np = _reshape_cgm_data(cgm_data_np, cgm_feature_names, len(df))
    
    # Logging de información
    print_info(f"Características CGM utilizadas: {len(cgm_feature_names)}")
    print_info(f"Otras características utilizadas (incluyendo indicadoras): {len(final_other_feature_names)}")
    print_info(f"Nombres finales de otras características: {final_other_feature_names}")
    print_info(f"Forma CGM (después de imputación): {cgm_data_np.shape}")
    print_info(f"Forma otras características (después de imputación): {other_data_np.shape}")
    
    # Validación final
    _validate_imputation(cgm_data_np, other_data_np)
        
    return cgm_data_np, other_data_np, final_other_feature_names

def create_temporal_splits(df: pl.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Crea divisiones temporales respetando la secuencia temporal para datos médicos.
    
    Parámetros:
    -----------
    df : pl.DataFrame
        DataFrame con datos procesados.
    train_ratio : float, opcional
        Proporción de datos para entrenamiento (default: 0.7).
    val_ratio : float, opcional
        Proporción de datos para validación (default: 0.15).
        
    Retorna:
    --------
    Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
        Tupla con (train_df, val_df, test_df).
    """
    # Ordenar por SubjectID y Timestamp para mantener secuencia temporal
    df_sorted = df.sort(['SubjectID', 'Timestamp'])
    
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    # División por sujeto para evitar data leakage
    for subject_id in df_sorted['SubjectID'].unique():
        subject_df = df_sorted.filter(pl.col('SubjectID') == subject_id)
        
        n_samples = len(subject_df)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        train_dfs.append(subject_df[:train_end])
        val_dfs.append(subject_df[train_end:val_end])
        test_dfs.append(subject_df[val_end:])
    
    train_df = pl.concat(train_dfs) if train_dfs else pl.DataFrame()
    val_df = pl.concat(val_dfs) if val_dfs else pl.DataFrame()
    test_df = pl.concat(test_dfs) if test_dfs else pl.DataFrame()
    
    return train_df, val_df, test_df

def _extract_context_for_drl(df: pl.DataFrame) -> Dict[str, np.ndarray]:
    """
    Extrae características contextuales específicas para DRL de un DataFrame.
    Asegura que las columnas base para el contexto ('meal_carbs', 'glucose_last', 'insulin_on_board')
    sean imputadas a 0.0 si son NaN, ya que la estrategia de indicador + imputación
    se maneja en `prepare_features_for_drl` para el estado principal del modelo.
    El contexto aquí debe reflejar los valores que el modelo usará (posiblemente imputados).

    Parámetros:
    -----------
    df : pl.DataFrame
        DataFrame del cual extraer el contexto.

    Retorna:
    --------
    Dict[str, np.ndarray]
        Diccionario con las características contextuales como arrays de NumPy.
    """
    context_dict: Dict[str, np.ndarray] = {}
    
    # Características contextuales directas (sleep, work, exercise)
    # Estas ya se asume que 0 es "no data" y NaNs se imputan a 0 en prepare_features_for_drl
    context_features_direct = {
        'sleep_quality': 'sleep_quality',
        'work_intensity': 'work_intensity',
        'exercise_intensity': 'exercise_intensity'
    }
    for key, col_name in context_features_direct.items():
        if col_name in df.columns:
            context_dict[key] = df[col_name].fill_null(0.0).to_numpy() # Asegurar imputación a 0 para el contexto
        else:
            print_warning(f"Columna contextual '{col_name}' para '{key}' no encontrada. Se usará array de ceros.")
            context_dict[key] = np.zeros(len(df))

    # Características mapeadas que forman parte del contexto principal
    # (meal_carbs, glucose_last, insulin_on_board)
    # Estas columnas son imputadas (y tienen indicadores) en `prepare_features_for_drl`.
    # Para el diccionario de contexto, usamos sus valores (posiblemente imputados a 0).
    context_features_mapped = {
        'carb_intake': 'meal_carbs',
        'current_glucose': 'glucose_last',
        'iob': 'insulin_on_board'
    }
    for key, col_name in context_features_mapped.items():
        if col_name in df.columns:
            # Aquí también, fill_null(0.0) para asegurar que el contexto no tenga NaNs
            # si por alguna razón la columna original no fue procesada por la lógica de indicadores
            # (aunque debería haberlo sido en prepare_features_for_drl).
            context_dict[key] = df[col_name].fill_null(0.0).to_numpy()
        else:
            print_warning(f"Columna base para contexto '{col_name}' (para '{key}') no encontrada. Se usará array de ceros.")
            context_dict[key] = np.zeros(len(df))
            
    return context_dict

def prepare_data_for_drl_training(df: pl.DataFrame) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray],      # Train
    np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray],      # Validation
    np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]       # Test
]:
    """
    Prepara los datos para entrenamiento de modelos DRL con división temporal apropiada
    y extrae diccionarios de contexto.
    
    Parámetros:
    -----------
    df : pl.DataFrame
        DataFrame con datos procesados.
        
    Retorna:
    --------
    Tuple[
        np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray],  # x_cgm_train, x_other_train, y_train, context_train
        np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray],  # x_cgm_val, x_other_val, y_val, context_val
        np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]   # x_cgm_test, x_other_test, y_test, context_test
    ]
        Tupla con datos de entrenamiento, validación y prueba, incluyendo características CGM,
        otras características, objetivos (target) y diccionarios de contexto.
    """
    # División temporal por sujeto
    coloured("Creando divisiones temporales por sujeto...", 'blue')
    train_df, val_df, test_df = create_temporal_splits(df)
    
    coloured(f"División temporal - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}", 'green')
    
    # Preparar características para cada conjunto
    x_cgm_train, x_other_train, _feature_names_train = prepare_features_for_drl(train_df, use_essential_only=False)
    x_cgm_val, x_other_val, _ = prepare_features_for_drl(val_df, use_essential_only=False)
    x_cgm_test, x_other_test, _ = prepare_features_for_drl(test_df, use_essential_only=False)
    
    # Extraer targets
    y_train = train_df['bolus'].to_numpy()
    y_val = val_df['bolus'].to_numpy()
    y_test = test_df['bolus'].to_numpy()

    # Extraer contexto para cada conjunto
    coloured("Extrayendo datos de contexto para DRL...", 'blue')
    context_train = _extract_context_for_drl(train_df)
    context_val = _extract_context_for_drl(val_df)
    context_test = _extract_context_for_drl(test_df)
    
    # Verificar que no hay divisiones vacías
    if len(y_train) == 0 or len(y_val) == 0 or len(y_test) == 0:
        raise ValueError("Una o más divisiones de datos (train, val, test) están vacías. Ajusta las proporciones de división o verifica los datos de entrada.")
    
    coloured("Características por conjunto:", 'blue')
    coloured(f"  Train - CGM: {x_cgm_train.shape}, Other: {x_other_train.shape}, Target: {y_train.shape}", 'green')
    coloured(f"  Val   - CGM: {x_cgm_val.shape}, Other: {x_other_val.shape}, Target: {y_val.shape}", 'green')
    coloured(f"  Test  - CGM: {x_cgm_test.shape}, Other: {x_other_test.shape}, Target: {y_test.shape}", 'green')
    
    coloured("Contexto por conjunto (muestra de claves):", 'blue')
    coloured(f"  Train - Claves: {list(context_train.keys())}", 'green')
    coloured(f"  Val   - Claves: {list(context_val.keys())}", 'green')
    coloured(f"  Test  - Claves: {list(context_test.keys())}", 'green')

    return (
        x_cgm_train, x_other_train, y_train, context_train,
        x_cgm_val, x_other_val, y_val, context_val,
        x_cgm_test, x_other_test, y_test, context_test
    )

def validate_contextual_data_coverage(df: pl.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Valida la cobertura de datos contextuales por conjunto de datos.
    
    Parámetros:
    -----------
    df : pl.DataFrame
        DataFrame con datos procesados.
        
    Retorna:
    --------
    Dict[str, Dict[str, float]]
        Diccionario con estadísticas de cobertura por característica contextual.
    """
    contextual_cols = ['sleep_quality', 'work_intensity', 'exercise_intensity']
    coverage_stats = {}
    
    for col in contextual_cols:
        if col in df.columns:
            non_null_count = df[col].is_not_null().sum()
            total_count = len(df)
            coverage = (non_null_count / total_count) * 100
            unique_values = df[col].unique().to_list()
            
            coverage_stats[col] = {
                'coverage_percentage': coverage,
                'unique_values': unique_values,
                'non_null_samples': non_null_count,
                'total_samples': total_count
            }
            
            coloured(f"{col}: {coverage:.1f}% cobertura ({non_null_count}/{total_count} muestras)", 'green')
    
    return coverage_stats

def prepare_drl_data(df_processed: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Prepara los datos procesados para entrenamiento de modelos DRL.
    
    Parámetros:
    -----------
    df_processed : pl.DataFrame
        DataFrame con datos procesados de polars
        
    Retorna:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]
        (x_cgm, x_other, y, contexto_adicional)
    """
    # Extraer columnas CGM y formar tensor 3D
    cgm_columns = [f'cgm_{i}' for i in range(24)]
    x_cgm = df_processed.select(cgm_columns).to_numpy()
    # Reshape a (samples, timesteps=24, features=1)
    x_cgm = x_cgm.reshape(-1, 24, 1)
    
    # Características principales (excluyendo CGM y target)
    feature_columns = [
        'meal_carbs_log1p', 'insulin_on_board_log1p', 'glucose_last',
        'glucose_slope', 'glucose_std', 'time_since_last_meal',
        'hour_of_day_normalized', 'day_of_week_normalized',
        'insulin_carb_ratio_log1p', 'insulin_sensitivity_factor'
    ]
    
    # Rellenar NaNs con 0 para características principales
    x_other = df_processed.select(feature_columns).fill_null(0.0).to_numpy()
    
    # Target (dosis de insulina)
    y = df_processed.select('bolus_log1p').to_numpy().flatten()
    
    # Contexto adicional para DRL
    contexto_adicional = {
        'sleep_quality': df_processed.select('sleep_quality').fill_null(0.0).to_numpy().flatten(),
        'work_intensity': df_processed.select('work_intensity').fill_null(0.0).to_numpy().flatten(),
        'exercise_intensity': df_processed.select('exercise_intensity').fill_null(0.0).to_numpy().flatten(),
        'carb_intake': df_processed.select('meal_carbs_log1p').fill_null(0.0).to_numpy().flatten(),
        'current_glucose': df_processed.select('glucose_last').to_numpy().flatten(),
        'iob': df_processed.select('insulin_on_board_log1p').fill_null(0.0).to_numpy().flatten()
    }
    
    return x_cgm, x_other, y, contexto_adicional
