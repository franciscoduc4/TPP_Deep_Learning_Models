import polars as pl
import numpy as np
from typing import Dict, List, Optional, Tuple

from config.params import CONFIG_PROCESSING
from constants.constants import CONTEXT_FEATURE_ORDER, GLUCOSE_COL, SUBJECT_ID_COL, TIMESTAMP_COL
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
    all_other_features = (
        GLUCOSE_AGGREGATED_FEATURES +
        GLUCOSE_24H_FEATURES +
        MEAL_INSULIN_FEATURES +
        CONTEXTUAL_FEATURES + 
        TEMPORAL_FEATURES +
        RISK_FEATURES
    )
    # La duplicación de nombres de características en la lista all_other_features
    # es generalmente inofensiva para la selección de columnas (pl.DataFrame.select),
    # pero si se requiere una lista única, se podría usar list(dict.fromkeys(all_other_features)).
    # Por ahora, se asume que las listas de características base son disjuntas o que
    # la duplicación intencional es manejada adecuadamente.

    return {
        'cgm_features': CORE_CGM_FEATURES,
        'glucose_current': GLUCOSE_AGGREGATED_FEATURES,
        'glucose_historical': GLUCOSE_24H_FEATURES,
        'nutrition_insulin': MEAL_INSULIN_FEATURES,
        'context': CONTEXTUAL_FEATURES, # Estas son las que no están en CONTEXT_FEATURE_ORDER explícitamente para el estado DRL
        'temporal': TEMPORAL_FEATURES,
        'risk': RISK_FEATURES,
        'other_features': list(set( # Crear una lista única de "otras" características no CGM y no de contexto directo
            GLUCOSE_AGGREGATED_FEATURES + 
            GLUCOSE_24H_FEATURES +
            MEAL_INSULIN_FEATURES +
            CONTEXTUAL_FEATURES + # Incluir estas aquí también para la parte 'x_other' del estado
            TEMPORAL_FEATURES +
            RISK_FEATURES
        ) - set(CONTEXT_FEATURE_ORDER) - set(CORE_CGM_FEATURES)),
        'explicit_context_features': CONTEXT_FEATURE_ORDER, # Usado para construir parte del estado DRL
        'target_variable': 'bolus_log1p' # O la columna target que se use
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
    Imputa valores faltantes en una columna y crea una columna indicadora de faltantes.
    Si la columna es completamente NaN, se imputa con 0.0.

    Parámetros:
    -----------
    df : pl.DataFrame
        DataFrame de entrada.
    column_name : str
        Nombre de la columna a imputar.
    new_indicator_col_name : str
        Nombre para la nueva columna indicadora de faltantes.

    Retorna:
    --------
    Tuple[pl.Series, pl.Series]
        Una tupla conteniendo la serie imputada y la serie indicadora.
    """
    if column_name not in df.columns:
        print_error(f"La columna '{column_name}' no existe en el DataFrame proporcionado a _impute_and_indicate_missing.")
        # Retornar series de nulos/ceros del tamaño esperado para evitar fallos posteriores,
        # aunque esto indica un problema de lógica previo.
        num_rows = df.height
        imputed_values = pl.Series(name=column_name, values=[0.0] * num_rows)
        is_missing_indicator = pl.Series(name=new_indicator_col_name, values=[1] * num_rows, dtype=pl.Int8)
        return imputed_values, is_missing_indicator

    is_missing_indicator = df[column_name].is_null().cast(pl.Int8)
    
    imputed_values: pl.Series
    if df[column_name].null_count() == df.height:
        # Si todos los valores son nulos, imputar con 0.0
        imputed_values = pl.lit(0.0).cast(df[column_name].dtype).alias(column_name)
    elif df[column_name].dtype in [pl.Float32, pl.Float64, pl.Float64]: # Incluye pl.Float64
        median_val = df[column_name].median()
        if median_val is None: # Esto puede ocurrir si, a pesar de no ser todos nulos, la mediana no se puede calcular (raro para numéricos) o es nula.
            imputed_values = df[column_name].fill_null(0.0)
        else:
            imputed_values = df[column_name].fill_null(median_val)
    elif df[column_name].dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]:
        median_val = df[column_name].median()
        if median_val is None:
            imputed_values = df[column_name].fill_null(0) # Imputar con 0 para enteros
        else:
            imputed_values = df[column_name].fill_null(median_val) # Imputar con la mediana para enteros
    else:
        # Para otros tipos de datos (ej. String, Boolean), la imputación con 0.0 o mediana numérica no es apropiada.
        # Se podría usar la moda o una categoría específica. Por ahora, se usa un valor por defecto si es posible o error.
        # Para el contexto actual, se asumen características numéricas o que ya fueron preprocesadas.
        # Si se llega aquí con un tipo inesperado, es mejor advertir y usar un fallback genérico.
        print_warning(f"Imputación para el tipo de dato {df[column_name].dtype} de la columna '{column_name}' no manejada específicamente. Usando fill_null(0.0) como fallback.")
        try:
            imputed_values = df[column_name].fill_null(0.0) # Fallback genérico, puede fallar si el tipo no es compatible
        except Exception: # pylint: disable=broad-except
            print_error(f"No se pudo aplicar fill_null(0.0) a la columna '{column_name}' de tipo {df[column_name].dtype}. La columna se dejará como está para esta imputación.")
            imputed_values = df[column_name]


    # Asegurar que la serie devuelta tenga el nombre correcto
    if isinstance(imputed_values, pl.Expr):
        imputed_values_series = df.select(imputed_values.alias(column_name))[column_name]
    else: # Ya es una pl.Series
        imputed_values_series = imputed_values.alias(column_name)
        
    return imputed_values_series, is_missing_indicator.alias(new_indicator_col_name)

def _handle_empty_dataframe(
    cgm_feature_cols: List[str], 
    other_feature_cols: List[str], 
    context_feature_order: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """Maneja el caso cuando el DataFrame está vacío."""
    empty_np_array = np.array([])
    empty_context_dict: Dict[str, np.ndarray] = {key: empty_np_array for key in context_feature_order}
    
    if cgm_feature_cols == CORE_CGM_FEATURES:
        x_cgm_empty = np.empty((0, len(CORE_CGM_FEATURES), 1), dtype=np.float32)
    else:
        num_cgm_features = len(cgm_feature_cols) if cgm_feature_cols else 0
        x_cgm_empty = np.empty((0, num_cgm_features), dtype=np.float32)

    return (
        x_cgm_empty,
        np.empty((0, len(other_feature_cols if other_feature_cols else []))),
        empty_np_array,
        empty_context_dict,
        empty_np_array.astype(int)
    )

def _impute_dataframe_features(
    df: pl.DataFrame, 
    cgm_feature_cols: List[str], 
    other_feature_cols: List[str], 
    target_col: str
) -> pl.DataFrame:
    """
    Imputa características faltantes en el DataFrame y añade columnas indicadoras.
    Si una columna de características no existe en df, se añade con 0.0s y se marca como faltante.
    """
    df_imputed = df.clone()
    all_features_to_process = list(set(cgm_feature_cols + other_feature_cols + [target_col])) # Usar set para evitar duplicados

    for col_name in all_features_to_process:
        indicator_col_name = f"{col_name}_is_missing"
        if col_name not in df_imputed.columns:
            # print_warning(f"Columna '{col_name}' no encontrada en DataFrame para imputación. Se creará con ceros y se marcará como faltante.")
            # Determinar el tipo de dato adecuado, por defecto Float64 para características.
            # El target_col podría tener un tipo diferente, pero usualmente es numérico para regresión/DRL.
            default_dtype = pl.Float64
            default_fill_value = 0.0
            
            df_imputed = df_imputed.with_columns(
                pl.lit(default_fill_value).cast(default_dtype).alias(col_name)
            )
            df_imputed = df_imputed.with_columns(
                pl.lit(1).cast(pl.Int8).alias(indicator_col_name) # Marcar todas las filas como faltantes
            )
        else:
            # La columna existe, imputar sus NaNs y crear indicador
            imputed_series, indicator_series = _impute_and_indicate_missing(df_imputed, col_name, indicator_col_name)
            df_imputed = df_imputed.with_columns(imputed_series) # Series ya tienen alias
            df_imputed = df_imputed.with_columns(indicator_series)

    return df_imputed

def _extract_arrays_from_dataframe(
    df_imputed: pl.DataFrame, 
    cgm_feature_cols: List[str], 
    other_feature_cols: List[str], 
    target_col: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extrae arrays NumPy del DataFrame."""
    x_cgm = df_imputed.select(cgm_feature_cols).to_numpy().astype(np.float32) if cgm_feature_cols else np.array([], dtype=np.float32)
    x_other = df_imputed.select(other_feature_cols).to_numpy().astype(np.float32) if other_feature_cols else np.array([], dtype=np.float32)
    y = df_imputed.select(target_col).to_numpy().astype(np.float32).ravel()
    
    if 'SubjectID' in df_imputed.columns:
        subject_ids = df_imputed['SubjectID'].to_numpy().astype(int)
    else:
        print_warning("Columna 'SubjectID' no encontrada. Se retornará un array vacío para subject_ids.")
        subject_ids = np.array([], dtype=int)
    
    return x_cgm, x_other, y, subject_ids

def _reshape_arrays_for_drl(
    x_cgm: np.ndarray, 
    x_other: np.ndarray, 
    cgm_feature_cols: List[str], 
    other_feature_cols: List[str], 
    df_height: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Reformatea arrays para DRL."""
    # Reshape x_cgm
    if x_cgm.ndim == 2 and x_cgm.shape[1] == len(CORE_CGM_FEATURES) and cgm_feature_cols == CORE_CGM_FEATURES:
        x_cgm = x_cgm.reshape(x_cgm.shape[0], len(CORE_CGM_FEATURES), 1)
    elif not cgm_feature_cols:
        x_cgm = np.empty((df_height, 0, 0), dtype=np.float32)

    # Reshape x_other
    if not other_feature_cols:
        x_other = np.empty((df_height, 0), dtype=np.float32)
    elif x_other.ndim == 1 and len(other_feature_cols) == 1:
        x_other = x_other.reshape(-1, 1)
    elif x_other.ndim == 1 and len(other_feature_cols) > 1:
        print_warning(f"x_other es 1D pero hay {len(other_feature_cols)} columnas. Se intentará reshape.")
        if x_other.shape[0] == df_height * len(other_feature_cols):
            x_other = x_other.reshape(df_height, len(other_feature_cols))
        else:
            print_error("No se pudo dar forma a x_other correctamente.")
            x_other = np.empty((df_height, 0), dtype=np.float32)

    return x_cgm, x_other

def prepare_features_for_drl(
    df: pl.DataFrame, 
    cgm_feature_cols: List[str], 
    other_feature_cols: List[str], 
    target_col: str,
    context_feature_order: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """
    Prepara las características de un DataFrame para el entrenamiento DRL.
    Maneja DataFrames vacíos, imputa características, extrae arrays y los remodela.
    """
    if df.is_empty():
        # print_warning("DataFrame de entrada para prepare_features_for_drl está vacío.")
        return _handle_empty_dataframe(cgm_feature_cols, other_feature_cols, context_feature_order)

    # 1. Imputar características y crear indicadores de faltantes
    # Esta función ahora maneja columnas que podrían no estar en df.
    df_imputed = _impute_dataframe_features(df, cgm_feature_cols, other_feature_cols, target_col)

    # 2. Extraer arrays NumPy del DataFrame imputado
    x_cgm, x_other, y_target, subject_ids_np = _extract_arrays_from_dataframe(
        df_imputed, cgm_feature_cols, other_feature_cols, target_col
    )
    
    # 3. Extraer características de contexto del DataFrame imputado (que ahora tiene todas las columnas)
    # _extract_context_for_drl espera que las columnas existan.
    context_features_dict = _extract_context_for_drl(df_imputed, context_feature_order)

    # 4. Remodelar arrays CGM y otros para DRL (ej: (muestras, timesteps, features_cgm))
    # Esta parte asume que x_cgm y x_other tienen el contenido correcto post-imputación.
    x_cgm_reshaped, x_other_reshaped = _reshape_arrays_for_drl(
        x_cgm, x_other, cgm_feature_cols, other_feature_cols, df_imputed.height
    )
    
    # Validar que no haya NaNs después de todo el preprocesamiento
    if np.isnan(x_cgm_reshaped).any() or np.isinf(x_cgm_reshaped).any():
        print_error("NaNs o Infs detectados en x_cgm_reshaped después de la preparación.")
    if np.isnan(x_other_reshaped).any() or np.isinf(x_other_reshaped).any():
        print_error("NaNs o Infs detectados en x_other_reshaped después de la preparación.")
    if np.isnan(y_target).any() or np.isinf(y_target).any():
        print_error("NaNs o Infs detectados en y_target después de la preparación.")
    for key, arr in context_features_dict.items():
        if np.isnan(arr).any() or np.isinf(arr).any():
            print_error(f"NaNs o Infs detectados en context_feature '{key}' después de la preparación.")

    return x_cgm_reshaped, x_other_reshaped, y_target, context_features_dict, subject_ids_np

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

def _extract_glucose_feature(df: pl.DataFrame, glucose_fallback_value: float) -> np.ndarray:
    """Extrae y procesa la característica de glucosa actual."""
    if 'glucose_last' in df.columns:
        return df['glucose_last'].fill_null(strategy="forward").fill_null(strategy="backward").fill_null(glucose_fallback_value).to_numpy()
    else:
        print_warning("Columna 'glucose_last' no encontrada para 'current_glucose' en _extract_context_for_drl. Usando ceros.")
        return np.full(len(df), glucose_fallback_value)

def _extract_single_feature(df: pl.DataFrame, column_name: str, default_value: float = 0.0) -> np.ndarray:
    """
    Extrae una columna de características como un array NumPy, imputando NaNs con default_value.
    Si la columna no existe, retorna un array de default_value.
    """
    if column_name in df.columns:
        feature_array = df.select(pl.col(column_name).fill_null(default_value)).to_numpy().flatten()
    else:
        print_warning(f"Columna '{column_name}' no encontrada en el DataFrame. Usando valor por defecto {default_value}.")
        feature_array = np.full(len(df), default_value, dtype=float)
    return feature_array.astype(np.float32)

def _extract_contextual_features(df: pl.DataFrame) -> Dict[str, np.ndarray]:
    """Extrae características contextuales (sleep, work, exercise)."""
    context_data = {}
    contextual_feature_names = ['sleep_quality', 'work_intensity', 'exercise_intensity']
    
    for feature_name in contextual_feature_names:
        context_data[feature_name] = _extract_single_feature(df, feature_name, feature_name)
    
    return context_data

def _ensure_missing_features(context_data: Dict[str, np.ndarray], df: pl.DataFrame, glucose_fallback_value: float) -> None:
    """Asegura que todas las características requeridas estén presentes."""
    for key in CONTEXT_FEATURE_ORDER:
        if key not in context_data:
            if key == 'current_glucose':
                print_warning(f"Característica de contexto DRL '{key}' no fue explícitamente extraída y falta. Se añadirá con {glucose_fallback_value}.")
                context_data[key] = np.full(len(df), glucose_fallback_value)
            else:
                print_warning(f"Característica de contexto DRL '{key}' no fue explícitamente extraída y falta. Se añadirá con ceros.")
                context_data[key] = np.zeros(len(df))

def _validate_feature_lengths(context_data: Dict[str, np.ndarray], df: pl.DataFrame, glucose_fallback_value: float) -> None:
    """Valida que todas las características tengan la longitud correcta."""
    for key in context_data:
        if len(context_data[key]) != len(df):
            print_error(f"Error de longitud en la característica de contexto DRL '{key}'. Esperado: {len(df)}, Obtenido: {len(context_data[key])}. Se reemplazará con ceros (o fallback para glucosa).")
            if key == 'current_glucose':
                context_data[key] = np.full(len(df), glucose_fallback_value)
            else:
                context_data[key] = np.zeros(len(df))

def _extract_context_for_drl(df: pl.DataFrame, context_feature_order: List[str]) -> Dict[str, np.ndarray]:
    """
    Extrae características de contexto de un DataFrame y las devuelve como un diccionario de arrays NumPy.
    Asegura que todas las columnas en context_feature_order existan, imputando con 0.0 si es necesario.
    """
    context_data: Dict[str, np.ndarray] = {}
    df_height = df.height
    
    # Clonar para evitar modificar el df original si se añaden columnas temporalmente
    temp_df = df.clone()

    for feature_name in context_feature_order:
        if feature_name not in temp_df.columns:
            # print_warning(f"Característica de contexto '{feature_name}' no encontrada. Se imputará con 0.0.")
            # Añadir la columna con 0.0s si no existe
            temp_df = temp_df.with_columns(pl.lit(0.0).cast(pl.Float64).alias(feature_name))
        
        # Imputar NaNs restantes en la columna de contexto (si existía y tenía NaNs, o si se acaba de añadir)
        # Usar una imputación simple de 0.0 para el contexto si hay nulos.
        # Esto es crucial porque CONTEXT_FEATURE_ORDER puede incluir columnas que son parte de other_features
        # y ya fueron imputadas por _impute_dataframe_features.
        # Sin embargo, si una columna de CONTEXT_FEATURE_ORDER no estaba en other_features, necesita imputación aquí.
        if temp_df[feature_name].is_null().any():
             temp_df = temp_df.with_columns(
                 temp_df[feature_name].fill_null(0.0).alias(feature_name)
             )
        context_data[feature_name] = temp_df[feature_name].to_numpy()

    # Validar longitudes (opcional, pero bueno para la robustez)
    for feature_name, arr in context_data.items():
        if len(arr) != df_height:
            print_error(f"Error de longitud para la característica de contexto '{feature_name}'. Esperado: {df_height}, Obtenido: {len(arr)}")
            # Manejar el error, ej: rellenar o lanzar excepción
            context_data[feature_name] = np.full(df_height, 0.0) # Fallback

    return context_data

def prepare_data_for_drl_training(df: pl.DataFrame) -> Tuple[
    # Train
    np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray,
    # Validation
    np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray,
    # Test
    np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray
]:
    """
    Prepara los datos completos para el entrenamiento DRL, dividiéndolos en conjuntos
    de entrenamiento, validación y prueba, y extrayendo todas las características necesarias.
    """
    coloured("Iniciando preparación de datos para DRL...", 'magenta')

    # 1. Definir características CGM, otras y objetivo
    feature_groups = get_feature_groups()
    cgm_feature_cols = feature_groups['cgm_features']
    other_feature_cols = feature_groups['other_features'] # Usar todas las otras características definidas
    target_col = feature_groups['target_variable'] # ej: 'bolus_log1p' o 'bolus'

    # Validar disponibilidad de características
    available_cgm, missing_cgm = validate_features_availability(df.columns, cgm_feature_cols)
    available_other, missing_other = validate_features_availability(df.columns, other_feature_cols)

    if missing_cgm:
        print_warning(f"Características CGM faltantes: {missing_cgm}. Se usarán las disponibles: {available_cgm}")
    if missing_other:
        print_warning(f"Otras características faltantes: {missing_other}. Se usarán las disponibles: {available_other}")
    if target_col not in df.columns:
        print_error(f"Columna objetivo '{target_col}' no encontrada en el DataFrame. No se puede continuar.")
        raise ValueError(f"Columna objetivo '{target_col}' faltante.")

    cgm_feature_cols = available_cgm
    other_feature_cols = available_other
    
    # Filtrar columnas que no existen del DataFrame antes de la división
    existing_cols_for_split = [col for col in df.columns if col in cgm_feature_cols or col in other_feature_cols or col == target_col or col == SUBJECT_ID_COL or col == TIMESTAMP_COL]
    # Añadir columnas de contexto directamente si no están en other_feature_cols pero sí en CONTEXT_FEATURE_ORDER
    for _ctx_key in CONTEXT_FEATURE_ORDER:
        # El mapeo de feature_map en _extract_context_for_drl usa los nombres de columna reales
        # Aquí necesitamos asegurar que esas columnas estén en df_filtered_for_split
        # Esto es un poco redundante si other_feature_cols ya las incluye.
        # Por simplicidad, asumimos que las columnas de contexto están en other_feature_cols o se manejarán.
        pass


    df_filtered_for_split = df.select(existing_cols_for_split)


    # 2. Dividir datos temporalmente
    df_train, df_val, df_test = create_temporal_splits(df_filtered_for_split)
    coloured(f"Datos divididos: Train {df_train.shape}, Val {df_val.shape}, Test {df_test.shape}", 'green')

    data_splits = {}
    for split_name, split_df in zip(['train', 'val', 'test'], [df_train, df_val, df_test]):
        coloured(f"Procesando split: {split_name}", 'yellow')
        if split_df.is_empty():
            print_warning(f"El split '{split_name}' está vacío. Se generarán arrays vacíos.")
            num_cgm_feats = len(cgm_feature_cols) if cgm_feature_cols else 1 # Evitar 0 si no hay cgm_cols
            num_other_feats = len(other_feature_cols) if other_feature_cols else 0

            x_cgm = np.empty((0, CONFIG_PROCESSING.get("window_steps", 24), num_cgm_feats if num_cgm_feats > 0 else 1)) # (0, timesteps, features)
            x_other = np.empty((0, num_other_feats if num_other_feats > 0 else 1)) # (0, features)
            y = np.empty((0,))
            context_data = {feat: np.empty((0,)) for feat in CONTEXT_FEATURE_ORDER}
            subject_id = np.empty((0,))
        else:
            x_cgm, x_other, y, context_data, subject_id = prepare_features_for_drl(
                split_df, cgm_feature_cols, other_feature_cols, target_col, CONTEXT_FEATURE_ORDER
            )
        data_splits[split_name] = (x_cgm, x_other, y, context_data, subject_id)
        coloured(f"  {split_name} - CGM: {x_cgm.shape}, Otros: {x_other.shape}, Target: {y.shape}, SubjectID: {subject_id.shape}", "cyan")
        if context_data:
            for k, v_arr in context_data.items():
                coloured(f"    Contexto {k}: {v_arr.shape}", "magenta")


    return (*data_splits['train'], *data_splits['val'], *data_splits['test'])

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
