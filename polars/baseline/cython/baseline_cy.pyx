# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, fabs
from cpython.datetime cimport datetime, timedelta
import polars as pl
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Define C types for numpy arrays
DTYPE = np.float32
ITYPE = np.int64
ctypedef np.float32_t DTYPE_t
ctypedef np.int64_t ITYPE_t

cpdef np.ndarray[DTYPE_t, ndim=1] get_cgm_window(datetime bolus_time, object cgm_df, int window_hours=2):
    """
    Get CGM window for a specific bolus time.
    """
    cdef:
        datetime window_start
        np.ndarray[DTYPE_t, ndim=1] window_data
    
    window_start = bolus_time - timedelta(hours=window_hours)
    window = cgm_df.filter(
        (pl.col("date") >= window_start) & (pl.col("date") <= bolus_time)
    ).sort("date").tail(24)
    
    if window.height < 24:
        return None
    
    window_data = window.get_column("mg/dl").to_numpy(dtype=DTYPE)
    return window_data

cpdef double calculate_iob(datetime bolus_time, object basal_df, double half_life_hours=4.0):
    """
    Calculate insulin on board (IOB) for a specific bolus time.
    """
    cdef:
        double iob = 0.0
        double duration_hours, time_since_start, remaining, rate
        datetime start_time, end_time
    
    if basal_df is None or basal_df.is_empty():
        return 0.0
    
    for row in basal_df.iter_rows(named=True):
        start_time = row["date"]
        duration_hours = row["duration"] / (1000.0 * 3600.0)
        end_time = start_time + timedelta(hours=duration_hours)
        rate = row["rate"] if row["rate"] is not None else 0.9
        
        if start_time <= bolus_time <= end_time:
            time_since_start = (bolus_time - start_time).total_seconds() / 3600.0
            remaining = rate * (1.0 - (time_since_start / half_life_hours))
            iob += max(0.0, remaining)
    return iob

cpdef list process_subject(str subject_path, int idx):
    """
    Process data for a specific subject.
    """
    cdef:
        list processed_data = []
        double iob, carb_input, bg_input, insulin_carb_ratio
        datetime bolus_time
        np.ndarray[DTYPE_t, ndim=1] cgm_window
    
    print(f"Processing {subject_path.split('/')[-1]} ({idx+1})")
    
    try:
        cgm_df = pl.read_excel(subject_path, sheet_name="CGM")
        bolus_df = pl.read_excel(subject_path, sheet_name="Bolus")
        try:
            basal_df = pl.read_excel(subject_path, sheet_name="Basal")
        except Exception:
            basal_df = None
            
    except Exception as e:
        print(f"Error loading {subject_path.split('/')[-1]}: {e}")
        return []

    cgm_df = cgm_df.with_columns(pl.col("date").cast(pl.Datetime))
    bolus_df = bolus_df.with_columns(pl.col("date").cast(pl.Datetime))
    if basal_df is not None:
        basal_df = basal_df.with_columns(pl.col("date").cast(pl.Datetime))
    
    cgm_df = cgm_df.sort("date")

    for row in bolus_df.iter_rows(named=True):
        bolus_time = row["date"]
        cgm_window = get_cgm_window(bolus_time, cgm_df)
        
        if cgm_window is not None:
            iob = calculate_iob(bolus_time, basal_df)
            carb_input = row["carbInput"] if row["carbInput"] is not None else 0.0
            bg_input = row["bgInput"] if row["bgInput"] is not None else cgm_window[-1]
            insulin_carb_ratio = row["insulinCarbRatio"] if row["insulinCarbRatio"] is not None else 10.0
            
            features = {
                'subject_id': idx,
                'cgm_window': cgm_window,
                'carbInput': carb_input,
                'bgInput': bg_input,
                'insulinCarbRatio': insulin_carb_ratio,
                'insulinSensitivityFactor': 50.0,
                'insulinOnBoard': iob,
                'normal': row["normal"]
            }
            processed_data.append(features)
    
    return processed_data

def rule_based_prediction(
    np.ndarray[DTYPE_t, ndim=2] X,
    double target_bg=100.0,
    object scaler_other=None
):
    """
    Rule-based prediction of insulin doses.
    """
    cdef:
        Py_ssize_t i, n_samples = X.shape[0]
        np.ndarray[DTYPE_t, ndim=1] result
        np.ndarray[DTYPE_t, ndim=2] transformed_data
        double carb_input, bg_input, icr, isf
    
    transformed_data = scaler_other.inverse_transform(X[:, 24:27])
    result = np.empty(n_samples, dtype=DTYPE)
    
    for i in range(n_samples):
        carb_input = transformed_data[i, 0]
        bg_input = transformed_data[i, 1]
        icr = X[i, 27]
        isf = X[i, 28]
        result[i] = carb_input / icr + (bg_input - target_bg) / isf
    
    return result

def calculate_rmse(
    np.ndarray[DTYPE_t, ndim=1] y_true,
    np.ndarray[DTYPE_t, ndim=1] y_pred
):
    """
    Calculate Root Mean Square Error (RMSE).
    """
    cdef:
        Py_ssize_t i, n = y_true.shape[0]
        double sum_squared_error = 0.0
        double diff
    
    for i in range(n):
        diff = y_true[i] - y_pred[i]
        sum_squared_error += diff * diff
    
    return sqrt(sum_squared_error / n)

def calculate_mae(
    np.ndarray[DTYPE_t, ndim=1] y_true,
    np.ndarray[DTYPE_t, ndim=1] y_pred
):
    """
    Calculate Mean Absolute Error (MAE).
    """
    cdef:
        Py_ssize_t i, n = y_true.shape[0]
        double sum_absolute_error = 0.0
        double diff
    
    for i in range(n):
        diff = y_true[i] - y_pred[i]
        sum_absolute_error += fabs(diff)
    
    return sum_absolute_error / n