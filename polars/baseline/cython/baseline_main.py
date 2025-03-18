import os
import polars as pl
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Import Cython functions
from baseline_cy import (
    process_subject,
    rule_based_prediction,
    calculate_rmse,
    calculate_mae
)

# Configure matplotlib
import matplotlib
matplotlib.use('TkAgg')

# Get subject files
subject_folder = os.path.join(os.getcwd(), "subjects")
subject_files = [f for f in os.listdir(subject_folder) if f.startswith("Subject") and f.endswith(".xlsx")]
print(f"Total subjects: {len(subject_files)}")

# Parallel processing using Cython implementation
all_processed_data = Parallel(n_jobs=-1)(
    delayed(process_subject)(
        os.path.join(subject_folder, f), 
        idx
    ) for idx, f in enumerate(subject_files)
)

# Flatten list of lists
all_processed_data = [item for sublist in all_processed_data for item in sublist]

# Convert to Polars DataFrame
df_processed = pl.DataFrame(all_processed_data)
print("Sample of combined processed data:")
print(df_processed.head())
print(f"Total samples: {len(df_processed)}")

# Split CGM window into columns
cgm_columns = [f'cgm_{i}' for i in range(24)]
df_cgm = pl.DataFrame({
    col: [row['cgm_window'][i] for row in all_processed_data]
    for i, col in enumerate(cgm_columns)
})

# Combine with other features
df_final = pl.concat([
    df_cgm,
    df_processed.drop('cgm_window')
], how="horizontal")

# Drop rows with null values
df_final = df_final.drop_nulls()

# Normalize features
scaler_cgm = MinMaxScaler(feature_range=(0, 1))
scaler_other = StandardScaler()

# Convert to numpy arrays for sklearn
X_cgm = scaler_cgm.fit_transform(df_final.select(cgm_columns).to_numpy())
X_other = scaler_other.fit_transform(
    df_final.select(['carbInput', 'bgInput', 'insulinOnBoard']).to_numpy()
)

# Combine features
X = np.hstack([
    X_cgm, 
    X_other, 
    df_final.select(['insulinCarbRatio', 'insulinSensitivityFactor', 'subject_id']).to_numpy()
])
y = df_final.get_column('normal').to_numpy()

# Convert to numpy arrays with correct dtype
X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.float32)

# Split data by subject
subject_ids = df_final.get_column('subject_id').unique().to_numpy()
train_subjects, temp_subjects = train_test_split(subject_ids, test_size=0.2, random_state=42)
val_subjects, test_subjects = train_test_split(temp_subjects, test_size=0.5, random_state=42)

train_mask = np.isin(df_final.get_column('subject_id').to_numpy(), train_subjects)
val_mask = np.isin(df_final.get_column('subject_id').to_numpy(), val_subjects)
test_mask = np.isin(df_final.get_column('subject_id').to_numpy(), test_subjects)

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]
subject_test = df_final.filter(pl.lit(test_mask)).get_column('subject_id').to_numpy()

# Define and train model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='linear')
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
    verbose=1
)

# Evaluate model
y_pred = model.predict(X_test).flatten()
y_rule = rule_based_prediction(X_test, 100.0, scaler_other)

# Calculate metrics using Cython functions
mae = calculate_mae(y_test, y_pred)
rmse = calculate_rmse(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

mae_rule = calculate_mae(y_test, y_rule)
rmse_rule = calculate_rmse(y_test, y_rule)
r2_rule = r2_score(y_test, y_rule)

print("\nModel Performance:")
print(f"Neural Network - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")
print(f"Rule-based    - MAE: {mae_rule:.2f}, RMSE: {rmse_rule:.2f}, R²: {r2_rule:.2f}")

# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Training History')

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, label='NN Predictions', alpha=0.5)
plt.scatter(y_test, y_rule, label='Rule-based', alpha=0.5)
plt.plot([0, 15], [0, 15], 'r--')
plt.xlabel('Actual Dose (units)')
plt.ylabel('Predicted Dose (units)')
plt.legend()
plt.title('Predictions vs Actual')
plt.tight_layout()
plt.show()