#!/usr/bin/env python3
"""
train_lstm.py

Train a robust LSTM model to predict next-step steering/throttle commands
from CSV data with potential large time gaps.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import datetime
import math

# ============================================================================
# 1. CONFIG
# ============================================================================
CSV_FILE = "training_data.csv"   
TIME_GAP_THRESHOLD = 5.0            # in seconds, define what "large gap" means for your data
SEQUENCE_LENGTH = 10                # number of consecutive time steps per training sample
TRAIN_SPLIT_RATIO = 0.8             # 80% training, 20% testing
EPOCHS = 50
BATCH_SIZE = 32

# ============================================================================
# 2. LOAD AND PREPROCESS THE CSV
# ============================================================================
# Assume columns: [timestamp, gps_lat, gps_lon, imu_heading, accel_x, accel_y, steering_cmd, throttle_cmd]
df = pd.read_csv(CSV_FILE)

# Convert float timestamp -> datetime
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)

# Sort by ascending time
df = df.sort_values(by='datetime').reset_index(drop=True)

# Compute time since start (in seconds)
start_time = df['datetime'].iloc[0]
df['time_since_start_s'] = (df['datetime'] - start_time).dt.total_seconds()

# Compute time delta between consecutive rows to detect large gaps
df['time_diff'] = df['time_since_start_s'].diff().fillna(0)

# Drop original columns we won't directly feed to LSTM
# (We'll keep 'timestamp' only if needed for debugging.)
df = df.drop(columns=['datetime'])

# Optional: handle missing values or noisy data
df = df.dropna()

# ============================================================================
# 3. SET UP FEATURE & TARGET COLUMNS
# ============================================================================
# Include time_since_start_s as a numeric feature to help the model
feature_cols = [
    'gps_lat',
    'gps_lon',
    'imu_heading',
    'accel_x',
    'accel_y',
    'steering_cmd',
    'throttle_cmd',
    'time_since_start_s'
]

target_cols = ['steering_cmd', 'throttle_cmd']

# Verify we have them all
for col in feature_cols:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in the DataFrame.")

# ============================================================================
# 4. SCALING
# ============================================================================
data = df[feature_cols].values

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# For convenience, let's store it back in a DataFrame
df_scaled = pd.DataFrame(data_scaled, columns=feature_cols)
df_scaled['time_diff'] = df['time_diff'].values  # keep real time_diff (unscaled) to detect gaps

# ============================================================================
# 5. CREATE SEQUENCES, BREAKING ON LARGE TIME GAPS
# ============================================================================
def create_sequences_with_gaps(df_scaled, seq_len=10, gap_threshold=5.0):
    """
    Creates sequences from df_scaled, stopping a sequence when a large
    time gap is detected. Returns X (samples) and y (targets).
    """
    X_list, y_list = [], []
    current_sequence = []

    for i in range(len(df_scaled)):
        # Always append current row to current_sequence
        current_sequence.append(df_scaled.iloc[i].to_dict())

        if i == len(df_scaled) - 1:
            # If we're at the last row, we can't form a new sample after this
            continue

        # Check the next row's time_diff to see if it's a large gap
        next_gap = df_scaled.iloc[i+1]['time_diff']

        # If we have at least seq_len rows, we can form a sample
        if len(current_sequence) >= seq_len:
            # The target is the row *after* this sequence
            # but only if we won't go out of bounds
            if i + 1 < len(df_scaled):
                # Build X from the last seq_len rows in current_sequence
                seq_x = current_sequence[-seq_len:]

                # Next row is the target, specifically columns 5 & 6
                # (assuming feature_cols order: index 5=steering_cmd, 6=throttle_cmd)
                next_row = df_scaled.iloc[i+1]
                seq_y = [next_row['steering_cmd'], next_row['throttle_cmd']]

                # Convert these to arrays
                x_array = np.array([[row[col] for col in feature_cols] for row in seq_x])
                y_array = np.array(seq_y)

                X_list.append(x_array)
                y_list.append(y_array)

        # If the next gap is beyond threshold, break the sequence
        if next_gap > gap_threshold:
            current_sequence = []  # start a new sequence

    X_array = np.array(X_list)
    y_array = np.array(y_list)
    return X_array, y_array

X, y = create_sequences_with_gaps(df_scaled, seq_len=SEQUENCE_LENGTH, gap_threshold=TIME_GAP_THRESHOLD)

print("Created sequences:")
print("X shape =", X.shape)  # (num_samples, seq_len, num_features)
print("y shape =", y.shape)  # (num_samples, 2)

if len(X) == 0:
    raise ValueError("No sequences were created. Adjust your TIME_GAP_THRESHOLD or SEQUENCE_LENGTH.")

# ============================================================================
# 6. TRAIN/TEST SPLIT
# ============================================================================
train_size = int(len(X) * TRAIN_SPLIT_RATIO)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Train set size: {X_train.shape[0]}")
print(f"Test set size:  {X_test.shape[0]}")

# ============================================================================
# 7. BUILD THE LSTM MODEL
# ============================================================================
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, len(feature_cols))))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(2))  # steering_cmd, throttle_cmd

model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

# ============================================================================
# 8. TRAIN
# ============================================================================
history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    shuffle=False  # keep time order
)

# ============================================================================
# 9. EVALUATE
# ============================================================================
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Test MSE: {test_loss:.6f}")

# ============================================================================
# 10. MAKE PREDICTIONS & INVERSE TRANSFORM
# ============================================================================
y_pred = model.predict(X_test)

# We only predicted steering_cmd, throttle_cmd
# We must reconstruct those columns in order to inverse_transform with the same scaler
num_samples = len(y_pred)
num_features = len(feature_cols)
reconstruct = np.zeros((num_samples, num_features))

# Insert predictions into the correct positions
# Indices in feature_cols: 0=gps_lat,1=gps_lon,2=imu_heading,3=accel_x,4=accel_y,5=steering_cmd,6=throttle_cmd,7=time_since_start_s
steering_idx = 5
throttle_idx = 6

reconstruct[:, steering_idx] = y_pred[:, 0]
reconstruct[:, throttle_idx] = y_pred[:, 1]

# We'll fill the other columns with something valid (e.g. 0.0) because
# MinMaxScaler expects the same shape, but it won't affect the inverse transform
# for the columns we actually care about (steering & throttle).
# Inverse transform
inversed = scaler.inverse_transform(reconstruct)
steering_pred = inversed[:, steering_idx]
throttle_pred = inversed[:, throttle_idx]

# For a direct comparison, also inverse-transform y_test
reconstruct_true = np.zeros((len(y_test), num_features))
reconstruct_true[:, steering_idx] = y_test[:, 0]
reconstruct_true[:, throttle_idx] = y_test[:, 1]
inversed_true = scaler.inverse_transform(reconstruct_true)
steering_true = inversed_true[:, steering_idx]
throttle_true = inversed_true[:, throttle_idx]

# Print sample results
print("\nSample predictions vs. ground truth:")
for i in range(min(5, len(steering_true))):
    print(f"Predicted (steering, throttle) = ({steering_pred[i]:.2f}, {throttle_pred[i]:.2f})  "
          f"True = ({steering_true[i]:.2f}, {throttle_true[i]:.2f})")

# ============================================================================
# 11. SAVE MODEL (OPTIONAL)
# ============================================================================
model.save("lstm_steering_throttle.h5")
print("\nSaved model to 'lstm_steering_throttle.h5'")
