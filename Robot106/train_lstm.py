#!/usr/bin/env python3
"""
train_lstm.py

Loads the collected training data (GPS + IMU + commands) from CSV,
trains an LSTM regression model to predict continuous [steering, throttle],
and saves the model to disk.

Includes a demonstration of how to use exponential smoothing + rounding
to achieve smooth, discrete commands at runtime.
"""

import csv
import os
import numpy as np
import argparse
import math

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ---- Configuration ----
DATA_LOG_PATH = "training_data.csv"
MODEL_SAVE_PATH = "lstm_navigation_model.h5"
SEQUENCE_LENGTH = 10   # how many timesteps per input sequence
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3

def load_data_for_training(csv_path=DATA_LOG_PATH, seq_length=SEQUENCE_LENGTH):
    """
    Loads the CSV file, creates overlapping sequences.
    Input Features: [gps_lat, gps_lon, imu_heading, accel_x, accel_y]
    Output Labels: [steering_cmd, throttle_cmd] (continuous but originally 0..126)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    data_rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data_rows.append(row)

    features = []
    labels = []
    for row in data_rows:
        gps_lat = float(row["gps_lat"])
        gps_lon = float(row["gps_lon"])
        imu_heading = float(row["imu_heading"])
        accel_x = float(row["accel_x"])
        accel_y = float(row["accel_y"])
        steering = float(row["steering_cmd"])
        throttle = float(row["throttle_cmd"])
        # Input
        features.append([gps_lat, gps_lon, imu_heading, accel_x, accel_y])
        # Output
        labels.append([steering, throttle])

    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    print("[Data] Raw shapes:", features.shape, labels.shape)

    # Optionally scale features here if desired:
    # e.g. features = scaler.transform(features)

    # Build sequences
    X_seq, y_seq = [], []
    for i in range(len(features) - seq_length):
        X_seq.append(features[i : i + seq_length])
        # We predict the command at the LAST step of the sequence
        y_seq.append(labels[i + seq_length - 1])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    print("[Data] Sequence shapes:", X_seq.shape, y_seq.shape)
    # X_seq: (num_samples, seq_length, num_features)
    # y_seq: (num_samples, 2)
    return X_seq, y_seq

def build_lstm_model(seq_length, num_features, lr=1e-3):
    """
    Builds a regression LSTM to output [steering, throttle].
    """
    model = Sequential()
    model.add(LSTM(64, input_shape=(seq_length, num_features), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='linear'))  # 2 outputs: steering, throttle
    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mse')
    model.summary()
    return model

def train_model(csv_path=DATA_LOG_PATH, model_path=MODEL_SAVE_PATH,
                seq_length=SEQUENCE_LENGTH, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE):
    """
    Main training routine: load data, build model, train, save model.
    """
    X, y = load_data_for_training(csv_path, seq_length)
    num_features = X.shape[2]

    model = build_lstm_model(seq_length, num_features, lr=lr)
    history = model.fit(
        X, y,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    model.save(model_path)
    print(f"[Training] Model saved to {model_path}")
    return model

# ============ Example Post-Processing for Smooth, Discrete Outputs ============

def exponential_smoothing(prev_cmd, new_cmd, alpha=0.3):
    """
    Smoothly blend the new command with the previous command.
    prev_cmd, new_cmd are each [steering, throttle].
    alpha in [0..1] controls smoothing (0=all old, 1=all new).
    """
    return alpha * np.array(new_cmd) + (1 - alpha) * np.array(prev_cmd)

def clamp_to_robot_range(cmd):
    """
    Ensures steering/throttle are in [0..126], then round to nearest integer.
    """
    steering, throttle = cmd
    steering_int = int(round(max(0, min(126, steering))))
    throttle_int = int(round(max(0, min(126, throttle))))
    return [steering_int, throttle_int]

def smooth_and_snap_command(prev_cmd, raw_prediction, alpha=0.3):
    """
    Combines exponential smoothing with integer snapping.
    prev_cmd: last integer command [steering, throttle]
    raw_prediction: new continuous output from LSTM
    returns: [steering_int, throttle_int] (smoothed & clamped)
    """
    # 1) Smooth the new prediction with the old command (to reduce jumpiness)
    smoothed = exponential_smoothing(prev_cmd, raw_prediction, alpha=alpha)
    # 2) Snap to [0..126] integer
    return clamp_to_robot_range(smoothed)

# ============ Optional: A small demonstration of using the model ============

def demo_inference(model_path=MODEL_SAVE_PATH, seq_length=SEQUENCE_LENGTH):
    """
    Show how you'd load a model, maintain a sliding window of sensor data,
    and produce smoothed, discrete commands.
    In a real robot script, you replace the random sensor data with real GPS/IMU reads.
    """
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)

    # Fake sensor window
    sensor_window = []
    # Start with a neutral command
    prev_cmd = [64, 64]

    # Suppose we run 20 timesteps
    for t in range(20):
        # In reality, read from IMU & GPS here:
        # (lat, lon, heading, ax, ay)
        fake_lat = 0.001 * t
        fake_lon = -0.001 * t
        fake_heading = (t * 5) % 360
        fake_ax = 0.0
        fake_ay = 0.0
        current_features = [fake_lat, fake_lon, fake_heading, fake_ax, fake_ay]

        sensor_window.append(current_features)
        if len(sensor_window) > seq_length:
            sensor_window.pop(0)

        # If we don't yet have enough to fill the window, skip
        if len(sensor_window) < seq_length:
            print(f"t={t}, Not enough data for a prediction yet.")
            continue

        # shape = (1, seq_length, num_features)
        input_batch = np.array(sensor_window[-seq_length:]).reshape(1, seq_length, 5)
        raw_prediction = model.predict(input_batch)[0]  # shape = (2,)

        # Post-process to get a smooth, discrete command
        new_cmd = smooth_and_snap_command(prev_cmd, raw_prediction, alpha=0.3)
        print(f"t={t}, raw={raw_prediction}, final={new_cmd}")

        # Update prev_cmd
        prev_cmd = new_cmd
        # Then you'd publish to your robot here, e.g. MQTT.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM for robot command regression.")
    parser.add_argument("--csv", type=str, default=DATA_LOG_PATH, help="Path to CSV data file.")
    parser.add_argument("--model", type=str, default=MODEL_SAVE_PATH, help="Path to save trained model.")
    parser.add_argument("--seq_length", type=int, default=SEQUENCE_LENGTH, help="Input sequence length.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size.")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate.")
    parser.add_argument("--demo", action="store_true", help="Run a short inference demo after training.")
    args = parser.parse_args()

    # Train the model
    model = train_model(
        csv_path=args.csv,
        model_path=args.model,
        seq_length=args.seq_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )

    # Optionally run a small inference demo
    if args.demo:
        demo_inference(model_path=args.model, seq_length=args.seq_length)
