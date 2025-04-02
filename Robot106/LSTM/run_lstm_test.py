#!/usr/bin/env python3
"""
run_lstm_test.py
"""

import time
import math
import numpy as np
import paho.mqtt.client as mqtt

# TensorFlow / Keras imports
from tensorflow.keras.models import load_model

# -----------------------------------------------------------------------------
# 1. MQTT CONFIG & CONNECTION
# -----------------------------------------------------------------------------
MQTT_SERVER = "192.168.1.120"   # Change if needed
MQTT_PORT = 1883
MQTT_TOPIC_COMMAND = "robot/control"

client = mqtt.Client()

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"Connected to MQTT server at {MQTT_SERVER}:{MQTT_PORT}")
    else:
        print(f"Failed to connect, return code {rc}")

client.on_connect = on_connect
client.connect(MQTT_SERVER, MQTT_PORT, 60)
client.loop_start()

# -----------------------------------------------------------------------------
# 2. LOAD LSTM MODEL (NO SCALER FILE)
# -----------------------------------------------------------------------------
MODEL_PATH = "lstm_steering_throttle.h5"  # <-- Your trained model file

lstm_model = load_model(MODEL_PATH)

# -----------------------------------------------------------------------------
# 3. PARAMETERS
# -----------------------------------------------------------------------------
# This script assumes your LSTM expects raw features such as:
#   [gps_lat, gps_lon, imu_heading, accel_x, accel_y, steering_cmd, throttle_cmd, time_since_start_s]
# (Adjust as needed. If your model uses fewer or different features, adapt accordingly.)

FEATURE_COLS = [
    "gps_lat",
    "gps_lon",
    "imu_heading",
    "accel_x",
    "accel_y",
    "steering_cmd",
    "throttle_cmd",
    "time_since_start_s"
]

SEQUENCE_LENGTH = 10  # same window length used in training

# Circle path parameters (example):
lat0, lon0 = 40.0, -74.0
radius_deg = 0.0001  # ~11 m if 1 deg ~ 111 km
NUM_POINTS = 50

# Movement speeds (placeholders – adapt to your robot's range):
FORWARD_THROTTLE_CMD = 100
BACKWARD_THROTTLE_CMD = 30
STRAIGHT_STEERING_CMD = 64  # "neutral" steering, if commands are in [0..128]

# -----------------------------------------------------------------------------
# 4. GENERATE A CIRCLE OF WAYPOINTS
# -----------------------------------------------------------------------------
waypoints = []
for i in range(NUM_POINTS):
    theta = 2 * math.pi * i / NUM_POINTS
    wlat = lat0 + radius_deg * math.cos(theta)
    wlon = lon0 + radius_deg * math.sin(theta)
    waypoints.append((wlat, wlon))

waypoints_forward = waypoints
waypoints_backward = list(reversed(waypoints))

# -----------------------------------------------------------------------------
# 5. UTILITY: BUILD A ROLLING WINDOW OF INPUT DATA
# -----------------------------------------------------------------------------
start_time = time.time()

def get_lstm_input(last_sequence, next_waypoint, last_steer, last_throttle):
    """
    Construct a new row of raw input features. 
    We'll pretend we know heading by looking at the circle geometry,
    and we set accel_x/y = 0 for demonstration.

    We then produce a full sequence of length SEQUENCE_LENGTH by combining
    this row with the (SEQUENCE_LENGTH - 1) previous rows in `last_sequence`.
    """
    lat, lon = next_waypoint
    dx = lat - lat0
    dy = lon - lon0
    heading = math.degrees(math.atan2(dy, dx)) % 360  # synthetic heading

    accel_x = 0.0
    accel_y = 0.0

    current_t = time.time() - start_time

    # Construct the new row in the same order as FEATURE_COLS
    row = [
        lat,               # gps_lat
        lon,               # gps_lon
        heading,           # imu_heading
        accel_x,           # accel_y
        accel_y,           # accel_y
        last_steer,        # steering_cmd (raw)
        last_throttle,     # throttle_cmd (raw)
        current_t          # time_since_start_s
    ]

    # If we don't yet have SEQUENCE_LENGTH-1 rows, pad them
    if len(last_sequence) < SEQUENCE_LENGTH - 1:
        while len(last_sequence) < SEQUENCE_LENGTH - 1:
            last_sequence.append(row)

    # Now build the final sequence
    if len(last_sequence) == SEQUENCE_LENGTH - 1:
        sequence = last_sequence + [row]
    else:
        sequence = last_sequence[-(SEQUENCE_LENGTH - 1):] + [row]

    return sequence

# -----------------------------------------------------------------------------
# 6. LOOP THROUGH WAYPOINTS, PREDICT STEERING/THROTTLE -> MOVE ROBOT
# -----------------------------------------------------------------------------
def move_circle(waypoints, go_forward=True):
    """
    Given a list of waypoints, repeatedly:
      - Build an input sequence for the LSTM
      - Predict next steering/throttle
      - Publish the command
    If go_forward=False, we forcibly set a smaller throttle to mimic reverse.
    """
    print(f"\nStarting circle movement. Forward={go_forward}")
    last_sequence = []
    last_steer = STRAIGHT_STEERING_CMD
    last_throttle = FORWARD_THROTTLE_CMD if go_forward else BACKWARD_THROTTLE_CMD

    for i, wp in enumerate(waypoints):
        # 1) Construct input sequence
        last_sequence = get_lstm_input(last_sequence, wp, last_steer, last_throttle)

        # 2) Convert to numpy and feed to the LSTM
        seq_array = np.array(last_sequence)  # shape (seq_len, 8)
        seq_array = np.expand_dims(seq_array, axis=0)  # (1, seq_len, 8)

        # 3) Predict
        y_pred = lstm_model.predict(seq_array)
        predicted_steer, predicted_throttle = y_pred[0]  # shape (2,)

        # 4) If we want forward or backward movement specifically,
        #    let's override the throttle while letting the model's steering stand
        if go_forward:
            predicted_throttle = FORWARD_THROTTLE_CMD
        else:
            predicted_throttle = BACKWARD_THROTTLE_CMD

        # Clip to valid range if needed (e.g., 0..128)
        steer_cmd = int(np.clip(predicted_steer, 0, 128))
        throttle_cmd = int(np.clip(predicted_throttle, 0, 128))

        # 5) Publish the command to the robot
        command_string = f"{throttle_cmd} {steer_cmd}"
        client.publish(MQTT_TOPIC_COMMAND, command_string)
        print(f"Waypoint {i+1}/{len(waypoints)} -> cmd={command_string}")

        # 6) Update last_steer & last_throttle
        last_steer = steer_cmd
        last_throttle = throttle_cmd

        # 7) Sleep to allow movement
        time.sleep(0.5)

# -----------------------------------------------------------------------------
# 7. MAIN EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Wait briefly for MQTT to connect
    time.sleep(2.0)

    # Move in a forward circle
    move_circle(waypoints_forward, go_forward=True)

    # Pause
    print("\nCompleted forward circle. Pausing briefly...")
    time.sleep(2.0)

    # Move in the same circle but backwards
    move_circle(waypoints_backward, go_forward=False)

    print("\nCompleted backward circle. Stopping robot...\n")

    # Send a final stop command
    stop_cmd = "64 64"  # Example "stop" if 64=neutral
    client.publish(MQTT_TOPIC_COMMAND, stop_cmd)

    # Cleanup
    client.loop_stop()
    client.disconnect()
    print("Done.")
