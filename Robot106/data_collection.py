#!/usr/bin/env python3
"""
data_collection.py
A Flask-based web GUI for manually controlling the robot and logging sensor data.
The interface works over SSH (headless) by running a web server.
"""

from flask import Flask, render_template_string, request, jsonify
import csv, os, time, math, threading
import gpsd
import IMU

# ----- Configuration -----
DATA_LOG_PATH = "training_data.csv"
LOG_INTERVAL = 0.1  # seconds (approx 10Hz)

# Discrete commands (steering, throttle) in range [0..126]
CMD_FORWARD  = (64, 126)  # (steering, throttle)
CMD_BACKWARD = (64, 0)
CMD_LEFT     = (126, 64)
CMD_RIGHT    = (0, 64)
CMD_STOP     = (64, 64)

# Global variables for current command and logging flag
current_command = CMD_STOP
logging_active = False

# Initialize Flask app
app = Flask(__name__)

# ------------------ Sensor Functions ------------------
def calculate_heading():
    """
    Reads IMU sensor values to compute heading [0..360].
    """
    ACCx = IMU.readACCx()
    ACCy = IMU.readACCy()
    ACCz = IMU.readACCz()
    MAGx = IMU.readMAGx()
    MAGy = IMU.readMAGy()
    MAGz = IMU.readMAGz()

    acc_magnitude = math.sqrt(ACCx**2 + ACCy**2 + ACCz**2)
    if acc_magnitude == 0:
        return 0.0
    accXnorm = ACCx / acc_magnitude
    accYnorm = ACCy / acc_magnitude
    try:
        pitch = math.asin(accXnorm)
    except ValueError:
        pitch = 0.0
    if abs(math.cos(pitch)) < 1e-9:
        roll = 0.0
    else:
        roll = -math.asin(accYnorm / math.cos(pitch))
    magXcomp = MAGx * math.cos(pitch) + MAGz * math.sin(pitch)
    magYcomp = (MAGx * math.sin(roll) * math.sin(pitch) +
                MAGy * math.cos(roll) -
                MAGz * math.sin(roll) * math.cos(pitch))
    heading = math.degrees(math.atan2(magYcomp, magXcomp))
    if heading < 0:
        heading += 360
    return heading

def get_accelerometer_data():
    """
    Returns (ax, ay) in m/s^2 by applying a scaling factor.
    """
    ACCx = IMU.readACCx()
    ACCy = IMU.readACCy()
    scaling_factor = 0.000598550415  # Adjust for your IMU
    return ACCx * scaling_factor, ACCy * scaling_factor

def get_gps_data():
    """
    Returns (lat, lon) from gpsd if available, else (0.0, 0.0).
    """
    try:
        pkt = gpsd.get_current()
        if pkt.mode >= 2:
            return pkt.lat, pkt.lon
    except Exception as e:
        print("GPS error:", e)
    return 0.0, 0.0

# ------------------ Data Logging Loop ------------------
def data_logging_loop():
    """
    Continuously log sensor data and current command to CSV.
    """
    global logging_active, current_command

    # Create CSV file with header if it doesn't exist
    if not os.path.exists(DATA_LOG_PATH):
        with open(DATA_LOG_PATH, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "gps_lat", "gps_lon", "imu_heading",
                "accel_x", "accel_y", "steering_cmd", "throttle_cmd"
            ])
    
    while logging_active:
        ts = time.time()
        lat, lon = get_gps_data()
        heading = calculate_heading()
        ax, ay = get_accelerometer_data()
        steering, throttle = current_command
        
        with open(DATA_LOG_PATH, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ts, lat, lon, heading, ax, ay, steering, throttle])
        
        time.sleep(LOG_INTERVAL)

# ------------------ Flask Routes ------------------
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Robot Data Collection</title>
    <style>
        button { font-size: 16px; padding: 10px; margin: 5px; }
        .container { text-align: center; padding: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Robot Data Collection GUI</h1>
        <div>
            <button onclick="sendCommand('forward')">Forward</button>
            <button onclick="sendCommand('backward')">Backward</button>
        </div>
        <div>
            <button onclick="sendCommand('left')">Left</button>
            <button onclick="sendCommand('stop')">Stop</button>
            <button onclick="sendCommand('right')">Right</button>
        </div>
        <div style="margin-top:20px;">
            <button onclick="startLogging()">Start Logging</button>
            <button onclick="stopLogging()">Stop Logging</button>
        </div>
        <div id="status" style="margin-top:20px; font-size:14px;"></div>
    </div>
    <script>
        function sendCommand(direction) {
            fetch("/set_command", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ direction: direction })
            }).then(response => response.json())
              .then(data => { document.getElementById("status").innerText = data.message; });
        }
        function startLogging() {
            fetch("/start_logging").then(response => response.json())
              .then(data => { document.getElementById("status").innerText = data.message; });
        }
        function stopLogging() {
            fetch("/stop_logging").then(response => response.json())
              .then(data => { document.getElementById("status").innerText = data.message; });
        }
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/set_command", methods=["POST"])
def set_command():
    global current_command
    data = request.get_json()
    direction = data.get("direction", "stop")
    if direction == "forward":
        current_command = CMD_FORWARD
    elif direction == "backward":
        current_command = CMD_BACKWARD
    elif direction == "left":
        current_command = CMD_LEFT
    elif direction == "right":
        current_command = CMD_RIGHT
    else:
        current_command = CMD_STOP
    return jsonify({"message": f"Command set to {direction.upper()}: {current_command}"})

@app.route("/start_logging")
def start_logging():
    global logging_active
    if not logging_active:
        logging_active = True
        threading.Thread(target=data_logging_loop, daemon=True).start()
        return jsonify({"message": "Data logging started."})
    return jsonify({"message": "Data logging already active."})

@app.route("/stop_logging")
def stop_logging():
    global logging_active
    logging_active = False
    return jsonify({"message": "Data logging stopped."})

@app.route("/status")
def status():
    # Return current sensor status (for future use with AJAX polling)
    lat, lon = get_gps_data()
    heading = calculate_heading()
    ax, ay = get_accelerometer_data()
    return jsonify({
        "gps": {"lat": lat, "lon": lon},
        "heading": heading,
        "accel": {"ax": ax, "ay": ay},
        "command": current_command
    })

# ------------------ Initialization ------------------
if __name__ == "__main__":
    try:
        gpsd.connect()
    except Exception as e:
        print("GPSD connection error:", e)
    IMU.detectIMU()
    if IMU.BerryIMUversion == 99:
        print("No BerryIMU found. Exiting.")
        exit(1)
    else:
        IMU.initIMU()
    
    # Run the web server on all interfaces (0.0.0.0) so you can access it via SSH.
    app.run(host="0.0.0.0", port=5000, debug=False)
