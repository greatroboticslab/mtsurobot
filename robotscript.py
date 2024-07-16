import socket  # for network communication
import time  # for sleep intervals
import numpy as np  # for numerical operations
import pandas as pd  # for data manipulation
import re  # for regex operations
import folium  # for creating maps
from threading import Thread, Lock  # for multi-threading and synchronization
from filterpy.kalman import ExtendedKalmanFilter as EKF  # for kalman filtering
import gradio as gr  # for creating web interfaces
from flask import Flask, request, jsonify  # for creating a web server

# global variables
gps_connected = False  # gps connection status
gps_data = []  # list to store gps data
current_speed = 350  # default speed of the robot
battery_status = "Unknown"  # default battery status
target_coordinates = None  # target coordinates for the robot

app = Flask(__name__)  # initialize flask app

class RobotSocket:
    def __init__(self, ip, port):
        self.ip = ip  # ip address of the robot
        self.port = port  # port number of the robot
        self.sock = None  # socket object
        self.lock = Lock()  # lock for thread safety
        self.connect()  # establish connection

    def connect(self):
        with self.lock:  # ensure thread safety
            if self.sock:
                self.sock.close()  # close existing socket if any
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # create new socket
            try:
                self.sock.connect((self.ip, self.port))  # connect to robot
                print(f"Successfully connected to {self.ip}:{self.port}")
            except ConnectionRefusedError:
                print(f"ConnectionRefusedError: Could not connect to {self.ip}:{self.port}")
                self.sock = None  # set socket to none on failure
            except Exception as e:
                print(f"Exception: {e}")
                self.sock = None  # set socket to none on exception

    def send_command(self, cmd):
        with self.lock:  # ensure thread safety
            if self.sock is None:
                self.connect()  # reconnect if not connected
                if self.sock is None:
                    print("Error: Not connected to robot")
                    return "Error: Not connected to robot"
            try:
                send_cmd = cmd + "\r\n"  # add CRLF to the command
                self.sock.sendall(send_cmd.encode())  # send command to robot
                response = self.sock.recv(1024).decode()  # receive response
                return response
            except Exception as e:
                print(f"Error in send_command: {e}")
                self.connect()  # reconnect on error
                return None

    def close(self):
        with self.lock:  # ensure thread safety
            if self.sock:
                self.sock.close()  # close socket
                self.sock = None  # set socket to none

def parse_gps_data(data):
    match = re.search(r'\$GPRMC,.*?,A,(.*?),(N|S),(.*?),(E|W)', data)  # parse gps data using regex
    if match:
        lat, lat_dir, lon, lon_dir = match.groups()  # extract latitude and longitude
        lat = float(lat[:2]) + float(lat[2:]) / 60  # convert latitude to decimal
        lon = float(lon[:3]) + float(lon[3:]) / 60  # convert longitude to decimal
        if lat_dir == 'S':
            lat *= -1  # adjust for south latitude
        if lon_dir == 'W':
            lon *= -1  # adjust for west longitude
        return lat, lon
    return None, None  # return none if parsing fails

def parse_imu_data(data):
    gyro_match = re.search(r'GYRO,(-?\d+),(-?\d+),(-?\d+)', data)  # parse gyroscope data
    acc_match = re.search(r'ACC,(-?\d+),(-?\d+),(-?\d+)', data)  # parse accelerometer data
    compass_match = re.search(r'COMP,(-?\d+),(-?\d+),(-?\d+)', data)  # parse compass data
    if gyro_match and acc_match and compass_match:
        gyro = np.array([int(g) for g in gyro_match.groups()])  # convert gyroscope data to array
        acc = np.array([int(a) for a in acc_match.groups()])  # convert accelerometer data to array
        compass = np.array([int(c) for c in compass_match.groups()])  # convert compass data to array
        heading = np.arctan2(compass[1], compass[0]) * (180 / np.pi)  # calculate heading angle
        return gyro, acc, heading
    return None, None, None  # return none if parsing fails

def move_robot(direction):
    try:
        estop_response = robot.send_command("MMW !MG")  # undo emergency stop
        if estop_response is None or 'Error' in estop_response:
            print("Failed to undo estop")
            return "Failed to undo estop"

        # send movement commands based on direction
        if direction == "right":
            response = robot.send_command(f"MMW !M {current_speed} {current_speed}")
        elif direction == "left":
            response = robot.send_command(f"MMW !M {-current_speed} {-current_speed}")
        elif direction == "forward":
            response = robot.send_command(f"MMW !M {current_speed} {-current_speed}")
        elif direction == "backward":
            response = robot.send_command(f"MMW !M {-current_speed} {current_speed}")
        elif direction == "stop":
            response = robot.send_command("MMW !EX")

        if response is None or 'Error' in response:
            print(f"Error: Failed to move {direction}. Response: {response}")
            return f"Failed to move {direction}. Response: {response}"

        print(f"Moved {direction} successfully. Response: {response}")
        return f"Moved {direction} successfully. Response: {response}"
    except Exception as e:
        print(f"Exception in move_robot: {e}")
        return f"Exception: {e}"

def move_to_coordinates(lat, lon):
    try:
        estop_response = robot.send_command("MMW !MG")  # undo emergency stop
        if estop_response is None or 'Error' in estop_response:
            print("Failed to undo estop")
            return "Failed to undo estop"

        while True:
            gps_response = robot.send_command("GPS")  # get current gps data
            current_lat, current_lon = parse_gps_data(gps_response)  # parse gps data

            if current_lat is None or current_lon is None:
                print("Error: Could not get current gps position")
                return "Error: Could not get current gps position"

            if abs(current_lat - lat) < 0.0001 and abs(current_lon - lon) < 0.0001:
                break  # stop if target coordinates are reached

            # adjust movement based on current and target coordinates
            if current_lat < lat:
                move_robot("forward")
            elif current_lat > lat:
                move_robot("backward")

            if current_lon < lon:
                move_robot("right")
            elif current_lon > lon:
                move_robot("left")

            time.sleep(1)  # wait for 1 second

        move_robot("stop")  # stop the robot
        return f"Moved to coordinates {lat}, {lon} successfully."
    except Exception as e:
        print(f"Exception in move_to_coordinates: {e}")
        return f"Exception: {e}"

def update_map():
    return create_map()  # create and return the map

def check_gps_connection():
    global gps_connected
    return "Connected" if gps_connected else "Disconnected"  # check gps connection status

def initialize_ekf():
    ekf = EKF(dim_x=6, dim_z=3)  # initialize extended kalman filter
    ekf.x = np.zeros(6)  # initial state
    ekf.F = np.eye(6)  # state transition matrix
    ekf.H = np.array([[1, 0, 0, 0, 0, 0],  # measurement matrix for gps lat
                      [0, 1, 0, 0, 0, 0],  # measurement matrix for gps lon
                      [0, 0, 1, 0, 0, 0]])  # measurement matrix for compass heading

    ekf.P *= 1000  # covariance matrix

    ekf.Q = np.eye(6)  # process noise matrix
    ekf.Q[3, 3] = 0.1  # process noise for velocity_lat
    ekf.Q[4, 4] = 0.1  # process noise for velocity_lon
    ekf.Q[5, 5] = 0.1  # process noise for heading_rate

    ekf.R = np.eye(3) * 5  # measurement noise matrix

    return ekf

def HJacobian(x):
    return np.array([[1, 0, 0, 0, 0, 0],  # jacobian matrix for gps lat
                     [0

, 1, 0, 0, 0, 0],  # jacobian matrix for gps lon
                     [0, 0, 1, 0, 0, 0]])  # jacobian matrix for compass heading

def Hx(x):
    return np.array([x[0], x[1], x[2]])  # expected measurement

def track_robot():
    global gps_connected, gps_data, robot, ekf, battery_status

    gps_data = []  # initialize gps data list
    ekf = initialize_ekf()  # initialize kalman filter

    try:
        while True:
            gps_response = robot.send_command("GPS")  # get gps data from robot
            imu_response = robot.send_command("IMU")  # get imu data from robot
            robot.send_command("SYS CAL")  # calibrate the system
            current_lat, current_lon = parse_gps_data(gps_response)  # parse gps data
            gyro, acc, heading = parse_imu_data(imu_response)  # parse imu data

            if gyro is not None and acc is not None:
                dt = 1  # assuming a constant time step of 1 second
                ekf.F = np.array([[1, 0, 0, dt, 0, 0],  # state transition matrix
                                  [0, 1, 0, 0, dt, 0],
                                  [0, 0, 1, 0, 0, dt],
                                  [0, 0, 0, 1, 0, 0],
                                  [0, 0, 0, 0, 1, 0],
                                  [0, 0, 0, 0, 0, 1]])

                ekf.predict()  # predict the next state

            if current_lat is not None and current_lon is not None and heading is not None:
                z = np.array([current_lat, current_lon, heading])  # measurement vector
                ekf.update(z, HJacobian, Hx)  # update the filter
                estimated_lat, estimated_lon, estimated_heading = ekf.x[0], ekf.x[1], ekf.x[2]

                if is_accurate_data(current_lat, current_lon, estimated_lat, estimated_lon):
                    gps_data.append({  # append the data to gps data list
                        'Pure_GPS_Lat': current_lat,
                        'Pure_GPS_Lon': current_lon,
                        'Kalman_GPS_Lat': estimated_lat,
                        'Kalman_GPS_Lon': estimated_lon,
                        'Kalman_Heading': estimated_heading,
                        'IMU_Gyro_X': gyro[0],
                        'IMU_Gyro_Y': gyro[1],
                        'IMU_Gyro_Z': gyro[2],
                        'IMU_Acc_X': acc[0],
                        'IMU_Acc_Y': acc[1],
                        'IMU_Acc_Z': acc[2],
                    })

            battery_response = robot.send_command("MM0 ?V")  # get battery status
            if battery_response:
                battery_voltage = parse_battery_status(battery_response)  # parse battery status
                battery_status = f"Battery Voltage: {battery_voltage}V"

            gps_connected = True  # set gps connection status to true
            time.sleep(1)  # wait for 1 second

    except Exception as e:
        print(f"Error tracking robot: {e}")
        gps_connected = False  # set gps connection status to false on error

def parse_battery_status(data):
    match = re.search(r'V=(\d+),(\d+),(\d+)', data)  # parse battery status using regex
    if match:
        voltage = int(match.group(1)) / 10.0  # convert voltage to float
        return voltage
    return "Unknown"  # return "unknown" if parsing fails

def is_accurate_data(current_lat, current_lon, estimated_lat, estimated_lon, threshold=0.0005):
    distance = np.sqrt((current_lat - estimated_lat)**2 + (current_lon - estimated_lon)**2)  # calculate distance
    return distance < threshold  # return true if distance is within the threshold

def create_map():
    global gps_data, target_coordinates

    if not gps_data:
        initial_lat, initial_lon = 0, 0  # default initial coordinates
    else:
        initial_lat = gps_data[0]['Pure_GPS_Lat']  # initial latitude
        initial_lon = gps_data[0]['Pure_GPS_Lon']  # initial longitude

    map_ = folium.Map(location=[initial_lat, initial_lon], zoom_start=30)  # create map

    if gps_data:
        df = pd.DataFrame(gps_data)  # convert gps data to dataframe

        folium.PolyLine(  # plot pure gps data (blue line)
            locations=df[['Pure_GPS_Lat', 'Pure_GPS_Lon']].values,
            color='blue',
            weight=2.5,
            opacity=1,
            popup='Pure GPS'
        ).add_to(map_)

        if 'Kalman_GPS_Lat' in df.columns and 'Kalman_GPS_Lon' in df.columns:
            folium.PolyLine(  # plot kalman gps data (red line)
                locations=df[['Kalman_GPS_Lat', 'Kalman_GPS_Lon']].dropna().values,
                color='red',
                weight=2.5,
                opacity=1,
                popup='Kalman GPS'
            ).add_to(map_)

        if 'Kalman_GPS_Lat' in df.columns and 'Kalman_GPS_Lon' in df.columns:
            folium.PolyLine(  # plot kalman gps+imu data (green line)
                locations=df[['Kalman_GPS_Lat', 'Kalman_GPS_Lon']].dropna().values,
                color='green',
                weight=2.5,
                opacity=1,
                popup='Kalman GPS+IMU'
            ).add_to(map_)

        if target_coordinates:
            folium.Marker(  # plot target coordinates
                location=target_coordinates,
                icon=folium.Icon(color='red'),
                popup='Target Location'
            ).add_to(map_)

    folium.LatLngPopup().add_to(map_)  # add latlng popup

    return map_._repr_html_()  # return map as html

def save_to_excel():
    global gps_data
    if gps_data:
        df = pd.DataFrame(gps_data)  # convert gps data to dataframe
        df.to_excel("gps_data.xlsx", index=False)  # save to excel
        print("GPS data has been saved to gps_data.xlsx")
    else:
        print("No gps data to save.")

def set_speed(speed):
    global current_speed
    current_speed = speed  # set current speed
    return f"Current speed set to: {current_speed}"

def main():
    global robot, move_buttons, gps_connected

    gps_connected = False  # set gps connection status to false
    robot_ip = "192.168.0.60"  # robot ip address
    robot_port = 10001  # robot port number
    robot = RobotSocket(robot_ip, robot_port)  # initialize robotsocket

    with gr.Blocks() as demo:  # create gradio interface
        gr.Markdown("## Robot Control Panel")  # add markdown

        # define buttons for robot movement
        move_forward_button = gr.Button("Move Forward", variant="primary")
        move_backward_button = gr.Button("Move Backward", variant="primary")
        move_left_button = gr.Button("Move Left", variant="primary")
        move_right_button = gr.Button("Move Right", variant="primary")
        stop_button = gr.Button("Stop", variant="stop")
        save_button = gr.Button("Save to Excel", variant="secondary")

        move_buttons = [move_forward_button, move_backward_button, move_left_button, move_right_button, stop_button]

        move_response = gr.Textbox(label="Move Response")  # textbox for move response

        # define button click actions
        move_forward_button.click(fn=lambda: move_robot("forward"), inputs=[], outputs=[move_response])
        move_backward_button.click(fn=lambda: move_robot("backward"), inputs=[], outputs=[move_response])
        move_left_button.click(fn=lambda: move_robot("left"), inputs=[], outputs=[move_response])
        move_right_button.click(fn=lambda: move_robot("right"), inputs=[], outputs=[move_response])
        stop_button.click(fn=lambda: move_robot("stop"), inputs=[], outputs=[move_response])
        save_button.click(fn=lambda: save_to_excel(), inputs=[], outputs=[])

        gps_status = gr.Label(value=check_gps_connection)  # label for gps status

        gps_map = gr.HTML()  # html element for map

        latitude_input = gr.Number(label="Latitude")  # input for latitude
        longitude_input = gr.Number(label="Longitude")  # input for longitude
        move_to_coords_button = gr.Button("Move to Coordinates")  # button to move to coordinates

        def offline_map_selector():
            return create_map()  # return created map

        def handle_move_to_coordinates(lat, lon):
            global target_coordinates
            if lat is not None and lon is not None:
                target_coordinates = (lat, lon)  # set target coordinates
                return move_to_coordinates(lat, lon)
            else:
                return "Invalid coordinates. Please enter valid latitude and longitude."

        move_to_coords_button

.click(fn=handle_move_to_coordinates, 
                                    inputs=[latitude_input, longitude_input], 
                                    outputs=[move_response])

        demo.load(fn=offline_map_selector, outputs=[gps_map], every=15)  # load map periodically
        demo.load(fn=check_gps_connection, outputs=gps_status, every=15)  # check gps status periodically

        speed_slider = gr.Slider(minimum=1, maximum=500, step=1, value=current_speed, label="Speed")  # speed slider
        speed_output = gr.Textbox(value=f"Current speed set to: {current_speed}", interactive=False)  # speed output

        speed_slider.change(fn=set_speed, inputs=[speed_slider], outputs=[speed_output])  # update speed on change

        battery_status_output = gr.Textbox(value=battery_status, label="Battery Status", interactive=False)  # battery status output

        demo.load(fn=lambda: battery_status, outputs=battery_status_output, every=15)  # update battery status periodically

        def start_tracking():
            track_robot()  # start tracking robot
            return "Tracking started"

        tracking_thread = Thread(target=start_tracking)  # start tracking thread
        tracking_thread.start()

        demo.launch(share=True)  # launch gradio interface

if __name__ == "__main__":
    main()  # run main function
```
