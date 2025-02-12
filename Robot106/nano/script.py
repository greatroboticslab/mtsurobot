import cv2
import numpy as np
import time
import os
import paho.mqtt.client as mqtt
import json
import base64

# MQTT Configuration
MQTT_SERVER = "192.168.1.120"  # Update if different
MQTT_PORT = 1883
MQTT_TOPIC_DETECTIONS = "robot/detections"
MQTT_TOPIC_CAMERA = "robot/camera"

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"Connected to MQTT server {MQTT_SERVER} successfully.")
    else:
        print(f"Failed to connect to MQTT server, return code {rc}")

def load_yolo(model_dir):
    """
    Load YOLOv3-Tiny model from the specified directory.

    Args:
        model_dir (str): Directory where YOLOv3-Tiny files are stored.

    Returns:
        net (cv2.dnn_Net): Loaded YOLOv3-Tiny network.
        classes (list): List of class names.
        output_layers (list): Names of the output layers.
    """
    # Paths to the YOLOv3-Tiny files
    config_path = os.path.join(model_dir, 'yolov3-tiny.cfg')
    weights_path = os.path.join(model_dir, 'yolov3-tiny.weights')
    names_path = os.path.join(model_dir, 'coco.names')
    
    # Verify file existence
    for path in [config_path, weights_path, names_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required YOLO file not found: {path}")
    
    # Load class names
    with open(names_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    # Initialize the network
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    
    # If CUDA is available and OpenCV is built with CUDA support, uncomment the following lines
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    # Get the output layer names
    layer_names = net.getLayerNames()
    try:
        # For OpenCV 4.x and above
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except TypeError:
        # For OpenCV 3.x
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    print("YOLOv3-Tiny model loaded successfully.")
    return net, classes, output_layers

def detect_persons(frame, net, output_layers, classes, conf_threshold=0.5, nms_threshold=0.4):
    """
    Detect persons in the given frame using YOLOv3-Tiny.

    Args:
        frame (numpy.ndarray): The input image frame.
        net (cv2.dnn_Net): Loaded YOLOv3-Tiny network.
        output_layers (list): Names of the output layers.
        classes (list): List of class names.
        conf_threshold (float): Confidence threshold to filter weak detections.
        nms_threshold (float): Non-Maximum Suppression threshold.

    Returns:
        frame (numpy.ndarray): The frame with detected persons highlighted.
        detections_list (list): List of detected persons with bbox, center, and area.
        fps (float): Frames per second.
    """
    height, width = frame.shape[:2]
    
    # Create a blob from the input frame
    blob = cv2.dnn.blobFromImage(frame, 
                                 scalefactor=1/255.0, 
                                 size=(416, 416), 
                                 swapRB=True, 
                                 crop=False)
    
    net.setInput(blob)
    start = time.time()
    outputs = net.forward(output_layers)
    end = time.time()
    
    # Initialize lists for detected bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []
    
    # Iterate over each detection
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Filter detections by confidence and class (person)
            if confidence > conf_threshold and classes[class_id] == "person":
                # Scale the bounding box coordinates back relative to the image size
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Calculate the top-left corner of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply Non-Maximum Suppression to suppress overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    detections_list = []
    
    # Check if any detections are present
    if len(indices) > 0:
        # Handle different types of indices based on OpenCV version
        if isinstance(indices, (list, tuple)):
            # If indices are lists or tuples, extract the first element
            indices = [i[0] if isinstance(i, (list, tuple, np.ndarray)) else i for i in indices]
        elif isinstance(indices, np.ndarray):
            # If indices are NumPy arrays, flatten them to a list
            indices = indices.flatten().tolist()
    
        for i in indices:
            # If i is a list or tuple, extract the first element
            if isinstance(i, (list, tuple, np.ndarray)):
                i = i[0]
            x, y, w_box, h_box = boxes[i]
            label = f"{classes[class_ids[i]]}: {int(confidences[i]*100)}%"
            color = (0, 255, 0)  # Green color for bounding boxes
            
            # Ensure bounding boxes are within frame boundaries
            x = max(0, x)
            y = max(0, y)
            w_box = min(w_box, width - x)
            h_box = min(h_box, height - y)
            
            # Draw the bounding box
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
            
            # Put the label above the bounding box
            cv2.putText(frame, label, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Calculate center and area
            cx = x + w_box // 2
            cy = y + h_box // 2
            area = w_box * h_box
            
            # Append detection info to the list
            detection_info = {
                'bbox': [x, y, w_box, h_box],
                'center': [cx, cy],
                'area': area
            }
            detections_list.append(detection_info)
    
    # Calculate FPS
    fps = 1 / (end - start) if end - start > 0 else 0
    
    return frame, detections_list, fps

def main():
    """
    Main function to perform real-time person detection using YOLOv3-Tiny and send detections and camera images via MQTT.
    """
    # Directory where YOLOv3-Tiny model files are stored
    model_dir = './yolov3-tiny'  # Ensure this directory contains 'yolov3-tiny.cfg', 'yolov3-tiny.weights', 'coco.names'
    model_dir = os.path.expanduser(model_dir)
    
    # Load YOLOv3-Tiny
    try:
        net, classes, output_layers = load_yolo(model_dir)
    except FileNotFoundError as e:
        print(e)
        return
    
    # Initialize MQTT Client
    client_mqtt = mqtt.Client()
    client_mqtt.on_connect = on_connect
    client_mqtt.connect(MQTT_SERVER, MQTT_PORT, 60)
    client_mqtt.loop_start()
    
    # Initialize video capture (0 for default webcam)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return
    
    # Optional: Set frame width and height for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Starting real-time person detection. Press 'q' to exit.")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture image")
                break
            
            # Perform person detection
            frame, detections, fps = detect_persons(frame, net, output_layers, classes)
            
            # Display FPS on frame
            cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                print("Failed to encode frame")
                continue
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            
            # Prepare and publish camera image data
            camera_payload = {
                'image': jpg_as_text
            }
            client_mqtt.publish(MQTT_TOPIC_CAMERA, json.dumps(camera_payload))
            print(f"Published camera frame to MQTT topic '{MQTT_TOPIC_CAMERA}'")
            
            # Publish detections via MQTT
            if detections:
                # Prepare the message payload
                payload = {
                    'detections': detections
                }
                # Publish the payload as a JSON string
                client_mqtt.publish(MQTT_TOPIC_DETECTIONS, json.dumps(payload))
                print(f"Published {len(detections)} detection(s) to MQTT topic '{MQTT_TOPIC_DETECTIONS}'")
            
            # Display the resulting frame (optional)
            cv2.imshow('YOLOv3-Tiny Real-Time Person Detection', frame)
            
            # Exit mechanism: Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting real-time person detection.")
                break
    
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        client_mqtt.loop_stop()
        client_mqtt.disconnect()

if __name__ == "__main__":
    main()
