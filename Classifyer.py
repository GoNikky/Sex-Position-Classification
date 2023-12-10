print("Starting...")

import threading
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import json
import cv2
import paho.mqtt.client as mqtt
import os

# Initialize the webcam
webcam_index = 0  # Change this value to the desired webcam index (e.g., 0, 1, 2, ...)
cap = cv2.VideoCapture(webcam_index, cv2.CAP_DSHOW)

# MQTT broker configuration
broker_ip = "xxx.xxx.xxx.xx"  # Change this to match the IP adress and port of your MQTT broker
broker_port = xxxx
topic = "predicted_class"

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the JSON config file
config_path = os.path.join(script_dir, "config.json")
with open(config_path, 'r') as f:
    config = json.load(f)

# Create the model using the config information
print("Loading model...")
model = timm.create_model(config["architecture"], pretrained=False, num_classes=config["num_classes"])
model.load_state_dict(torch.load(os.path.join(script_dir, 'pytorch_model.bin')))
model.eval()
print("Model loaded.")

# Define the image transformation pipeline using the config information
transform = transforms.Compose([
    transforms.Resize((config["pretrained_cfg"]["input_size"][1], config["pretrained_cfg"]["input_size"][2])),
    transforms.ToTensor(),
    transforms.Normalize(mean=config["pretrained_cfg"]["mean"], std=config["pretrained_cfg"]["std"])
])

# Initialize the MQTT client and connect to the MQTT broker
client = mqtt.Client()

connected = False
try:
    client.connect(broker_ip, broker_port)
    print("Connected to MQTT broker.")
    connected = True
except ConnectionRefusedError:
    print("MQTT broker not found.")

# Variable for tracking the previous predicted class
previous_class = None

# Perform inference on the image
def classify_image(image):
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add a batch dimension

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)[0]
        predicted_class_idx = torch.argmax(probabilities).item()
        predicted_class = config["label_names"][predicted_class_idx]
        confidence = round(probabilities[predicted_class_idx].item(), 3)

    print("Predicted class:", predicted_class)
    print("Confidence:", confidence)

    return predicted_class

# Display the webcam feed
def display_webcam():
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        cv2.imshow("Webcam Feed", frame)

        # Exit the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Process the frames and perform classification
def process_frames():
    global previous_class
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        predicted_class = classify_image(image)

        if predicted_class != previous_class:
            client.publish(topic, predicted_class)
            previous_class = predicted_class

# Start the display and processing threads
display_thread = threading.Thread(target=display_webcam)
processing_thread = threading.Thread(target=process_frames)

display_thread.start()
processing_thread.start()

# Wait for both threads to finish
display_thread.join()
processing_thread.join()

# Disconnect from the MQTT broker
client.disconnect()