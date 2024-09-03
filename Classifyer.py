# Display "Starting..." message
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
broker_ip = "xxx.xxx.xxx.xx"	# Change this to match the IP adress of your MQTT broker
broker_port = xxxx		# Change this to match the port of your MQTT broker
class_topic = "class"
confidence_topic = "confidence"

# Get the absolute path of the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the JSON config file
config_path = os.path.join(script_dir, "config.json")
with open(config_path, 'r') as f:
    config = json.load(f)

# Check if a GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create the model using the config information
print("Loading model...")
model = timm.create_model(config["architecture"], pretrained=False, num_classes=config["num_classes"])
model.load_state_dict(torch.load(os.path.join(script_dir, 'pytorch_model.bin'), weights_only=True))
model.to(device)  # Move model to GPU if available
model.eval()
print("Model loaded.")

# Prompt the user for the n-th frame for inference
n_frame = int(input("Which Nth frame should be used for inference? (e.g., 5 for every 5th frame): ").strip())

# Prompt the user if they want to display the video
display_video = input("Do you want to display the video? (y/n): ").strip().lower()

# Define the image transformation pipeline using the config information
transform = transforms.Compose([
    transforms.Resize((config["pretrained_cfg"]["input_size"][1], config["pretrained_cfg"]["input_size"][2])),
    transforms.ToTensor(),
    transforms.Normalize(mean=config["pretrained_cfg"]["mean"], std=config["pretrained_cfg"]["std"])
])

# Initialize the MQTT client
client = mqtt.Client(protocol=mqtt.MQTTv311)

# Connect to the MQTT broker
connected = False
try:
    client.connect(broker_ip, broker_port)
    print("Connected to MQTT broker.")
    connected = True
except ConnectionRefusedError:
    print("MQTT broker not found.")

# Variable for tracking the previous predicted class and confidence
previous_class = None
previous_confidence = None

# Function to perform inference on the image
def classify_image(image):
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)  # Move tensor to GPU if available

    # Perform inference
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)[0]
        predicted_class_idx = torch.argmax(probabilities).item()
        predicted_class = config["label_names"][predicted_class_idx]
        confidence = round(probabilities[predicted_class_idx].item(), 1)  # Round to 1 decimal

    return predicted_class, confidence

# Function to display the webcam feed
def display_webcam():
    while True:
        # Read the current frame from the webcam
        ret, frame = cap.read()

        # Check if the frame is valid
        if not ret:
            break

        # Display the frame in the window
        cv2.imshow("Webcam Feed", frame)

        # Exit the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()

# Function to process the frames and perform classification
def process_frames():
    global previous_class, previous_confidence
    frame_count = 0  # Initialize frame counter
    while True:
        # Read the current frame from the webcam
        ret, frame = cap.read()

        # Check if the frame is valid
        if not ret:
            break

        # Convert the frame to PIL Image format
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        frame_count += 1  # Increment the frame counter

        # Perform classification only on every nth frame
        if frame_count % n_frame == 0:
            predicted_class, confidence = classify_image(image)

            # Check if the predicted class or confidence has changed before printing
            if predicted_class != previous_class or confidence != previous_confidence:
                print("Predicted class:", predicted_class)
                print("Confidence:", confidence)

                # Publish the predicted class if it has changed
                client.publish(class_topic, predicted_class)
                previous_class = predicted_class

                # Publish the confidence score if it has changed
                client.publish(confidence_topic, confidence)
                previous_confidence = confidence

# Start the display thread if the user wants to display the video
if display_video == 'y':
    display_thread = threading.Thread(target=display_webcam)
    display_thread.start()
else:
    print("Video display skipped.")

# Start the processing thread
processing_thread = threading.Thread(target=process_frames)
processing_thread.start()

# Wait for both threads to finish if display_video is True
if display_video == 'y':
    display_thread.join()

processing_thread.join()

# Disconnect from the MQTT broker
client.disconnect()
