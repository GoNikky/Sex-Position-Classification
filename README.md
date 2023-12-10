## Description
This repository contains a Python script for detecting different sex positions in a video stream using porntechs pre-trained image classification model. Eight classes are supported: ["blowjob", "hardcore", "titjob", "handjob", "pussy-licking", "fingering", "other", "solo"]. The result is published to an MQTT server whenever the detected class changes.


## Usage
To use the script, follow these steps:

1. Download the classification model from https://huggingface.co/porntech/sex-position/resolve/main/pytorch_model.bin and place it in the same directory as the other files
2. Install the required dependencies (pip install torch timm opencv-python paho-mqtt)
3. Set up your MQTT broker (e.g Mosquitto or HiveMQ). Make sure to add the configuration to line 16/17 of the script (broker_ip; broker_port)
4. Run the script, which will start capturing the video stream from the webcam.
5. The script will perform sex position classification on each frame of the video stream and publish the results to the specified MQTT server, whenever they change.
6. Subscribe to the topic "predicted_class" to receive the classification results in real-time.


## Acknowledgments
The sex position classification script utilizes a pre-trained model developed by porntech on Hugging Face. You can find more information about the model and its author on their Hugging Face Model Hub page (https://huggingface.co/porntech/sex-position/tree/main).
