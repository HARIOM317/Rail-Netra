import cv2
import numpy as np
from keras.models import load_model
from collections import deque
import time
import os
from datetime import datetime

from google.cloud import firestore
from firebase_admin import credentials, initialize_app, storage

# Initialize Firestore client
db = firestore.Client.from_service_account_json('app.json')
# Init firebase with your credentials
cred = credentials.Certificate("app.json")
initialize_app(cred, {'storageBucket': 'railway-management-57dbe.appspot.com'})

def save_video_to_firebase_storage(fileName):
    # Put your local file path 
    # fileName = "arson.mp4"
    bucket = storage.bucket()
    blob = bucket.blob('crime_videos/'+fileName)
    blob.upload_from_filename(fileName)
    # Opt : if you want to make public access from the URL
    blob.make_public()
    return blob.public_url

def save_to_firestore(platform, cctv_url, timestamp, label, accuracy, video_url):
    doc_ref = db.collection(u'crime_videos').document()
    doc_ref.set({
        u'cctv_url': cctv_url,
        u'timestamp': timestamp,
        u'label': label,
        u'accuracy': accuracy,
        u'platform': platform,
        u'video_url': save_video_to_firebase_storage(video_url)
    })

# Load the trained model and define CLASSES_LIST
model = load_model('LRCN_model_fighting_normal_shooting__Date_Time_2023_10_05__12_47_24___Loss_0.601302981376648___Accuracy_0.699999988079071.h5')
CLASSES_LIST = ["Fighting","Normal","Shooting"]

# Parameters
SEQUENCE_LENGTH = 20
# Setting the output directory for generated video clips
OUTPUT_DIR = "analomies_detected/video"  
# Change this label to the one want to avoid detect and trigger video recording
LABEL_TO_DETECT = "Normal"  

def get_output_filename(probability, predicted_class_name):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{predicted_class_name}_{timestamp}_{probability:.2f}.mp4"
    return filename

def predict_on_live_cctv(CCTV_URL, platform):
    # Initialize video capture from a live CCTV URL
    video_capture = cv2.VideoCapture(CCTV_URL)

    # Check if the video capture was successful
    if not video_capture.isOpened():
        print("Error: Could not open video stream.")
        return

    # Get the video's width and height
    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))

    # Initialize a deque to store frames for prediction
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)

    # Initialize variables to manage video clip recording
    recording = False
    start_time = None
    out = None

    while True:
        ret, frame = video_capture.read()

        if not ret:
            break

        # Preprocess the frame (resize and normalize)
        resized_frame = cv2.resize(frame, (64, 64))
        normalized_frame = resized_frame / 255.0

        # Append the pre-processed frame to the queue
        frames_queue.append(normalized_frame)

        if len(frames_queue) == SEQUENCE_LENGTH:
            # Predict the action label
            predicted_labels_probabilities = model.predict(np.expand_dims(frames_queue, axis=0))[0]
            predicted_label = np.argmax(predicted_labels_probabilities)
            predicted_class_name = CLASSES_LIST[predicted_label]

            # Display the predicted label on the frame
            cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if predicted_class_name != LABEL_TO_DETECT:
                # Start recording if the detected label is not Normal
                if not recording:
                    recording = True
                    start_time = time.time()
                    probability = np.max(predicted_labels_probabilities)
                    output_filename = get_output_filename(probability,predicted_class_name)
                    fourcc = cv2.VideoWriter_fourcc(*'H264')
                    out = cv2.VideoWriter(output_filename, fourcc, 30.0, (frame_width, frame_height))

            # Stop recording after a certain duration (e.g., 5 seconds)
            if recording and (time.time() - start_time) > 5:
                recording = False
                out.release()
            # Save information to Firestore ::changes
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                accuracy = float(np.max(predicted_labels_probabilities))
                save_to_firestore(platform, CCTV_URL, timestamp, predicted_class_name, accuracy, output_filename)
       
        if recording:
            # Write the current frame to the video clip
            out.write(frame)

        # Display the frame with predictions
        cv2.imshow('Crime Detection', frame)

        # Exit the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and video writer objects
    video_capture.release()
    if recording:
        out.release()
        # Save information to Firestore ::changes
        save_to_firestore(platform, CCTV_URL, timestamp, predicted_class_name, accuracy, output_filename)
    cv2.destroyAllWindows()


# Replace with CCTV URL
# CCTV_URL = 'https://firebasestorage.googleapis.com/v0/b/railway-management-57dbe.appspot.com/o/videos%2Fshooting%2FVID-20231002-WA0017.mp4?alt=media&token=1e9c539a-d68b-4440-9ca7-26c600c4e345&_gl=1*12go3xa*_ga*MTIwMzA1MDMyMS4xNjk1NjE2MzY0*_ga_CW55HF8NVT*MTY5NjI1MTI0Mi45LjEuMTY5NjI1NDAyOS4xNS4wLjA.mp4'  
# Run the function
# predict_on_live_cctv(CCTV_URL,'Platform 3')


def fetch_cctv_data():
    cctv_documents = db.collection(u'cctv_urls').get()
    # Process each document
    for doc in cctv_documents:
        document_data = doc.to_dict()
        platform = document_data.get(u'platform')
        cctv_url = document_data.get(u'urlLink')
        # You can add more fields as needed

        # Call your video analysis function here using platform and cctv_url
        analyze_video(platform, cctv_url)

def analyze_video(platform, cctv_url):
    # video analysis code 
    print(f"\nAnalyzing video from platform: {platform}, \nURL: {cctv_url}\n")
    predict_on_live_cctv(cctv_url, platform)
# Fetch and analyze CCTV data
fetch_cctv_data()



