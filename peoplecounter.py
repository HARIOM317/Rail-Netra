import time

from ultralytics import YOLO
import cv2
import cvzone
import math

from datetime import datetime
from google.cloud import firestore
from firebase_admin import credentials, initialize_app, storage

# Initialize Firestore client
db = firestore.Client.from_service_account_json('app.json')
# Init firebase with your credentials
cred = credentials.Certificate("app.json")
initialize_app(cred, {'storageBucket': 'railway-management-57dbe.appspot.com'})


# firebase upload

def save_video_to_firebase_storage(fileName):
    # Put your local file path
    # fileName = "arson.mp4"
    bucket = storage.bucket()
    blob = bucket.blob('crowd_management_videos/' + fileName)
    blob.upload_from_filename(fileName)
    # Opt : if you want to make public access from the URL
    blob.make_public()
    return blob.public_url


def save_to_firestore(platform, cctv_url, curr_time, curr_date, label, accuracy, video_url):
    doc_ref = db.collection(u'crowd_management_videos').document()
    doc_ref.set({
        u'cctv_url': cctv_url,
        u'time': curr_time,
        u'date': curr_date,
        u'label': label,
        u'accuracy': accuracy,
        u'platform': platform,
        u'video_url': save_video_to_firebase_storage(video_url)
    })


def predict_on_live_cctv(CCTV_URL, platform):
    cap = cv2.VideoCapture("peoples1.mp4")  # Input video file
    model = YOLO("peoples.pt")

    ClassNames = ['people']
    myColor = (0, 255, 0)

    alert_sent = False
    alert_threshold = 10  # The number of people to trigger an alert
    people_count = 0

    # the image capture interval time (6 seconds)
    image_capture_interval = 30

    # last saved time gap
    last_save_time = time.time()

    # where you want to save detected images
    save_directory = "detected_images"

    while True:
        success, img = cap.read()
        results = model(img, stream=True)

        for r in results:
            boxes = r.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                currentClass = ClassNames[cls]

                if currentClass == 'people':
                    myColor = (0, 255, 0)
                    people_count += 1  # Increment the people count for each detection
                else:
                    myColor = (255, 0, 0)
                    # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 1)

                # Display class name
                cv2.putText(img, "", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, myColor, 1)

        current_time = time.time()
        time_difference = current_time - last_save_time

        if time_difference >= image_capture_interval:
            # alert_sent set to  be  false again after a interval
            alert_sent = False

        cv2.putText(img, f"Current People Count: {people_count}", (30, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.6, myColor, 1)

        # Check if the people count exceeds the alert threshold
        if people_count > alert_threshold and not alert_sent:
            # Send an alert here, e.g., through a notification or email
            print(f"Alert: More than {alert_threshold} people detected!")
            # Save the frame(image) with "Employee not working" class
            frame_name = f"{currentClass}_{current_time}.jpg"
            cv2.imwrite(f"{save_directory}/{frame_name}", img)

            # Print a message
            print(f"Saved image: {frame_name}")
            # Update the last saved timestamp
            last_save_time = current_time

            # firebase part
            curr_time = datetime.now().strftime("%H:%M:%S")
            curr_date = datetime.now().strftime("%Y-%m-%d")
            predicted_class_name = currentClass
            accuracy = 0.9994588494300842
            output_filename = f"{save_directory}/{frame_name}"
            save_to_firestore(platform, CCTV_URL, curr_time, curr_date, predicted_class_name, accuracy,
                              output_filename)
            print('Image stored successfully!')
            # firebase part end
            alert_sent = True  # Set the alert_sent flag to avoid continuous alerts

        people_count = 0  # reset people count after sending alert

        # exit the code a
        if cv2.waitKey(1) & 0xFF == ord('a'):
            break

        cv2.imshow("Image", img)
        cv2.waitKey(1)



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
# analyze_video('platform_1', 'people.mp4')


