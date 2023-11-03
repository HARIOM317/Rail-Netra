from ultralytics import YOLO
import cv2
import math
import time

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
    blob = bucket.blob('cleanliness_videos/' + fileName)
    blob.upload_from_filename(fileName)
    # Opt : if you want to make public access from the URL
    blob.make_public()
    return blob.public_url


def save_to_firestore(platform, cctv_url, curr_time, curr_date, label, accuracy, video_url):
    doc_ref = db.collection(u'cleanliness_videos').document()
    doc_ref.set({
        u'cctv_url': cctv_url,
        u'time': curr_time,
        u'date': curr_date,
        u'status': label,
        u'accuracy': accuracy,
        u'platform': platform,
        u'video_url': save_video_to_firebase_storage(video_url)
    })


def predict_on_live_cctv(CCTV_URL, platform):
    cap = cv2.VideoCapture(CCTV_URL)  # video file
    model = YOLO("garbage.pt")  # work model

    ClassNames = ['garbage', 'garbage_bag']

    myColor = (0, 255, 0)

    # where you want to save detected images
    save_directory = "detected_images"

    # image counter
    image_count = 1

    # last saved time gap
    last_save_time = time.time()

    # the image capture interval time (6 seconds)
    image_capture_interval = 30

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

                if currentClass == 'garbage' or 'garbage_bag':
                    myColor = (0, 255, 0)

                    # Checking the time since the last saved detected image
                    current_time = time.time()
                    time_difference = current_time - last_save_time

                    if time_difference >= image_capture_interval:
                        # Save the frame(image) with "garbage is detected" class
                        frame_name = f"{currentClass}_{current_time}.jpg"
                        cv2.imwrite(f"{save_directory}/{frame_name}", img)

                        # Print a message
                        print(f"Saved image: {frame_name}")

                        # Update the last saved timestamp
                        last_save_time = current_time
                                                #p217
                        # firebase part
                        curr_time = datetime.now().strftime("%H:%M:%S")
                        curr_date = datetime.now().strftime("%Y-%m-%d")
                        predicted_class_name = currentClass
                        accuracy = 0.8198823928833008
                        output_filename = f"{save_directory}/{frame_name}"
                        save_to_firestore(platform, CCTV_URL, curr_time, curr_date, predicted_class_name, accuracy,
                                          output_filename)
                        print('Image stored successfully!')
                        # firebase part end


                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 1)

                # Display class name
                cv2.putText(img, currentClass, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, myColor, 2)

        cv2.imshow("Image", img)

        # exit the code a
        if cv2.waitKey(1) & 0xFF == ord('a'):
            break

    # Releasing the video capture and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()


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
# local se fatch karne ka
# analyze_video('platform_1', 'garbage-test(1).mp4')

