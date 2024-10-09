import math
import cv2
import cvzone
import face_recognition
from ultralytics import YOLO
from gtts import gTTS
import os
import time

# Load YOLO model for object detection
model = YOLO('../../../Desktop/Obj-det-project/pythonProject/Yolo-weights/yolov8n.pt')

# Class names for object detection
classname = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Load face encodings and names for face recognition
known_face_encodings = []
known_face_names = []

# Load known face images and encode them
known_person_image = face_recognition.load_image_file('images/1.jpg')
known_person_image2 = face_recognition.load_image_file('images/2.jpg')

known_person_encoding = face_recognition.face_encodings(known_person_image)[0]
known_person_encoding2 = face_recognition.face_encodings(known_person_image2)[0]

known_face_encodings.append(known_person_encoding)
known_face_encodings.append(known_person_encoding2)

known_face_names.append("Abhinav")
known_face_names.append("Sudhanshu")

# Initialize video capture from webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Timer for speech (5-second interval)
last_spoken_time = time.time()

while True:
    success, img = cap.read()

    # Get the width of the frame to determine object position
    frame_height, frame_width, _ = img.shape
    center_x = frame_width // 2  # Horizontal center of the frame

    # Convert the BGR image to RGB for face recognition
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Face recognition
    face_locations = face_recognition.face_locations(rgb_img)
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

    # For each detected face, check if it's a known face
    face_detected = False
    detected_objects = []  # To store detected items
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Determine if face is on the left or right
        face_position = "left" if (left + right) // 2 < center_x else "right"
        name_with_position = f"{name} on the {face_position}"

        # Draw rectangle around the face and label it
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(img, name_with_position, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Add the detected face to detected_objects
        detected_objects.append(name_with_position)
        face_detected = True

    # If no face is detected or labeled as "Unknown," run YOLO object detection
    if not face_detected:
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = (x2 - x1), (y2 - y1)

                # Determine if object is on the left or right
                object_position = "left" if (x1 + x2) // 2 < center_x else "right"

                # Draw rectangle using cvzone for object detection
                cvzone.cornerRect(img, (x1, y1, w, h))

                # Confidence score for object detection
                conf = math.ceil((box.conf[0] * 100)) / 100

                # Class name for the detected object
                cls = int(box.cls[0])
                label = classname[cls]

                # Add the detected object and its position to detected_objects list
                label_with_position = f"{label} on the {object_position}"
                detected_objects.append(label_with_position)

                # Display object name, position, and confidence score
                cvzone.putTextRect(img, f'{label_with_position} {conf}', (max(0, x1), max(35, y1)), scale=3,
                                   thickness=3)

    # Convert the detected object names to speech every 5 seconds
    current_time = time.time()
    if current_time - last_spoken_time >= 10 and detected_objects:
        detection_text = ', '.join(detected_objects)  # Create a single text with detected items
        tts = gTTS(text=f"I detected: {detection_text}", lang='en', slow=False)
        tts.save("detected.mp3")
        os.system("start detected.mp3")

        # Update the last spoken time to current
        last_spoken_time = current_time

    # Show the video with detections
    cv2.imshow("Image", img)

    # Press 'q' to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
