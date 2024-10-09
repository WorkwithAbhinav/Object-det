import math
import cv2
import cvzone
import face_recognition
from ultralytics import YOLO

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
known_person_image = face_recognition.load_image_file('../images/1.jpg')
known_person_image2 = face_recognition.load_image_file('../images/2.jpg')

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

while True:
    success, img = cap.read()

    # Convert the BGR image to RGB for face recognition
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Face recognition
    face_locations = face_recognition.face_locations(rgb_img)
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

    # For each detected face, check if it's a known face
    face_detected = False
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw rectangle around the face and label it
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

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

                # Draw rectangle using cvzone for object detection
                cvzone.cornerRect(img, (x1, y1, w, h))

                # Confidence score for object detection
                conf = math.ceil((box.conf[0] * 100)) / 100

                # Class name for the detected object
                cls = int(box.cls[0])

                # If the detected object is 'person' and no known face is found, label as "person"
                label = classname[cls] if classname[cls] != 'person' else 'person'

                # Display object name and confidence score
                cvzone.putTextRect(img, f'{label} {conf}', (max(0, x1), max(35, y1)), scale=3, thickness=3)

    # Show the video with detections
    cv2.imshow("Image", img)

    # Press 'q' to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()






















# import math
#
# from ultralytics import YOLO
# import cv2
# import cvzone
#
# model = YOLO('../../../Desktop/Obj-det-project/pythonProject/Yolo-weights/yolov8n.pt')
#
# classname =[
#   'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
#   'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
#   'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
#   'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#   'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
#   'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
#   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
#   'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#   'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
#   'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
#   'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
#   'toothbrush'
# ]
#
# cap = cv2.VideoCapture(0)
# cap.set(3,1280)
# cap.set(4,720)
#
# while True:
#     success, img = cap.read()
#     results = model(img, stream=True)
#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             x1,y1,x2,y2 = box.xyxy[0]
#             x1, y1, x2, y2 = int(x1),int(y1), int(x2), int(y2)
#             # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
#             w , h = (x2-x1) , (y2-y1)
#             cvzone.cornerRect(img,(x1,y1,w,h))
#
#             # putting a confidence matrix
#             conf = math.ceil((box.conf[0]*100)) /100
#
#             # writing class names
#             cls = int(box.cls[0])
#
#             cvzone.putTextRect(img , f'{classname[cls]} {conf}',(max(0,x1) , max(35 ,y1)) , scale=3 , thickness=3)
#     cv2.imshow("Image",img)
#     cv2.waitKey(5000)





