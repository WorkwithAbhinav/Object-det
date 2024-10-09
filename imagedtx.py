# Import the required module for text
# to speech conversion
from gtts import gTTS

# This module is imported so that we can
# play the converted audio
import os

# The text that you want to convert to audio
mytext = 'heheahea , re betichodke , pichvade me se dhua nikal jaoga !'

# Language in which you want to convert
language = 'en'

# Passing the text and language to the engine,
# here we have marked slow=False. Which tells
# the module that the converted audio should
# have a high speed
myobj = gTTS(text=mytext, lang=language, slow=False)

# Saving the converted audio in a mp3 file named
# welcome
myobj.save("welcome.mp3")

# Playing the converted file
os.system("start welcome.mp3")













# import cv2
# import face_recognition
#
# known_face_encodings = []
# known_face_names = []
#
# known_person_image = face_recognition.load_image_file('images/1.jpg')
# known_person_image2 = face_recognition.load_image_file('images/2.jpg')
# # known_person_image3 = face_recognition.load_image_file('images/3.png')
#
# known_person_encoding = face_recognition.face_encodings(known_person_image)[0]
# known_person_encoding2 = face_recognition.face_encodings(known_person_image2)[0]
# # known_person_encoding3 = face_recognition.face_encodings(known_person_image3)[0]
#
# known_face_encodings.append(known_person_encoding)
# known_face_encodings.append(known_person_encoding2)
# # known_face_encodings.append(known_person_encoding3)
#
# known_face_names.append("Abhinav")
# known_face_names.append("Sudhanshu")
# # known_face_names.append("Sarthak")
#
#
# cap = cv2.VideoCapture(0)
#
# while True:
#     ret, frame = cap.read()
#
#     face_location = face_recognition.face_locations(frame)
#     face_encodings = face_recognition.face_encodings(frame , face_location)
#
#     for (top , right, bottom , left ), face_encodings in zip(face_location , face_encodings):
#         matches = face_recognition.compare_faces(known_face_encodings,face_encodings)
#         name = "unknown"
#
#         if True in matches:
#             first_match_index = matches.index(True)
#             name = known_face_names[first_match_index]
#
#         cv2.rectangle(frame , (left , top), (right,bottom), (0,0,255) ,2)
#         cv2.putText(frame , name ,(left , top -10), cv2.FONT_HERSHEY_SIMPLEX, 0.9 , (0,0,255) ,2)
#     cv2.imshow("video" , frame)
#
#     if cv2.waitKey(1) & 0xff == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()