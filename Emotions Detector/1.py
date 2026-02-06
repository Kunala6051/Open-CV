import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses INFO and WARNING

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Suppresses old-style logs

#Ignore above codes they are just to prevent warning messages``


import cv2 as cv
from deepface import DeepFace
import time

harr_cascade = cv.CascadeClassifier('Face Recognizer/harr_faces.xml')

img = cv.imread("Photos/lady.jpg")
cv.imshow("lady", img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


face_rect = harr_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4)
# value of scaleFactor should be between 1.1 and 1.4
# More the scaleFactor, the more the image is reduced at each scale, which can lead to fewer detections but faster processing.
# MinNeighbors is a parameter specifying how many neighbors each candidate rectangle should have to retain it.

for (x,y,w,h) in face_rect:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)


predictions = DeepFace.analyze(img, actions = ["emotion"], enforce_detection=False)
result = predictions[0]['dominant_emotion']
print(predictions[0]['dominant_emotion'])

cv.putText(img, result, (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)

cv.imshow("Emotions Recognized", img)

cv.waitKey(0)



video = cv.VideoCapture(0)


while True:
    isTrue, frame = video.read()
    result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)


    gray2 = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    face_rect2 = harr_cascade.detectMultiScale(gray2, scaleFactor=1.1, minNeighbors=4)
    # value of scaleFactor should be between 1.1 and 1.4
    # More the scaleFactor, the more the image is reduced at each scale, which can lead to fewer detections but faster processing.

    for(x,y,w,h) in face_rect2:
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv.putText(frame, result[0]['dominant_emotion'], (50,50), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)

    resized_frame = cv.resize(frame, (800, 600))  # Resize to 800x600 or any suitable size
    cv.imshow("Faces Detected in First frame of video", resized_frame)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

video.release()
cv.destroyAllWindows()
