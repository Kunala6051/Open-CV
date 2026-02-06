import numpy as np
import cv2 as cv


people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']

harr_cascade = cv.CascadeClassifier('Face Recognizer/harr_faces.xml')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read("Face Recognizer/face_trained.yml")

img = cv.imread(r'C:\Users\Kunal\OneDrive\Desktop\Kunal\OpenCv Tutorial\Faces For Testing\jerry_seinfeld\3.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Person", gray)

face_rect = harr_cascade.detectMultiScale(gray, 1.1, 4)

for(x,y,w,h) in face_rect:
    cropped = gray[y:y+h, x:x+w]
    label, confidence = face_recognizer.predict(cropped)
    print(f'Label: {people[label]} with a confidence of {confidence}')

    # showing the results on img itself
    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

cv.imshow("FACE DETECTED",img)

cv.waitKey(0)