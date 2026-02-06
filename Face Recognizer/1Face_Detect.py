import cv2 as cv
import numpy as np

# We will perform FACE DETECTION with OpenCV using Haar Cascades
# Load the pre-trained Haar Cascade classifier for face detection

# Classifier is a file that contains the trained model for detecting faces that gives us the ability to detect faces in images or videos

img = cv.imread("Photos/group 2.jpg")
cv.imshow("img", img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray", gray)
    
harr_cascade = cv.CascadeClassifier('Face Recognizer/harr_faces.xml')

face_rect = harr_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
# scaleFactor: Specifies how much the image size is reduced at each image scale.
# minNeighbors: Specifies how many neighbors each candidate rectangle should have to retain it.
# detectMultiScale returns a list of rectangles where faces are detected.

# just remember more neighbors means fewer false positives but may miss some faces
# less neighbors means more false positives but may detect more faces
# As it relates to noise in the image, more neighbors can help filter out noise, while fewer neighbors may lead to detecting noise as faces.

print(f'Number of faces found: {len(face_rect)}')

for(x,y,w,h) in face_rect:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
cv.imshow("Faces Detected",img)

cv.waitKey(0)



video = cv.VideoCapture("Videos/Sample2.mp4")

cv.namedWindow("Faces Detected in First frame of video", cv.WINDOW_NORMAL)
cv.resizeWindow("Faces Detected in First frame of video", 800, 600)

while True:
    isTrue, frame = video.read()
    if not isTrue:
        break

    gray2 = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    face_rect2 = harr_cascade.detectMultiScale(
        gray2, scaleFactor=1.2, minNeighbors=4
    )

    for (x, y, w, h) in face_rect2:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imshow("Faces Detected in First frame of video", frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

video.release()
cv.destroyAllWindows()

# Remember haar cascades are not perfect and may not detect all faces, especially in complex images or videos.
# The effectiveness of face detection can vary based on factors like lighting, angle, and occlusion.

# The effective way to improve detection is to use more advanced methods like deep learning-based face detectors, such as DNN (Deep Neural Networks) models.