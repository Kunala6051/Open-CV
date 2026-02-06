import os
import cv2 as cv
import numpy as np

people = []
# storing all the names of people whose images we have

# os.listdir() is used to 
for i in os.listdir(r'C:\Users\Kunal\OneDrive\Desktop\Kunal\OpenCv Tutorial\Faces_Train'):
    # r'' is used to specify a raw string literal, which means backslashes are treated literally
    # instead of being interpreted as escape characters.
    # It returns a list of all files and directories in the specified directory.
    people.append(i)

print(people)
# people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']

features = []
labels = []

DIR = r'C:\Users\Kunal\OneDrive\Desktop\Kunal\OpenCv Tutorial\Faces_Train'

harr_cascade = cv.CascadeClassifier('Face Recognizer/harr_faces.xml')

def create_Train():
    for person in people:
        path = os.path.join(DIR, person) # Joining the directory with the person's name to get the path to their images
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_arr = cv.imread(img_path)

            gray = cv.cvtColor(img_arr, cv.COLOR_BGR2GRAY)

            faces_rect = harr_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                cropped_face = gray[y:y+h, x:x+w] # Cropping the face from the image
                features.append(cropped_face)
                labels.append(label)

create_Train()

print("Training done successfully! -------------")

# print(f'Length of features: {len(features)}')
# print(f'Length of labels: {len(labels)}')

features = np.array(features, dtype='object') # Convert to numpy array
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
# LBPH (Local Binary Patterns Histograms) is a popular algorithm for face recognition.
face_recognizer.train(features, labels)
# Train the face recognizer with the features and labels

face_recognizer.save('Face Recognizer/face_trained.yml')
# Save the trained model to a file
# saving the trained model allows you to use it later without retraining, which saves time and computational resources.

np.save('features.npy', features)
np.save('labels.npy', labels)
# Saving the features and labels as numpy arrays for future use
