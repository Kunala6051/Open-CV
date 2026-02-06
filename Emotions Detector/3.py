import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Load pretrained emotion model (use FER2013-based)
model = load_model('emotion_model.h5')  # You must have a pre-trained model
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                                  int(bboxC.width * w), int(bboxC.height * h)

            # Crop face
            face = frame[y:y+height, x:x+width]
            if face.size > 0:
                gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (48, 48))
                gray = gray / 255.0
                gray = np.expand_dims(np.expand_dims(gray, -1), 0)

                # Predict emotion
                preds = model.predict(gray)
                emotion = emotion_labels[np.argmax(preds)]

                # Display
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
