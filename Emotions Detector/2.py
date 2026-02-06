import cv2 as cv
from deepface import DeepFace
from collections import deque
import time

# Use built-in cascade to avoid path issues
cascade = cv.CascadeClassifier('Face Recognizer/harr_faces.xml')

cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Cannot open camera. Close other apps using camera and try again.")

process_every_n_frames = 3   # analyze every N frames to reduce lag
frame_count = 0
last_pred = None
pred_history = deque(maxlen=7)   # smoothing buffer (majority vote)
t0 = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_count += 1
    display_frame = frame.copy()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)  # tweak minNeighbors/scaleFactor if needed

    # Draw boxes for all detected faces
    for (x, y, w, h) in faces:
        cv.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Only run DeepFace occasionally and on the largest face ROI for stability
    if frame_count % process_every_n_frames == 0 and len(faces) > 0:
        try:
            # choose the largest face
            faces_sorted = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
            x, y, w, h = faces_sorted[0]

            # Expand box slightly (optional) but keep in image bounds
            pad = int(0.1 * w)
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
            face_roi = frame[y1:y2, x1:x2]

            # Optional preprocessing: resize and convert to RGB
            face_rgb = cv.cvtColor(cv.resize(face_roi, (224, 224)), cv.COLOR_BGR2RGB)

            # analyze only the ROI; try different detector_backends if you want improved accuracy
            preds = DeepFace.analyze(face_rgb,
                                     actions=['emotion'],
                                     enforce_detection=False,     # False avoids exceptions if roi imperfect
                                     detector_backend='opencv')   # try 'retinaface' or 'mtcnn' if available

            # DeepFace returns a dict for single image, list for batch — handle both
            if isinstance(preds, list):
                emotion = preds[0]['dominant_emotion']
            else:
                emotion = preds['dominant_emotion']

            pred_history.append(emotion)  # smoothing
            # majority vote
            if len(pred_history) > 0:
                last_pred = max(set(pred_history), key=pred_history.count)

        except Exception as e:
            # Print error for debugging (remove suppression while debugging)
            print("DeepFace error:", str(e))
            last_pred = None

    # Put text near the largest face if available, else top-left
    if last_pred and len(faces) > 0:
        x, y, w, h = faces_sorted[0]
        cv.putText(display_frame, last_pred, (x, max(20, y - 10)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    elif last_pred:
        cv.putText(display_frame, last_pred, (20, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # FPS display
    t1 = time.time()
    fps = 1.0 / (t1 - t0 + 1e-6)
    t0 = t1
    cv.putText(display_frame, f"FPS: {fps:.1f}", (20, 80), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv.imshow("Emotion Detection", display_frame)

    if cv.waitKey(1) & 0xFF == ord('d'):
        break

cap.release()
cv.destroyAllWindows()
