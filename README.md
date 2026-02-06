# OpenCV Tutorial & Mini Projects 🚀

This repository documents my **OpenCV learning journey**, starting from **core concepts** to **mini-projects** like **Face Detection, Face Recognition, and basic Emotion Detection**.

It is organized step-by-step and focuses on **hands-on practice using Python and OpenCV**.



## 📂 Folder Structure


OpenCv Tutorial/
│
├── 1Basics/
│   ├── 4EssentialFunctions.py
│   ├── 5ImageTransformation.py
│   ├── 6ContourDetection.py
│   ├── 7ColorSpaces.py
│   ├── 8ColorChannels.py
│   ├── 9SmoothingNBluring.py
│   ├── 10Bitwise.py
│   ├── 11Masking.py
│   ├── 12Histograms.py
│   ├── 13Thresholding.py
│   └── 14EdgeDetection.py
│
├── Emotions Detector/
│   ├── 1.py
│   ├── 2.py
│   └── 3.py
│
├── Face Recognizer/
│   ├── 1Face_Detect.py
│   ├── 2Face_Train.py
│   ├── 3Face_Recognition.py
│   ├── haar_faces.xml
│   ├── face_trained.yml
│   ├── features.npy
│   └── labels.npy
│
├── Faces_Train/
├── Faces_For_Testing/
├── Photos/
├── Videos/
└── README.md




## 🧠 What I Learned

### 🔹 OpenCV Basics

* Reading images and videos
* Essential image operations
* Image transformations
* Contour detection
* Color spaces and channels
* Smoothing and blurring
* Bitwise operations
* Masking
* Histograms
* Thresholding
* Edge detection

### 🔹 Face Detection

* Haar Cascade classifier
* Face detection in images and videos
* Drawing bounding boxes

### 🔹 Face Recognition

* Training a face recognizer
* LBPH (Local Binary Pattern Histogram)
* Saving and loading trained models
* Real-time face recognition

### 🔹 Emotion Detection (Basic)

* Facial feature extraction
* Basic emotion prediction pipeline (learning stage)



## ⚙️ Requirements

* Python 3.x
* OpenCV (contrib version)
* NumPy



Install dependencies:

pip install opencv-contrib-python numpy



## ▶️ How to Run

### Run a basic script
python 1Basics/4EssentialFunctions.py


### Face Detection
python "Face Recognizer/1Face_Detect.py"


### Train Face Recognition Model
python "Face Recognizer/2Face_Train.py"


### Face Recognition
python "Face Recognizer/3Face_Recognition.py"




## 📌 Notes

* Haar cascade file (`haar_faces.xml`) is required for face detection.
* Trained model files (`.yml`, `.npy`) are generated after training.
* Large video files are tracked using **Git LFS**.



## 🎯 Purpose of This Repository

* To document my **OpenCV learning journey**
* To build a strong **foundation in Computer Vision**
* To serve as a **reference for future projects**



## ⭐ Future Improvements

* Improve emotion detection accuracy
* Add deep learning–based face recognition
* Add real-time webcam emotion analysis



## 🤝 Contributions

This is a personal learning repository, but suggestions and improvements are welcome.



## 📜 License

This project is for **educational purposes**.


