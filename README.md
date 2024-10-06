# Real-Time-ASL-Detection

This project uses a pre-trained YOLOv5 model to detect and classify hand gestures in real-time via a webcam feed. The goal is to recognize different sign language gestures with high accuracy and display relevant metrics.

## DEMO OF THIS PROJECT
- https://youtu.be/Mt7fet5Ib3Q

## Features
- Real-time sign language detection from live video feed using YOLOv5.
- Displays bounding boxes and confidence scores for detected gestures.
- Calculates class-wise and overall accuracy based on model predictions.
- Visualizes accuracy metrics using bar plots for each class and overall performance.

## Requirements

## Dataset
- you can download the dataset in here : https://public.roboflow.com/object-detection/american-sign-language-letters/1

### Hardware
- **GPU (Recommended)**: While the code can run on CPU, using a GPU is recommended for real-time performance.
- **Webcam**: A camera connected to your machine for live video feed.

### Software
- **Python 3.8+**
- **PyTorch** (with CUDA support for GPU)
- **OpenCV** for video processing
- **Ultralytics YOLO** for detection
- **Matplotlib** for plotting accuracy metrics

### Python Packages
To install the required packages, run:

```bash
pip install torch opencv-python ultralytics matplotlib
```
=========================================

## SETUP
1. Clone this repo to your directory
2. Download YOLOv5 Weights:
   - Download the pre-trained weights for YOLOv5. You can either use the provided pre-trained weights or train your own model.

## RUNNING REAL TIME DETECTION
1. run the code (real_time_test5)
   - the script wil:
   - Capture video from your webcam.
   - Perform detection on each frame.
   - Display the bounding boxes, class labels, and confidence scores on the video.
   - Calculate and visualize class-wise and overall accuracy.
3. Press "q" to stop the program

## OUTPUT 

## TRAIN MODEL
- ![WhatsApp Image 2024-10-06 at 20 17 14_97f49d59](https://github.com/user-attachments/assets/69b0502c-89a8-4d03-8e06-7199ddbb71c8)

## AFTER PROGRAM STOP
- ![WhatsApp Image 2024-10-06 at 20 17 13_76ca0016](https://github.com/user-attachments/assets/608d42ed-ec71-4eae-a01d-6da1302803a2)
- ![WhatsApp Image 2024-10-06 at 20 17 13_eea6a247](https://github.com/user-attachments/assets/06874af2-c912-4007-bb7e-c0d32e9abe61)


