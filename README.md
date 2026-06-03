# 🧠 Convolutional Neural Network – YOLOv8 Object Detection

Object detection training using **YOLOv8** on the **Pascal VOC 2012** dataset.

---

## Overview

This project demonstrates object detection using a convolutional neural network-based YOLOv8 model trained on the Pascal VOC 2012 dataset.

The goal of the project is to detect and classify objects in images by learning bounding box localization and object category prediction.

YOLOv8 was selected because it is a modern real-time object detection architecture that provides high accuracy, efficient training, and fast inference.

---

## Dataset

The project uses the **Pascal VOC 2012** dataset, which is a widely used benchmark dataset for object detection and image recognition tasks.

Pascal VOC 2012 contains annotated images with object categories such as:

* Person
* Car
* Bus
* Bicycle
* Motorbike
* Dog
* Cat
* Bird
* Horse
* Chair
* Bottle
* TV/Monitor
* Sofa
* Train
* Aeroplane

The dataset includes images with bounding box annotations that are used to train the YOLOv8 detection model.

---

## Model

The model is based on **YOLOv8**.

YOLO stands for **You Only Look Once**, meaning the model predicts object classes and bounding boxes in a single forward pass.

The model performs two main tasks:

1. **Object Classification** – identifying what object is present in the image
2. **Object Localization** – predicting the bounding box around the object

---

## Workflow

```text
Pascal VOC 2012 Dataset
        │
        ▼
Annotation Conversion
        │
        ▼
YOLO Format Dataset
        │
        ▼
YOLOv8 Training
        │
        ▼
Model Evaluation
        │
        ▼
Object Detection Results
```

---

## Features

* YOLOv8 object detection
* Pascal VOC 2012 dataset training
* Bounding box prediction
* Multi-class object detection
* Image annotation processing
* Model evaluation
* Detection result visualization

---

## Technology Stack

| Component               | Technology      |
| ----------------------- | --------------- |
| Language                | Python          |
| Deep Learning Framework | PyTorch         |
| Object Detection Model  | YOLOv8          |
| Library                 | Ultralytics     |
| Dataset                 | Pascal VOC 2012 |
| Image Processing        | OpenCV          |
| Visualization           | Matplotlib      |

---

## Installation

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/Convolutional-Neural-Network.git
cd Convolutional-Neural-Network
```

### Install Dependencies

```bash
pip install ultralytics opencv-python matplotlib numpy
```

---

## Training

To train the YOLOv8 model, run:

```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640
```

Explanation:

* `data=data.yaml` – dataset configuration file
* `model=yolov8n.pt` – pretrained YOLOv8 nano model
* `epochs=50` – number of training epochs
* `imgsz=640` – input image size

---

## Evaluation

After training, the model can be evaluated using:

```bash
yolo detect val model=runs/detect/train/weights/best.pt data=data.yaml
```

Evaluation metrics include:

* Precision
* Recall
* mAP@50
* mAP@50-95
* Loss curves

---

## Inference

To run object detection on test images:

```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=images/
```

The detected objects will be displayed with bounding boxes and class labels.

---

## Project Structure

```text
Convolutional-Neural-Network/
│
├── dataset/
├── images/
├── labels/
├── runs/
├── data.yaml
├── train.py
├── predict.py
└── README.md
```

---

## Results

The trained YOLOv8 model is able to detect multiple object classes from the Pascal VOC 2012 dataset and draw bounding boxes around detected objects.

Example outputs may include:

* Detected persons
* Vehicles
* Animals
* Indoor objects
* Transportation objects

---

## Applications

* Object detection
* Image understanding
* Autonomous systems
* Surveillance
* Robotics
* Computer vision education

---

## Author

**Artashes Grigoryan**

National Polytechnic University of Armenia

---

## Purpose

This project was developed for educational purposes to understand convolutional neural networks, object detection pipelines, YOLO architecture, and dataset preparation for computer vision tasks.
