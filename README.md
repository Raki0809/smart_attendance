# Smart Attendance System

This project implements a Smart Attendance System using face detection and recognition techniques. It was developed and tested using a Raspberry Pi 4 for real-time attendance tracking.

## Overview

The system captures images, trains a model, and detects faces to automatically mark attendance. It eliminates manual attendance processes and improves efficiency.

## Features

* Face detection using OpenCV
* Image capture and dataset creation
* Model training for recognition
* Automated attendance marking

## Hardware Used

* Raspberry Pi 4
* USB Webcam

## Project Files

* `capture.py` → Captures images for dataset
* `train.py` → Trains the face recognition model
* `attendance.py` → Detects faces and marks attendance
* `face_model.xml` → Face detection model
* `model.yml` → Trained model file

## Note

The dataset is not included in this repository due to size limitations.
Please create a `Dataset/` folder and add images before running the project.

## Technologies Used

* Python
* OpenCV
* Raspberry Pi imager(Raspberry pi software)

## Purpose

This project demonstrates a practical application of computer vision in building an automated attendance system using embedded hardware.
