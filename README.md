# End-to-End Self-Driving Car 🚗

This repository contains the implementation of an NVIDIA-based end-to-end self-driving car model, which predicts steering angles and performs lane detection, segmentation, and full-stack driving tasks. The project showcases a comprehensive pipeline combining advanced computer vision and deep learning models for autonomous driving. 🌟

---

## 📂 Project Structure

### **Directory and File Layout**
1. **📓 notebooks/**
   - Contains Jupyter notebooks for exploratory analysis and visualization.
   - Example: `lane_detection.ipynb` demonstrates the lane detection pipeline using deep learning.

2. **💾 saved_models/**
   - Stores pre-trained models for:
     - Lane detection.
     - Semantic segmentation.
     - Steering wheel regression.

3. **📜 src/inference/**
   - Contains inference scripts for different tasks:
     - `run_steering_wheel.py` 🛞: Predicts the steering angle using the end-to-end driving model.
     - `run_segmentation.py` 🛣️: Performs semantic segmentation for lane and object detection.
     - `run_fsd.py` 🚘: Implements full-stack driving, combining lane detection, segmentation, and steering.

4. **📑 requirements.txt**
   - Lists all dependencies required for this project.

5. **⚙️ setup.py**
   - Script for packaging and installation.

6. **📖 README.md**
   - This documentation file.

---

## ✨ Key Features

### 1. **🛞 Steering Wheel Angle Prediction**
   - Implements the NVIDIA end-to-end deep learning model to predict the steering wheel angle from a single camera image.
   - **Model Details**:
     - Input: 66x200 preprocessed RGB images.
     - Output: Steering angle (in degrees).

### 2. **🛣️ Lane Detection**
   - A deep learning-based lane detection pipeline using semantic segmentation.
   - **Features**:
     - Robust to different road conditions.
     - Outputs lane masks.

### 3. **🚘 Full-Stack Driving**
   - Combines lane detection, steering angle prediction, and segmentation for a comprehensive autonomous driving pipeline.
   - Designed to simulate real-world driving scenarios.

---

## 🚀 Getting Started

### 1. **📋 Prerequisites**
   - Python >= 3.8 🐍
   - TensorFlow/Keras >= 2.11.0 🧠
   - OpenCV >= 4.5 📸
   - NumPy, SciPy, Matplotlib 📊

### 2. **⚙️ Installation**
   - Clone the repository:
     ```bash
     git clone https://github.com/your_username/self-driving-car.git
     cd self-driving-car
     ```
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```

### 3. **🛠️ Project Setup**
   - Ensure the `saved_models/` directory contains the pre-trained models:
     - Lane detection model.
     - Steering wheel regression model.
   - Modify file paths in scripts as needed to point to your datasets and model files.

---

## 📝 How to Use

### 1. **🛣️ Lane Detection**
   - Run the lane detection notebook for a detailed step-by-step guide:
     ```bash
     jupyter notebook notebooks/lane_detection.ipynb
     ```

### 2. **🛞 Steering Angle Prediction**
   - Execute the steering wheel inference script:
     ```bash
     python src/inference/run_steering_wheel.py
     ```
   - This script will:
     - Load the steering regression model.
     - Predict steering angles for a sequence of driving frames.

### 3. **📸 Semantic Segmentation**
   - Perform segmentation using:
     ```bash
     python src/inference/run_segmentation.py
     ```
   - Outputs segmented images with detected lanes and objects.

### 4. **🚘 Full-Stack Driving**
   - Combine all models to simulate autonomous driving:
     ```bash
     python src/inference/run_fsd.py
     ```

---

## 🤖 Models and Methods

### 1. **NVIDIA End-to-End Model**
   - **Architecture**:
     - 5 convolutional layers for feature extraction.
     - Fully connected layers to predict steering angle.
   - Optimized for real-time inference. ⚡

### 2. **Lane Detection Model**
   - Uses U-Net-based architecture for pixel-wise lane segmentation.
   - Pretrained on diverse road condition datasets.

### 3. **Full-Stack Driving**
   - Combines lane detection and steering prediction into a cohesive driving framework.

---

## 📊 Results and Visualization

### 1. **🛣️ Lane Detection**
   - Outputs binary masks showing detected lanes.

### 2. **🛞 Steering Wheel Visualization**
   - Simulates the movement of a steering wheel based on predicted angles.

### 3. **🎥 Driving Simulations**
   - Demonstrates real-time driving scenarios with combined pipelines.

---

## 🌟 Future Work

1. **🔗 Integrate Sensor Data:**
   - Fuse LIDAR or radar inputs with vision-based predictions.

2. **🛠️ Real-World Testing:**
   - Test the models on real-world datasets or on a driving simulator (e.g., CARLA).

3. **⚡ Model Optimization:**
   - Enhance model efficiency for edge device deployment.

---

## 🙌 Contributing

Contributions are welcome! Feel free to fork the repository, create a feature branch, and submit a pull request. 🛠️

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 💖 Acknowledgments

- **🚗 [SullyChen Driving Datasets](https://github.com/SullyChen/driving-datasets):** This project uses the dataset from SullyChen, which contains images and corresponding steering wheel angles (in radians). It has been an essential resource for training and testing the model.
- NVIDIA for the End-to-End Self-Driving Car Model concept.
- Open-source datasets and pre-trained models used in this project.

---
