import os
import cv2
from ultralytics import YOLO
import tensorflow as tf
import numpy as np
import colorsys
from typing import List, Tuple
from keras.models import load_model
# from src.models import model
from model import CustomScalingLayer, build_model, model

# class SegmentationVisualizer:
#     def __init__(self, lane_model_path: str, object_model_path: str, conf_threshold: float = 0.5):
#         self.lane_model = YOLO(lane_model_path)
#         self.object_model = YOLO(object_model_path)
#         self.conf_threshold = conf_threshold
#         self.lane_classes = self.lane_model.names
#         self.object_classes = self.object_model.names
#         self.object_colors = self._generate_colors(len(self.object_classes))
#         self.lane_color = (144, 238, 144)

#     def _generate_colors(self, num_classes: int) -> List[Tuple[int, int, int]]:
#         colors = []
#         for i in range(num_classes):
#             hue = i / num_classes
#             sat = 0.9
#             val = 0.9
#             rgb = colorsys.hsv_to_rgb(hue, sat, val)
#             color = tuple(int(x * 255) for x in rgb)
#             colors.append(color)
#         return colors

#     def process_image(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         lane_overlay = img.copy()
#         object_overlay = img.copy()

#         # Predict lanes
#         lane_results = self.lane_model.predict(img, conf=self.conf_threshold)
#         lane_overlay = self._apply_lane_results(lane_overlay, lane_results)

#         # Predict objects
#         object_results = self.object_model.predict(img, conf=self.conf_threshold)
#         object_overlay = self._apply_object_results(object_overlay, object_results)

#         return lane_overlay, object_overlay

#     def _apply_lane_results(self, overlay: np.ndarray, results) -> np.ndarray:
#         for result in results:
#             if result.masks is None:
#                 continue
#             for mask in result.masks.xy:
#                 points = np.int32([mask])
#                 cv2.fillPoly(overlay, points, self.lane_color)
#         return overlay

#     def _apply_object_results(self, overlay: np.ndarray, results) -> np.ndarray:
#         for result in results:
#             if result.boxes is None:
#                 continue
#             for box in result.boxes:
#                 class_id = int(box.cls[0])
#                 confidence = float(box.conf[0])
#                 color = self.object_colors[class_id]

#                 # Draw bounding boxes
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

#                 # Add labels
#                 label = f"{self.object_classes[class_id]}: {confidence:.2f}"
#                 cv2.putText(overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#         return overlay


# class SteeringAnglePredictor:
#     def __init__(self, model_path):
#         # Register the CustomScalingLayer during model loading
#         self.model = load_model(model_path, custom_objects={"CustomScalingLayer": CustomScalingLayer})
#         self.smoothed_angle = 0

#     def predict_angle(self, image):
#         preprocessed_image = np.expand_dims(image, axis=0)  # Add batch dimension
#         predicted_radians = self.model.predict(preprocessed_image, verbose=0)[0][0]
#         return predicted_radians * 180.0 / np.pi  # Convert radians to degrees

#     def smooth_angle(self, predicted_angle):
#         if self.smoothed_angle == 0:
#             self.smoothed_angle = predicted_angle
#         else:
#             smoothing_factor = 0.7  # Adjust smoothing factor as needed
#             self.smoothed_angle = smoothing_factor * self.smoothed_angle + \
#                                    (1 - smoothing_factor) * predicted_angle
#         return self.smoothed_angle

#     def close(self):
#         self.session.close()


# def run_self_driving(input_folder, lane_model_path, object_model_path, steering_model_path, steering_image_path):
#     # Get valid image files
#     image_files = [img for img in os.listdir(input_folder) if img.endswith((".jpg", ".png"))]
    
#     # Filter out non-numeric filenames
#     image_files = [img for img in image_files if img.split('.')[0].isdigit()]
    
#     # Sort files numerically
#     image_files.sort(key=lambda x: int(x.split('.')[0]))

#     # Load models
#     visualizer = SegmentationVisualizer(lane_model_path, object_model_path)
#     predictor = SteeringAnglePredictor(steering_model_path)

#     steering_image = cv2.imread(steering_image_path, 0)
#     rows, cols = steering_image.shape

#     for image_file in image_files:
#         image_path = os.path.join(input_folder, image_file)
#         image = cv2.imread(image_path)
#         if image is not None:
#             resized_image = cv2.resize(image[-150:], (200, 66)) / 255.0

#             # Predict steering angle
#             predicted_angle = predictor.predict_angle(resized_image)
#             smoothed_angle = predictor.smooth_angle(predicted_angle)

#             # Lane and object segmentation
#             lane_overlay, object_overlay = visualizer.process_image(image)

#             # Display results
#             combined_overlay = cv2.addWeighted(lane_overlay, 0.6, object_overlay, 0.4, 0)
#             rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
#             rotated_steering_wheel = cv2.warpAffine(steering_image, rotation_matrix, (cols, rows))

#             cv2.imshow('Original Image', image)
#             cv2.imshow('Segmented View', combined_overlay)
#             cv2.imshow('Steering Wheel', rotated_steering_wheel)

#             print(f"Frame: {image_file} | Predicted Angle: {predicted_angle:.2f}° | Smoothed Angle: {smoothed_angle:.2f}°")

#             if cv2.waitKey(10) == ord('q'):
#                 break
#         else:
#             print(f"Failed to load {image_file}")

#     predictor.close()
#     cv2.destroyAllWindows()

# # Usage example
# if __name__ == "__main__":
#     input_folder = "D:\\Code\\Self-Driving\\data\driving_dataset"
#     lane_model_path = "D:\\Code\\Self-Driving\\saved_models\\lane_segmentation_model\\best_lane_detection.pt"
#     object_model_path = "D:\\Code\\Self-Driving\\saved_models\\object_detection_model\\yolo11s-seg.pt"
#     steering_model_path = "D:\\Code\\Self-Driving\\saved_models\\regression_model\\model.h5"
#     steering_image_path = "D:\\Code\\Self-Driving\\data\\steering_wheel_image.jpg"

#     run_self_driving(input_folder, lane_model_path, object_model_path, steering_model_path, steering_image_path)

"""EX CODE"""
# Preprocessing function for images
def preprocess_image(image):
    cropped = image[60:-25, :, :]  # Crop the top and bottom
    resized = cv2.resize(cropped, (200, 66))  # Resize to match model input
    normalized = resized / 255.0  # Normalize pixel values
    return normalized

# Steering angle prediction class
class SteeringAnglePredictor:
    def __init__(self, model_path):
        # Register the custom object when loading the model
        with tf.keras.utils.custom_object_scope({'CustomScalingLayer': CustomScalingLayer}):
            self.model = load_model(model_path)
        self.smoothed_angle = 0

    def predict_angle(self, image):
        preprocessed_image = preprocess_image(image)
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)  # Add batch dimension
        predicted_radians = self.model.predict(preprocessed_image, verbose=0)[0][0]
        return predicted_radians * 180.0 / np.pi  # Convert radians to degrees

    def smooth_angle(self, predicted_angle):
        smoothing_factor = 0.7  # Adjust smoothing factor for smoothness
        if self.smoothed_angle == 0:
            self.smoothed_angle = predicted_angle
        else:
            self.smoothed_angle = smoothing_factor * self.smoothed_angle + (1 - smoothing_factor) * predicted_angle
        return self.smoothed_angle

# Lane and object segmentation visualizer class
class SegmentationVisualizer:
    def __init__(self, lane_model_path: str, object_model_path: str, conf_threshold: float = 0.5):
        self.lane_model = YOLO(lane_model_path)
        self.object_model = YOLO(object_model_path)
        self.conf_threshold = conf_threshold
        self.lane_classes = self.lane_model.names
        self.object_classes = self.object_model.names
        self.object_colors = self._generate_colors(len(self.object_classes))
        self.lane_color = (144, 238, 144)  # Light green for lane markings

    def _generate_colors(self, num_classes: int):
        colors = []
        for i in range(num_classes):
            hue = i / num_classes
            sat = 0.9
            val = 0.9
            rgb = colorsys.hsv_to_rgb(hue, sat, val)
            color = tuple(int(x * 255) for x in rgb)
            colors.append(color)
        return colors

    def process_image(self, img: np.ndarray):
        lane_overlay = img.copy()
        object_overlay = img.copy()

        # Predict lanes
        lane_results = self.lane_model.predict(img, conf=self.conf_threshold)
        lane_overlay = self._apply_lane_results(lane_overlay, lane_results)

        # Predict objects
        object_results = self.object_model.predict(img, conf=self.conf_threshold)
        object_overlay = self._apply_object_results(object_overlay, object_results)

        return lane_overlay, object_overlay

    def _apply_lane_results(self, overlay: np.ndarray, results):
        for result in results:
            if result.masks is None:
                continue
            for mask in result.masks.xy:
                points = np.int32([mask])
                cv2.fillPoly(overlay, points, self.lane_color)
        return overlay

    def _apply_object_results(self, overlay: np.ndarray, results):
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                color = self.object_colors[class_id]

                # Draw bounding boxes
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

                # Add labels
                label = f"{self.object_classes[class_id]}: {confidence:.2f}"
                cv2.putText(overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return overlay

# Driving simulator function
def run_self_driving(input_folder, lane_model_path, object_model_path, steering_model_path, steering_image_path):
    # Load models
    visualizer = SegmentationVisualizer(lane_model_path, object_model_path)
    predictor = SteeringAnglePredictor(steering_model_path)

    steering_image = cv2.imread(steering_image_path, 0)  # Grayscale steering wheel image
    rows, cols = steering_image.shape

    # Process images
    image_files = [img for img in os.listdir(input_folder) if img.endswith((".jpg", ".png"))]
    image_files = [img for img in image_files if img.split('.')[0].isdigit()]  # Exclude non-numeric filenames
    image_files.sort(key=lambda x: int(x.split('.')[0]))

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        full_image = cv2.imread(image_path)
        if full_image is None:
            print(f"Failed to load {image_file}")
            continue

        resized_image = preprocess_image(full_image)
        predicted_angle = predictor.predict_angle(full_image)
        smoothed_angle = predictor.smooth_angle(predicted_angle)

        # Lane and object segmentation
        lane_overlay, object_overlay = visualizer.process_image(full_image)
        combined_overlay = cv2.addWeighted(lane_overlay, 0.6, object_overlay, 0.4, 0)

        # Display results
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle * 3, 1)
        rotated_steering_wheel = cv2.warpAffine(steering_image, rotation_matrix, (cols, rows))

        cv2.imshow("Driving Frame", full_image)
        cv2.imshow("Segmented View", combined_overlay)
        cv2.imshow("Steering Wheel", rotated_steering_wheel)

        print(f"Frame {image_file}: Predicted Angle: {predicted_angle:.2f}° | Smoothed Angle: {smoothed_angle:.2f}°")
        if cv2.waitKey(10) == ord('q'):
            break

    cv2.destroyAllWindows()

# Main function
if __name__ == "__main__":
    input_folder = "D:\\Code\\Self-Driving\\data\\driving_dataset"
    lane_model_path = "D:\\Code\\Self-Driving\\saved_models\\lane_segmentation_model\\best_lane_detection.pt"
    object_model_path = "D:\\Code\\Self-Driving\\saved_models\\object_detection_model\\yolo11s-seg.pt"
    steering_model_path = "D:\\Code\\Self-Driving\\saved_models\\regression_model\\model.h5"
    steering_image_path = "D:\\Code\\Self-Driving\\data\\steering_wheel_image.jpg"

    try:
        run_self_driving(input_folder, lane_model_path, object_model_path, steering_model_path, steering_image_path)
    except Exception as e:
        print(f"Error: {e}")
