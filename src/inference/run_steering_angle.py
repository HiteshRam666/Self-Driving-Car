import os 
import numpy as np
import cv2
from model import model, build_model
from subprocess import call 
from ultralytics import YOLO
import tensorflow as tf
from keras import layers
from keras.layers import Layer
from keras.models import load_model
import time

# class SteeringAnglePredictor:
#     def __init__(self, model_path, input_shape):
#         self.model = self.load_model(model_path, input_shape)
#         self.smoothed_angle = 0

#     @staticmethod
#     def load_model(model_path, input_shape):
#         # Initialize the model using the build_model function
#         model_instance = build_model(input_shape)  # Build the model using the input shape
#         model_instance.load_weights(model_path)  # Load weights into the model
#         return model_instance
    
#     def predict_angle(self, image):
#         # Add batch dimension and normalize
#         image = np.expand_dims(image, axis=0)
#         predicted_radians = self.model.predict(image, verbose=True)[0][0]
#         degrees = predicted_radians * 180.0 / np.pi
#         return degrees
        
#     def smooth_angle(self, predicted_angle):
#         smoothing_factor = 0.7  # Higher value makes it smoother (0 < factor < 1)
#         self.smoothed_angle = (
#             smoothing_factor * self.smoothed_angle + (1 - smoothing_factor) * predicted_angle
#             if self.smoothed_angle != 0
#             else predicted_angle
#         )
#         return self.smoothed_angle



# class DrivingSimulator:
#     def __init__(self, predictor, data_dir, steering_image_path, is_windows=False):
#         self.predictor = predictor
#         self.data_dir = data_dir
#         self.steering_image = cv2.imread(steering_image_path, 0)
#         self.is_windows = is_windows
#         self.rows, self.cols = self.steering_image.shape

#     def start_simulation(self):
#         i = 0
#         while cv2.waitKey(10) != ord('q'):
#             image_path = os.path.join(self.data_dir, f"{i}.jpg")
#             if not os.path.exists(image_path):
#                 print(f"Image {image_path} not found. Ending simulation.")
#                 break

#             full_image = cv2.imread(image_path)
#             resized_image = cv2.resize(full_image[-150:], (200, 66)) / 255.0

#             predicted_angle = self.predictor.predict_angle(resized_image)
#             smoothed_angle = self.predictor.smooth_angle(predicted_angle)

#             if not self.is_windows:
#                 os.system("clear")
#             print(f"Predicted steering angle: {predicted_angle:.2f} degrees")

#             self.display_frames(full_image, smoothed_angle)
#             i += 1

#         cv2.destroyAllWindows()

#     def display_frames(self, full_image, smoothed_angle):
#         cv2.imshow("frame", full_image)
#         rotation_matrix = cv2.getRotationMatrix2D((self.cols / 2, self.rows / 2), -smoothed_angle, 1)
#         rotated_steering_wheel = cv2.warpAffine(self.steering_image, rotation_matrix, (self.cols, self.rows))
#         cv2.imshow("steering wheel", rotated_steering_wheel)

# if __name__ == "__main__":
#     model_path = "D:/Code/Self-Driving/saved_models/regression_model/model.h5"
#     # Ensure this matches your saved weights file
#     data_dir = "D:\Code\Self-Driving\data\driving_dataset"
#     steering_image_path = "D:\Code\Self-Driving\data\steering_wheel_image.jpg "

#     # Determine if running on Windows
#     is_windows = os.name == 'nt'

#     # Define the input shape for the model
#     input_shape = (66, 200, 3)

#     predictor = SteeringAnglePredictor(model_path, input_shape)
#     simulator = DrivingSimulator(predictor, data_dir, steering_image_path, is_windows)

#     try:
#         simulator.start_simulation()
#     except Exception as e:
#         print(f"Error during simulation: {e}")
#     finally:
#         print("Closing simulator.")

""""""

# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# from keras.models import load_model
# from model import CustomScalingLayer, build_model

# # Preprocessing function for images
# def preprocess_image(image):
#     cropped = image[60:-25, :, :]
#     resized = cv2.resize(cropped, (200, 66))
#     normalized = resized / 255.0
#     return normalized

# # Steering angle prediction class
# class SteeringAnglePredictor:
#     def __init__(self, model_path):
#         # Register the custom object when loading the model
#         with tf.keras.utils.custom_object_scope({'CustomScalingLayer': CustomScalingLayer}):
#             self.model = load_model(model_path)
#         self.smoothed_angle = 0

#     def predict_angle(self, image):
#         preprocessed_image = preprocess_image(image)
#         preprocessed_image = np.expand_dims(preprocessed_image, axis=0)  # Add batch dimension
#         predicted_radians = self.model.predict(preprocessed_image, verbose=0)[0][0]
#         return predicted_radians * 180.0 / np.pi  # Convert radians to degrees

#     def smooth_angle(self, predicted_angle):
#         smoothing_factor = 0.7  # Higher value makes it smoother (0 < factor < 1)
#         if self.smoothed_angle == 0:
#             self.smoothed_angle = predicted_angle
#         else:
#             self.smoothed_angle = smoothing_factor * self.smoothed_angle + (1 - smoothing_factor) * predicted_angle
#         return self.smoothed_angle

# # Driving simulator class
# class DrivingSimulator:
#     def __init__(self, predictor, data_dir, steering_image_path):
#         self.predictor = predictor
#         self.data_dir = data_dir
#         self.steering_image = cv2.imread(steering_image_path, 0)  # Load steering wheel image in grayscale
#         self.rows, self.cols = self.steering_image.shape

#     def start_simulation(self):
#         i = 0
#         while cv2.waitKey(1) != ord('q'):  # Refresh every 1 ms (adjustable)
#             image_path = os.path.join(self.data_dir, f"{i}.jpg")
#             if not os.path.exists(image_path):
#                 print(f"Image {image_path} not found. Ending simulation.")
#                 break

#             full_image = cv2.imread(image_path)
#             predicted_angle = self.predictor.predict_angle(full_image)
#             smoothed_angle = self.predictor.smooth_angle(predicted_angle)

#             # Print the steering angles for each frame
#             print(f"Frame {i}: Predicted: {predicted_angle:.2f}° | Smoothed: {smoothed_angle:.2f}°")
#             self.display_frames(full_image, smoothed_angle)
#             i += 1

#         cv2.destroyAllWindows()

#     def display_frames(self, full_image, smoothed_angle):
#         """Display the driving frame and steering wheel movement."""
#         cv2.imshow("Driving Frame", full_image)
        
#         # Scale the angle for better visibility, you can experiment with this factor.
#         scaled_angle = smoothed_angle * 3  # Increase this factor if the movement is too subtle
#         rotation_matrix = cv2.getRotationMatrix2D((self.cols / 2, self.rows / 2), -scaled_angle, 1)
#         rotated_wheel = cv2.warpAffine(self.steering_image, rotation_matrix, (self.cols, self.rows))
#         cv2.imshow("Steering Wheel", rotated_wheel)

# if __name__ == "__main__":
#     model_path = "D:\\Code\\Self-Driving\\saved_models\\regression_model\\model.h5"
#     data_dir = "D:\\Code\\Self-Driving\\data\\driving_dataset"
#     steering_image_path = "D:\\Code\\Self-Driving\\data\\steering_wheel_image.jpg"

#     # Initialize the predictor and simulator
#     predictor = SteeringAnglePredictor(model_path)
#     simulator = DrivingSimulator(predictor, data_dir, steering_image_path)

#     try:
#         simulator.start_simulation()
#     except Exception as e:
#         print(f"Error during simulation: {e}")
#     finally:
#         print("Simulation ended.")


import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from model import CustomScalingLayer, build_model

# Preprocessing function for images
def preprocess_image(image):
    cropped = image[60:-25, :, :]
    resized = cv2.resize(cropped, (200, 66))
    normalized = resized / 255.0
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
        smoothing_factor = 0.7  # Higher value makes it smoother (0 < factor < 1)
        if self.smoothed_angle == 0:
            self.smoothed_angle = predicted_angle
        else:
            self.smoothed_angle = smoothing_factor * self.smoothed_angle + (1 - smoothing_factor) * predicted_angle
        return self.smoothed_angle

# Driving simulator class
class DrivingSimulator:
    def __init__(self, predictor, data_dir, steering_image_path, start_index=0):
        self.predictor = predictor
        self.data_dir = data_dir
        self.steering_image = cv2.imread(steering_image_path, 0)  # Load steering wheel image in grayscale
        self.rows, self.cols = self.steering_image.shape
        self.start_index = start_index  # Start from this index

    def start_simulation(self):
        i = self.start_index  # Start from the specified index
        while cv2.waitKey(1) != ord('q'):  # Refresh every 1 ms (adjustable)
            image_path = os.path.join(self.data_dir, f"{i}.jpg")
            if not os.path.exists(image_path):
                print(f"Image {image_path} not found. Ending simulation.")
                break

            full_image = cv2.imread(image_path)
            predicted_angle = self.predictor.predict_angle(full_image)
            smoothed_angle = self.predictor.smooth_angle(predicted_angle)

            # Print the steering angles for each frame
            print(f"Frame {i}: Predicted: {predicted_angle:.2f}° | Smoothed: {smoothed_angle:.2f}°")
            self.display_frames(full_image, smoothed_angle)
            i += 1

        cv2.destroyAllWindows()

    def display_frames(self, full_image, smoothed_angle):
        """Display the driving frame and steering wheel movement."""
        cv2.imshow("Driving Frame", full_image)
        
        # Scale the angle for better visibility, you can experiment with this factor.
        scaled_angle = smoothed_angle * 3  # Increase this factor if the movement is too subtle
        rotation_matrix = cv2.getRotationMatrix2D((self.cols / 2, self.rows / 2), -scaled_angle, 1)
        rotated_wheel = cv2.warpAffine(self.steering_image, rotation_matrix, (self.cols, self.rows))
        cv2.imshow("Steering Wheel", rotated_wheel)

if __name__ == "__main__":
    model_path = "D:\\Code\\Self-Driving\\saved_models\\regression_model\\model.h5"
    data_dir = "D:\\Code\\Self-Driving\\data\\driving_dataset"
    steering_image_path = "D:\\Code\\Self-Driving\\data\\steering_wheel_image.jpg"
    start_index = 951  # Specify the starting image index

    # Initialize the predictor and simulator
    predictor = SteeringAnglePredictor(model_path)
    simulator = DrivingSimulator(predictor, data_dir, steering_image_path, start_index=start_index)

    try:
        simulator.start_simulation()
    except Exception as e:
        print(f"Error during simulation: {e}")
    finally:
        print("Simulation ended.")

