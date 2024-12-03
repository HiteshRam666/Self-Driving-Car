import cv2
import random
import numpy as np

xs = []
ys = []

# Points to the end of the last batch 
train_batch_pointer = 0
val_batch_pointer = 0

# Reads data.txt, where each line contains an image filename and the corresponding steering angle.
with open("D:/Code/Self-Driving/data/driving_dataset/data.txt") as f:
    for line in f:
        image_path = "D:/Code/Self-Driving/data/driving_dataset/" + line.split()[0]  # Image paths
        steering_angle = float(line.split()[1]) * 3.14159265 / 180  # Convert to radians
        xs.append(image_path)
        ys.append(steering_angle)

# Get number of images
num_images = len(xs)

# Shuffle the list of images  
c = list(zip(xs, ys))  # Combine image paths and steering angles
random.shuffle(c)  # Shuffle the dataset
xs, ys = zip(*c)  # Unzip shuffled data back into separate lists

# Splitting the Dataset
train_xs = xs[:int(len(xs) * 0.8)]
train_ys = ys[:int(len(xs) * 0.8)]

val_xs = xs[-int(len(xs) * 0.2):]
val_ys = ys[-int(len(xs) * 0.2):]

num_train_images = len(train_xs)
num_val_images = len(val_xs)

def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []
    
    for i in range(batch_size):
        image_path = train_xs[(train_batch_pointer + i) % num_train_images]
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"Warning: Failed to load image {image_path}")
            continue  # Skip this image if it can't be loaded
        
        # Crop, resize and normalize
        img_resized = cv2.resize(img[-150:], (200, 66)) / 255.0
        x_out.append(img_resized)
        
        y_out.append([train_ys[(train_batch_pointer + i) % num_train_images]])

    train_batch_pointer += batch_size
    return np.array(x_out), np.array(y_out)

def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []
    
    for i in range(batch_size):
        image_path = val_xs[(val_batch_pointer + i) % num_val_images]
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"Warning: Failed to load image {image_path}")
            continue  # Skip this image if it can't be loaded
        
        img_resized = cv2.resize(img[-150:], (200, 66)) / 255.0
        x_out.append(img_resized)
        
        y_out.append([val_ys[(val_batch_pointer + i) % num_val_images]])

    val_batch_pointer += batch_size
    return np.array(x_out), np.array(y_out)
