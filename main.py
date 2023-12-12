import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import load_img, img_to_array
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Ensure that Pillow is installed
try:
    from PIL import Image
except ImportError:
    raise ImportError("Please install Pillow library: pip install Pillow")

# Function to load images and labels from a directory
def load_images_from_directory(directory, label):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            img_path = os.path.join(directory, filename)
            image = Image.open(img_path).convert('RGB')
            image = image.resize((256, 256))
            image = img_to_array(image)
            images.append(image)
            labels.append(label)
    return np.array(images), np.array(labels)

# Function to load the test set
def load_test_set(directory):
    test_images = []
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            img_path = os.path.join(directory, filename)
            image = load_img(img_path, target_size=(256, 256))
            image = img_to_array(image)
            test_images.append(image)
    return np.array(test_images)

# Variables to store loaded datasets
healthy_images, healthy_labels = None, None
parkinson_images, parkinson_labels = None, None
test_images = None

# Function to load the dataset of a healthy person
def load_healthy_dataset():
    global healthy_images, healthy_labels
    folder_path = filedialog.askdirectory(title="Select Healthy Dataset Folder")
    if folder_path:
        healthy_images, healthy_labels = load_images_from_directory(folder_path, label=0)
        if check_dataset(healthy_images, "Healthy"):
            status_label.config(text="Healthy dataset loaded successfully.")
            result_text.delete(1.0, tk.END)  # Clear previous results
        else:
            status_label.config(text="Invalid Healthy dataset. Please check the data.")

# Function to load the dataset of a person with Parkinson's disease
def load_parkinson_dataset():
    global parkinson_images, parkinson_labels
    folder_path = filedialog.askdirectory(title="Select Parkinson's Dataset Folder")
    if folder_path:
        parkinson_images, parkinson_labels = load_images_from_directory(folder_path, label=1)
        if check_dataset(parkinson_images, "Parkinson's"):
            status_label.config(text="Parkinson's dataset loaded successfully.")
            result_text.delete(1.0, tk.END)  # Clear previous results
        else:
            status_label.config(text="Invalid Parkinson's dataset. Please check the data.")

# Function to load test images
def load_test_images():
    global test_images
    folder_path = filedialog.askdirectory(title="Select Test Dataset Folder")
    if folder_path:
        test_images = load_test_set(folder_path)
        if check_dataset(test_images, "Test"):
            status_label.config(text="Test dataset loaded successfully.")
            result_text.delete(1.0, tk.END)  # Clear previous results
        else:
            status_label.config(text="Invalid Test dataset. Please check the data.")

# Function to check if the loaded dataset is valid
def check_dataset(data, dataset_name):
    if data is None or len(data) == 0:
        status_label.config(text=f"Invalid {dataset_name} dataset. No data found.")
        return False
    elif len(data.shape) != 4 or data.shape[1:] != (256, 256, 3):
        status_label.config(text=f"Invalid {dataset_name} dataset. Incorrect data shape.")
        return False
    return True

# Function to display test images with predictions
def display_test_results(predictions):
    result_frame = tk.Toplevel(root)
    result_frame.title("Test Results")
    result_frame.geometry("400x350")

    for i, (prediction, filename) in enumerate(
            zip(predictions, os.listdir('test'))):
        img_path = os.path.join('test', filename)
        img = Image.open(img_path)
        img.thumbnail((100, 100))  # Resize the image for display
        img_tk = ImageTk.PhotoImage(img)

        label = tk.Label(result_frame, text=f"Image {filename}\nPrediction: {'Parkinsons Disease' if prediction >= 0.5 else 'Healthy'}")
        label.pack()

        img_label = tk.Label(result_frame, image=img_tk)
        img_label.image = img_tk
        img_label.pack()

# Function to train the model and display test results
def train_and_test():
    global healthy_images, healthy_labels, parkinson_images, parkinson_labels, test_images

    if healthy_images is None or parkinson_images is None or test_images is None:
        result_text.delete(1.0, tk.END)  # Clear previous results
        status_label.config(text="Please load all datasets before training and testing.")
        return

    # Combine datasets
    X = np.concatenate((healthy_images, parkinson_images), axis=0)
    y = np.concatenate((healthy_labels, parkinson_labels), axis=0)

    # Print information about the loaded datasets
    status_label.config(
        text=f"Total number of samples: {len(X)}\nNumber of healthy samples: {np.sum(y == 0)}\nNumber of Parkinson's samples: {np.sum(y == 1)}")


    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize data
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)
    X_train_scaled = X_train_scaled.reshape(X_train.shape)
    X_test_scaled = X_test_scaled.reshape(X_test.shape)

    # Build CNN model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_data=(X_test_scaled, y_test))

    # Reshape the test images to the correct shape
    test_images = test_images.reshape(-1, 256, 256, 3)

    # Scale the test images using the same scaler used for training data
    test_images_flat = test_images.reshape(test_images.shape[0], -1)
    test_images_scaled = scaler.transform(test_images_flat)

    # Reshape the data back to 4D
    test_images_scaled = test_images_scaled.reshape(test_images.shape)

    # Get predictions for the test set
    predictions = model.predict(test_images_scaled)

    # Display test results in the main window
    result_text.delete(1.0, tk.END)  # Clear previous results
    for i, (prediction, filename) in enumerate(
            zip(predictions, os.listdir('test'))):
        result_text.insert(tk.END, f"Image {filename}: {'Parkinsons Disease' if prediction >= 0.5 else 'Healthy'}\n")

    # Evaluate accuracy on the test set
    accuracy = model.evaluate(X_test_scaled, y_test)[1]
    result_text.insert(tk.END, f'\nTest Accuracy: {accuracy}')

    # Display test results
    display_test_results(predictions)

# Create GUI window
root = tk.Tk()
root.title("Parkinson's Disease Prediction")
root.geometry("500x350")  # Set constant window size

# Frame to hold buttons in a row
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

# Buttons to load healthy and Parkinson's datasets
load_healthy_button = tk.Button(button_frame, text="Load Healthy Dataset", command=load_healthy_dataset)
load_healthy_button.pack(side=tk.LEFT, padx=10)

load_parkinson_button = tk.Button(button_frame, text="Load Parkinson's Dataset", command=load_parkinson_dataset)
load_parkinson_button.pack(side=tk.LEFT, padx=10)

load_test_button = tk.Button(button_frame, text="Load Test Images", command=load_test_images)
load_test_button.pack(side=tk.LEFT, padx=10)

# Label to display status messages
status_label = tk.Label(root, text="", pady=10)
status_label.pack()

# Text widget to display results
result_text = tk.Text(root, height=10, width=50)
result_text.pack(pady=10)

# Button to initiate training and display test results
train_button = tk.Button(root, text="Train and Test", command=train_and_test)
train_button.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
