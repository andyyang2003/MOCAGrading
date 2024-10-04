import numpy as np
import tensorflow as tf
from keras.models import load_model
import os
import cv2
from tkinter import *
from PIL import ImageTk, Image

# Load the pre-trained model
model = load_model('C:/Users/Andy/Desktop/moca training/models/lines.h5')

# Paths for input and output folders
input_folder = 'C:/Users/Andy/Desktop/moca training/lines/notused/'
output_folder = 'C:/Users/Andy/Desktop/moca training/lines/notusedbutgraded/'
base_image_path = 'C:/Users/Andy/Desktop/moca training/linetestbyandy.png'  # Base image path
os.makedirs(output_folder, exist_ok=True)

# Get a list of all images in the folder
image_list = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg'))]
current_image_index = 0  # Start with the first image

# Function to predict pass or fail for an image
def predict_image(image_path):
    img = cv2.imread(image_path)
    resized = tf.image.resize(img, (256, 256))
    yhat = model.predict(np.expand_dims(resized/255, axis=0))
    return 'Pass' if yhat[0][0] >= 0.4 else 'Fail', yhat[0][0]

# Function to update the displayed image and result
def update_image(index):
    # Display current image
    img_path = os.path.join(input_folder, image_list[index])
    img = Image.open(img_path)
    img = img.resize((300, 300), Image.LANCZOS)
    img_tk = ImageTk.PhotoImage(img)

    # Display base image
    base_img = Image.open(base_image_path)
    base_img = base_img.resize((300, 300), Image.LANCZOS)
    base_img_tk = ImageTk.PhotoImage(base_img)

    # Update the UI components (file name and images)
    label_image.config(image=img_tk)
    label_image.image = img_tk  # Keep a reference to avoid garbage collection
    
    label_base_image.config(image=base_img_tk)
    label_base_image.image = base_img_tk  # Keep a reference to avoid garbage collection

    # Update labels for file names
    label_current_filename.config(text=f"Current Image: {os.path.basename(img_path)}")
    label_base_filename.config(text=f"Base Image: {os.path.basename(base_image_path)}")

    # Get prediction and update result label
    result, score = predict_image(img_path)
    label_result.config(text=f"{result} ({score:.2f})")

# Function for next image
def next_image():
    global current_image_index
    if current_image_index < len(image_list) - 1:
        current_image_index += 1
    update_image(current_image_index)

# Function for previous image
def prev_image():
    global current_image_index
    if current_image_index > 0:
        current_image_index -= 1
    update_image(current_image_index)

# Set up the Tkinter window
window = Tk()
window.title("Image Prediction Viewer")

# Labels for file names
label_base_filename = Label(window, text="Base Image:", font=("Arial", 14))
label_base_filename.grid(row=0, column=0)

label_current_filename = Label(window, text="Current Image:", font=("Arial", 14))
label_current_filename.grid(row=0, column=1)

# Display area for the base image
label_base_image = Label(window)
label_base_image.grid(row=1, column=0, padx=10, pady=10)

# Display area for the current image
label_image = Label(window)
label_image.grid(row=1, column=1, padx=10, pady=10)

# Display area for the result (Pass/Fail)
label_result = Label(window, text="", font=("Arial", 20))
label_result.grid(row=2, column=0, columnspan=2)

# Buttons for navigation
button_prev = Button(window, text="<< Previous", command=prev_image)
button_prev.grid(row=3, column=0, padx=10, pady=10)

button_next = Button(window, text="Next >>", command=next_image)
button_next.grid(row=3, column=1, padx=10, pady=10)

# Start by displaying the first image and the base image
update_image(current_image_index)

# Start the Tkinter event loop
window.mainloop()
