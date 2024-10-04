import numpy as np
import tensorflow as tf # api for building data set when running program
from keras.models import Sequential # api for building model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D # api for building layers
from keras.metrics import Precision, Recall, BinaryAccuracy
from keras.models import load_model
import os
import cv2
import matplotlib.pyplot as plt

model = load_model('C:/Users/Andy/Desktop/moca training/cube/cube_model.h5')
img1 = cv2.imread('C:/Users/Andy/Desktop/moca training/cube/Cube/224133105437936349_copy.png')
img2 = cv2.imread('C:/Users/Andy/Desktop/moca training/cube/Cube/3111c (Custom).PNG')  # Replace with your second image path
folder = 'C:/Users/Andy/Desktop/moca training/cube'
# Resize the images
resized1 = tf.image.resize(img1, (256, 256))
resized2 = tf.image.resize(img2, (256, 256))

# Predict
yhat1 = model.predict(np.expand_dims(resized1/255, axis=0))
yhat2 = model.predict(np.expand_dims(resized2/255, axis=0))

# Plot the images with predictions
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(img1)
axs[0].set_title(f'Cube {yhat1[0][0]:.2f}' if yhat1[0][0] < 0.5 else f'Not Cube {yhat1[0][0]:.2f}')
axs[0].axis('off')

axs[1].imshow(img2)
axs[1].set_title(f'Cube {yhat2[0][0]:.2f}' if yhat2[0][0] < 0.5 else f'Not Cube {yhat2[0][0]:.2f}')
axs[1].axis('off')

plt.show()