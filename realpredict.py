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
img = cv2.imread('C:/Users/Andy/Desktop/moca training/cube/NotCube/3022c.png')
resized = tf.image.resize(img, (256, 256))
yhat = model.predict(np.expand_dims(resized/255, axis=0))
print(yhat)
plt.imshow(img)
plt.title('Cube' if yhat[0][0] < 0.5 else 'Not Cube')
plt.show() 
