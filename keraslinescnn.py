# tf.keras.utils.image
import numpy as np
import tensorflow as tf # api for building data set when running program
from keras.models import Sequential # api for building model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D # api for building layers
from keras.metrics import Precision, Recall, BinaryAccuracy
from keras.models import load_model
import os
import cv2
import matplotlib.pyplot as plt
gpus = tf.config.experimental.list_physical_devices('CPU')
data = tf.keras.utils.image_dataset_from_directory('C:/Users/Andy/Desktop/moca training/lines/lines', labels="inferred") # api that loads data into dataset while also applying image filters
data = data.map(lambda x, y: (x/255, y)) # normalize the data while 
data_iterator = data.as_numpy_iterator() # allows us to iterate through the dataset
batch = data_iterator.next() # get the next batch of data
#shows what cube value is 
""" fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, img, in enumerate(batch[0][:4]):
    ax[idx].imshow(img)a
    ax[idx].title.set_text(batch[1][idx]) # 0 IF CUBE, 1 IF NOT CUBE
plt.show() """
train_size = int(len(data) * 0.7) # 70% training
valid_size = int(len(data) * 0.2)+1 # 20% validation -> train size and valid size is part of training phase
test_size = int(len(data) * 0.1)+1 # 10% testing

train = data.take(train_size) # take the first 70% of the data
val = data.skip(train_size).take(valid_size) # skip the first 70% and only take valid part of the data
test = data.skip(train_size+valid_size).take(test_size) # take the first 20% of the remaining data

model = Sequential()
# 32 filters, 3x3 kernel size, relu activation function, input shape of 256x256x3
# relu y = x x>0, y = 0 x<= 0
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(256, 256, 3))) 
model.add(MaxPooling2D()) # max pooling downsamples the image to reduce computation

model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten()) # convert to single dimension

model.add(Dense(1, activation='sigmoid')) # sigmoid activation function for binary classification

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy']) # adam optimizer, binary crossentropy loss function, accuracy metric

model.summary() # prints layers of model, representation of what model does to our data

logdir = 'logs'
#specific logging + checkpoint
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir) # tensorboard callback to visualize training

# train and fit model
#epoch = 1 run of trainirng set

history = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback]) # train model on training data, validate on validation data

""" 
fig = plt.figure()
plt.plot(history.history['loss'], color='red', label='loss')
plt.plot(history.history['val_loss'], color='blue', label='val_loss')
plt.legend(loc="upper left")
plt.show() """

# metrics of accuracy
precision = Precision()
recall = Recall() # recall of predictions wrt labels
accuracy = BinaryAccuracy()  
for b in test.as_numpy_iterator():
    x, y = b
    yhat = model.predict(x)
    precision.update_state(y, yhat)
    recall.update_state(y, yhat)
    accuracy.update_state(y, yhat)
print(f'Pre{precision.result().numpy()}, Re{recall.result().numpy()}, Acc{accuracy.result().numpy()}') # print precision, recall, accuracy

#img = cv2.imread('C:/Users/Andy/Desktop/moca training/ImNotSure/4367179142755331221_copy.png')
#img = cv2.imread('cube.png')

model.save(os.path.join('models','lines.h5'))