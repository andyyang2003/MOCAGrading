import cv2
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
base_path = "C:/Users/Andy/Desktop/moca training/"
# Load data
data = pd.read_csv('C:/Users/Andy/Desktop/moca training/clocks.csv')
# use feature extraction to create a model using labels from extract_functions
# Feature extraction functions
def extract_circularity(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circularity = 0
    if contours:
        cnt = contours[0]
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            circularity = 4 * np.pi * (area / (perimeter ** 2))
    return circularity

def extract_number_positioning(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    ideal_positions = [(x, y) for x, y in zip(range(12), range(12))]  # Example ideal positions
    detected_positions = []

    for number in range(1, 13):
        template = cv2.imread(f'templates/{number}.png', cv2.IMREAD_GRAYSCALE)
        res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)
        detected_positions.append(max_loc)

    positioning_error = np.mean([np.linalg.norm(np.array(d) - np.array(i)) for d, i in zip(detected_positions, ideal_positions)])
    return positioning_error

def extract_clock_hands(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    angles = []
    if lines is not None:
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta)
            angles.append(angle)

    ideal_angles = [0, 30, 60]
    angle_error = np.mean([min(abs(a - i) for i in ideal_angles) for a in angles])
    return angle_error

# Extract features for all images
features = []
print(data['filename'])
for filename in data['filename']:
    full_path = os.path.join("C:\\Users\\Andy\\Desktop\\moca training\\clocks", filename)
    print(full_path)
    circularity = extract_circularity(full_path)
    number_positioning = extract_number_positioning(full_path)
    clock_hands = extract_clock_hands(full_path)
    features.append([circularity, number_positioning, clock_hands])

# Convert to DataFrame
features_df = pd.DataFrame(features, columns=['circularity', 'number_positioning', 'clock_hands'])

# Prepare data for model training
X = features_df
y = data[['circularity', 'number_positioning', 'clock_hands']]

# Build model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(64, activation='relu'),
    Dense(3)  # Output layer for three labels
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(X, y, epochs=50, batch_size=32)

# Predict
predictions = model.predict(X)
