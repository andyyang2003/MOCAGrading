#mnist keras

import os
import cv2b.,gv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import cv2
import pytesseract
import numpy as np

# Load the image

# Function to show image:
def show(img):
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.imshow('image',img)
    cv2.waitKey(0)

img = cv2.imread('C:/Users/Andy/Desktop/moca training/clocks/11341809851277719.png', cv2.IMREAD_GRAYSCALE)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img1 = img.copy()
img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

th = cv2.threshold(img_g,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
edged = cv2.Canny(gray, 30, 200)
        cv2.waitKey(0)
        blured = cv2.blur(gray, (5,5), 0)    
        img_thresh = cv2.adaptiveThreshold(blured, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
        threshed = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, rect_kernel)

        contours, hierarchy = cv2.findContours(edged,
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
for contour in contours:    
    if cv2.contourArea(contour) > 170:
        [X, Y, W, H] = cv2.boundingRect(contour)
        cv2.rectangle(img1, (X, Y), (X + W, Y + H), (0,0,255), 2)
        Cropped0 = th[Y - 2:Y + H +2, X - 2:X + W + 2]
        show(Cropped0)