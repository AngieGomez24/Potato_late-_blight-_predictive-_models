# -*- coding: utf-8 -*-
"""
@author: ANGIE GÓMEZ
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2, os

def read_images():
    path_dir = os.getcwd()
    path_images = os.path.join(path_dir, 'images')
    full_path_images = []
    images_name = os.listdir(path_images)
    for image in images_name:
        full_path_images.append(os.path.join(path_images, image))
    return full_path_images


def segmentation(img):
    #hsv
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    th_low_green = np.array([30, 30, 0])
    th_high_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(img_hsv, th_low_green, th_high_green)
    img_mask = cv2.bitwise_and(img, img, mask = green_mask)
    th_low_brown = np.array([9, 50, 50])
    th_high_brown = np.array([25, 255, 200])
    brown_mask = cv2.inRange(img_hsv, th_low_brown, th_high_brown)
    new_img = cv2.bitwise_and(img, img, mask = brown_mask)
    opor = cv2.bitwise_or(img_mask,new_img)

    
    return [opor,green_mask]
    
def metadata(image):
    path_dir = os.getcwd()
    path_images = os.path.join(path_dir, image)

    image_data = os.path.split(path_images)
    image_name = os.path.splitext(image_data[1])

    metadata = {
      "image": image_name[0],
      "format": image_name[1]
    }
        
    return metadata

def processedDataImages():
    images = read_images()
    n = -1
    for image in images:
        n = n + 1
        img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        width = 256
        height = 256
        dim = (width, height)
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        seg = segmentation(resized)
        cv2.imwrite('./without_background/'+metadata(image)['image']+'.JPG',seg[0])




processedDataImages()