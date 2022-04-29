# -*- coding: utf-8 -*-
"""
@author: ANGIE GÓMEZ
"""

import os
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Cargar archivo modelo h5
model = load_model("./cnn_aug_v1.h5")
model.summary()  #Output el estado del parámetro de cada capa del modelo

 
SIZE = 150

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
    return opor

def get_inputs(src=[]):
    pre_x = []
    for s in src:
        image = cv2.imread(s)
        img_seg = segmentation(image)
        img_seg = Image.fromarray(img_seg, 'RGB')
        img_seg = img_seg.resize((SIZE, SIZE)) 
        pre_x.append(np.array(img_seg))
    return pre_x

 #La imagen a predecir se guarda aquí
predict_dir = '.\\'+input("Introduzca la ruta de la imagen que se va a predecir:")

test = os.listdir(predict_dir)
print(test)

 # Cree una nueva lista para guardar la dirección de la imagen predicha
images = []
 #Obtenga la dirección de cada imagen y guárdela en la lista de imágenes
for testpath in test:    # Ciclo para obtener las imágenes que deben probarse en la ruta de prueba
    for fn in os.listdir(os.path.join(predict_dir, testpath)):
        if fn.endswith('jpg'):
            fd = os.path.join(predict_dir, testpath, fn)
            print(fd)
            images.append(fd)
            
pre_x = get_inputs(images)
print(len(pre_x))

#predicción
n=5  #Seleccionar index de la imágen para pruebas
img = pre_x[n]
plt.imshow(img)
input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
print("The prediction for this image is: ", model.predict(input_img))
# 1 is healthy
# 0 is late_blight