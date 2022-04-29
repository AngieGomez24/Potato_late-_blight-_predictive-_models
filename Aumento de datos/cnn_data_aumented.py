# -*- coding: utf-8 -*-
"""
@author: ANGIE GÓMEZ
"""
import cv2, os

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='constant') 

############################

#Guardar imágenes aumentadas
def save_images_augmented(image):
    img = load_img(image)  
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in datagen.flow(x, batch_size=1,  
                          save_to_dir='dataset_augmented/healthy/', save_prefix='augmented', save_format='jpg'):
        i += 1
        if i > 7:
            break

#Leer imágenes
def read_images():
    path_dir = os.getcwd()
    path_images = os.path.join(path_dir, 'dataset_without_background/healthy')
    full_path_images = []
    images_name = os.listdir(path_images)
    for image in images_name:
        full_path_images.append(os.path.join(path_images, image))
    return full_path_images


def dataImages():
    images = read_images()
    for image in images:
        save_images_augmented(image)
        
dataImages()