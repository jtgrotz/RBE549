import zipfile

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf


#import data in a useful way
directory = os.getcwd()
local_zip = directory+'/MerchData.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/')
zip_ref.close()


BATCH_SIZE = 10
IMG_SIZE = (224, 224)

train_dataset = tf.keras.utils.image_dataset_from_directory('/tmp/MerchData/',
                                                            labels="inferred",
                                                            shuffle=True,
                                                            validation_split=0.3,
                                                            subset='training',
                                                            seed=117,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE)


validation_dataset = tf.keras.utils.image_dataset_from_directory('/tmp/MerchData/',
                                                                 labels="inferred",
                                                                 shuffle=True,
                                                                 validation_split=0.3,
                                                                 subset='validation',
                                                                 seed=117,
                                                                 batch_size=BATCH_SIZE,
                                                                 image_size=IMG_SIZE)

class_names = train_dataset.class_names

#buffer images for performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

#load the pretrained model

#base_model = tf.keras.applications.
#freeze the initial layer
#create data augmentation layers
translation_range = (-30., 30.)
scale_range = (-0.1,0.1)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomTranslation(height_factor=translation_range),
    tf.keras.layers.RandomTranslation(width_factor=translation_range),
    tf.keras.layers.RandomZoom(height_factor=scale_range),
    tf.keras.layers.RandomZoom(width_factor=scale_range),
])

#add layer to rescale images to the [-1,1] range
#preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)

#setup model and compile
epochs = 6
initial_learning_rate = 0.0003

#train
#plot results
#test to random images?
