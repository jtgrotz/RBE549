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
IMG_SIZE = (299, 299)

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

#plot example images
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

plt.show()

#buffer images for performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

#load the pretrained model and freeze it
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE,
                                               include_top=False,

                                               weights='imagenet')

base_model.summary()

#freeze the initial layer
base_model.trainable = False

#image preprocessing function from model
pre_processing = tf.keras.applications.inception_v3.preprocess_input

#create data augmentation layers
translation_range = (-0.15, 0.15)
scale_range = (-0.1,0.1)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomTranslation(height_factor=translation_range, width_factor=translation_range),
    #tf.keras.layers.RandomTranslation(height_factor=translation_range),
    tf.keras.layers.RandomZoom(height_factor=scale_range, width_factor=scale_range),
    #tf.keras.layers.RandomZoom(width_factor=scale_range),
])

#create new layers
classes = len(class_names)
#pooling layer to reduce 8,8,2048 space to single dimension
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

#new dense layer with 5 outputs for predictions
dense_layer = tf.keras.layers.Dense(classes)

#build the new model
inputs =  tf.keras.Input(shape=(299,299,3))
x = data_augmentation(inputs)
x = pre_processing(x)
x = base_model(x, training=False)
x = global_average_layer(x)
outputs = dense_layer(x)
model = tf.keras.Model(inputs, outputs)

model.summary()

#setup model and compile
epochs = 6
initial_learning_rate = 0.0003

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#train
loss0, accuracy0 = model.evaluate(validation_dataset)

#showcase initial results
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

#train new modified model
history = model.fit(train_dataset,
                    epochs=epochs,
                    validation_data=validation_dataset)

#plotting results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
#plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
#plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


# Retrieve a batch of images from the test set
image_batch, label_batch = validation_dataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch)


#Softmax the outputs to be able to tell what class has the highest probability
predictions = tf.nn.softmax(predictions)

plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].astype("uint8"))
  prediction = np.argmax(predictions[i])
  plt.title(class_names[prediction])
  plt.axis("off")

plt.show()
