import numpy as np
import PIL
import tensorflow as tf
import pathlib
import cv2 as cv

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

already_trained = 1

if already_trained == 0:
    #import data
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos.tar', origin=dataset_url, extract=True)
    data_dir = pathlib.Path(data_dir).with_suffix('')

    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)

    #create data set
    batch_size = 32
    img_height = 180
    img_width = 180

    #crete train set
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    #create test set
    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    #set classes
    class_names = train_ds.class_names
    print(class_names)
    num_classes = len(class_names)

    #set parameters for best data fetching storage
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

    #normalize data
    normalization_layer = layers.Rescaling(1./255)

    #create augmentation layer
    data_aug = keras.Sequential([
        layers.RandomFlip("horizontal",
                          input_shape=(img_height,img_width,3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])


    model = Sequential([
        data_aug,
        layers.Rescaling(1./255, input_shape=(img_height,img_width, 3)),
        layers.Conv2D(16,3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32,3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64,3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, name="outputs")
    ])

    #compile model
    model.compile(optimizer='adam',
                  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    #train model
    epochs = 15
    history = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs=epochs
    )

    #save model
    #model.save_weights('my_checkpoints')
    model.save('flower_model.keras')
elif already_trained == 1:
    model = tf.keras.models.load_model('flower_model.keras')
    class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

#Infer from new picture

#create video object.
vid = cv.VideoCapture(0)

while vid.isOpened():
    # find frame
    ret, img = vid.read()
    dim = img.shape
    vid_width = dim[0]
    vid_height = dim[1]
    font = cv.FONT_HERSHEY_SIMPLEX


    #process and predict
    resized_image = cv.resize(img,(180,180))
    img_array = tf.keras.utils.img_to_array(resized_image)
    img_array = tf.expand_dims(img_array, 0)

    #predict flower
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    guess_class = class_names[np.argmax(score)]
    print(guess_class)

    #visualize results
    cv.rectangle(resized_image,(0,0),(int(vid_height/6),int(vid_width/10)),(255,0,0),-1)
    cv.putText(resized_image, guess_class, (0,30),font,1,(255,255,255),2,cv.LINE_AA)
    cv.imshow('flower', resized_image)

#   wait for escape key to exit app
    k = cv.waitKey(50) & 0xFF
    if k == 27:
        break

#ensure video and recording are ended properly.
vid.release()
cv.destroyAllWindows()
