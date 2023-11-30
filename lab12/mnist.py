from datetime import datetime

import keras.callbacks
import tensorflow as tf
import time
import tensorboard

#define tensor board callback
logdir = "logs/fit/" + datetime.now().strftime("%Y%m%%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

#my_opt = 'adam'
#my_opt = 'adagrad'
#my_opt = 'RMSprop'
my_opt = 'sgd'

#load data
mnist = tf.keras.datasets.mnist

#set train and test
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#normalize data
x_train, x_test = x_train/255.0, x_test/255.0
x_train = x_train[...,tf.newaxis].astype('float32')
x_test = x_test[...,tf.newaxis].astype('float32')


#set model and layers
model = tf.keras.models.Sequential([
    #optional layers added to test more options
    tf.keras.layers.Conv2D(32,3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    #tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

#create loss function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

#configure the model with optimizer
model.compile(optimizer=my_opt,
              loss=loss_fn,
              metrics=['accuracy'])

#train model
print(time.time())
model.fit(x_train,y_train, epochs=5, callbacks=[tensorboard_callback])
#model.fit(train_ds, epochs=5)
print(time.time())

#evaulate model
model.evaluate(x_test, y_test, verbose=2)





