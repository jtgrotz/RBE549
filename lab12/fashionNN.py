import tensorflow as tf
from tensorflow import keras


#import data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#create layers for model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

#compile the model and loss function
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#train model

model.fit(train_images, train_labels, epochs=5)

#test with test images
test_loss, test_acc = model.evaluate(test_images, test_labels)

print(test_loss)
print(test_acc)
