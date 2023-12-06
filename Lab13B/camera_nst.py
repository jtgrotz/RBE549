import os
import cv2 as cv
import keras
import tensorflow as tf
# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time
import functools

#1 if you want to train the model, 0 is you want to load the previously trained model
train_model = 0

#function to convert the network tensor to an image
def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

#function to load an image in the correct dimension
def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

#function for making image the right dimensions for video stream
def convert_image(img):
    max_dim = 512
    #img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

#load in content and style image
content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
style_path = tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

#function show display an image
def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)



#load in VGG19 model for out training
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

print()
for layer in vgg.layers:
  print(layer.name)

#layers chosen for syle and content
content_layers = ['block5_conv2']

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


def vgg_layers(layer_names):
  """ Creates a VGG model that returns a list of intermediate output values."""
  # Load our model. Load pretrained VGG, trained on ImageNet data
  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False

  outputs = [vgg.get_layer(name).output for name in layer_names]

  model = tf.keras.Model([vgg.input], outputs)
  return model

#function to calculate the gram matrix
def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)


#define the model for style
@keras.saving.register_keras_serializable()
class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    super(StyleContentModel, self).__init__()
    self.vgg = vgg_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False

  def call(self, inputs):
    "Expects float input in [0,1]"
    inputs = inputs*255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    outputs = self.vgg(preprocessed_input)
    style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                      outputs[self.num_style_layers:])

    style_outputs = [gram_matrix(style_output)
                     for style_output in style_outputs]

    content_dict = {content_name: value
                    for content_name, value
                    in zip(self.content_layers, content_outputs)}

    style_dict = {style_name: value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}

    return {'content': content_dict, 'style': style_dict}


#create new model and run extractor
if train_model == 1:

    # display chosen images
    content_image = load_img(content_path)
    style_image = load_img(style_path)

    # define the style extractor model
    style_extractor = vgg_layers(style_layers)
    style_outputs = style_extractor(style_image * 255)

    plt.subplot(1, 2, 1)
    imshow(content_image, 'Content Image')

    plt.subplot(1, 2, 2)
    imshow(style_image, 'Style Image')

    extractor = StyleContentModel(style_layers, content_layers)

    results = extractor(tf.constant(content_image))

    #run style transfer modelling
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']
    #initialize image variable
    image = tf.Variable(content_image)

    #keep values between 0 and 1
    def clip_0_1(image):
      return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    #form compiler for model
    opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    #define two losses
    style_weight=1e-2
    content_weight=1e4

    #define loss function
    def style_content_loss(outputs):
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2)
                               for name in style_outputs.keys()])
        style_loss *= style_weight / num_style_layers

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2)
                                 for name in content_outputs.keys()])
        content_loss *= content_weight / num_content_layers
        loss = style_loss + content_loss
        return loss

    #define training step function
    @tf.function()
    def train_step(image):
      total_variation_weight = 30
      with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)
        #additional loss to remove high frequency content
        loss += total_variation_weight*tf.image.total_variation(image)

      grad = tape.gradient(loss, image)
      opt.apply_gradients([(grad, image)])
      image.assign(clip_0_1(image))


    #perform optimization
    import time
    start = time.time()

    epochs = 2
    steps_per_epoch = 25

    step = 0
    for n in range(epochs):
      for m in range(steps_per_epoch):
        step += 1
        train_step(image)
        print(".", end='', flush=True)
      display.clear_output(wait=True)
      display.display(tensor_to_image(image))
      print("Train step: {}".format(step))

    end = time.time()
    print("Total time: {:.1f}".format(end-start))

    extractor.save('styler.keras')

    file_name = 'stylized-image.png'
    tensor_to_image(image).save(file_name)
elif train_model == 0:
    #because the trained model is so slow, this option uses a faster pretrained model for real time image styling
    #test it on another image.

    #load style image here
    style_image = load_img('fractal.jpg')
    #plt.figure()
    #imshow(style_image)
    #plt.show()

    style_on = False

    #import module for faster image styling
    import tensorflow_hub as hub
    hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    # set up camera here
    #create video object
    vid = cv.VideoCapture(0)

    while vid.isOpened():
        #get camera frame
        ret, frame = vid.read()
        dim = frame.shape
        vid_width = dim[0]
        vid_height = dim[1]

        frame = convert_image(frame)
        #if setting is enabled convert image, else don't
        if style_on:
            #convert image to style
            my_image = hub_model(tf.constant(frame), tf.constant(style_image))[0]
            #show styled image
            new_image = np.array(tensor_to_image(my_image))

        else:
            new_image = np.array(frame[0])

        #display image
        cv.imshow('My style', new_image)

        #   wait for escape key to exit app
        k = cv.waitKey(50) & 0xFF
        if k == ord('s'):
            style_on = not style_on
        if k == 27:
            break