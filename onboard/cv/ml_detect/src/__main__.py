from __future__ import absolute_import, division, print_function
import os
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.keras.backend as K
from tensorflow.keras import layers
import numpy as np
import PIL.Image as Image
import asyncio
from rover_common.aiohelper import run_coroutines
from rover_msgs import CircleImage, ConfidenceScore

lcm_ = aiolcm.AsyncLCM()

classifier_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2" 
#@param {type:"string"}

def classifier(x):
  classifier_module = hub.Module(classifier_url)
  return classifier_module(x)
  
IMAGE_SIZE = hub.get_expected_image_size(hub.Module(classifier_url))
classifier_layer = layers.Lambda(classifier, input_shape = IMAGE_SIZE+[3])
classifier_model = tf.keras.Sequential([classifier_layer])

sess = K.get_session()
sess.run(tf.global_variables_initializer())

def classify(channel,msg):
  '''
  Takes in an image and return the confidence score
  of the image being a tennis ball
  '''
  img = CircleImage.decode(msg).img
  img = np.array(img) / 255.0

  result = classifier_model.predict(img[np.newaxis, ...])

  # predicted_class = np.argmax(result[0], axis=-1)

  # result_sorted = np.argsort(result[0])

  # labels_path = tf.keras.utils.get_file('ImageNetLabels.txt',
  #   'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
  # imagenet_labels = np.array(open(labels_path).read().splitlines())

  # predicted_class_name = imagenet_labels[predicted_class]

  # for i in range(-1,-6,-1):
  #   print("%s: %.4f" % (imagenet_labels[result_sorted[i]], result[0][result_sorted[i]]/10))

  confidence_score = (result[0][853] / 10.0)
  ml_msg = ConfidenceScore()
  ConfidenceScore.score = confidence_score
  lcm_.publish("/ml_score", ml_msg.encode())

def main():
  lcm_.subscribe("/ml_tennisball", classify)
  run_coroutines(lcm_.loop())