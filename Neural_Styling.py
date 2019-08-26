
# import libraries.
# using tensorflow as backend.
import os
import cv2
import keras 
import numpy as np 
import tensorflow as tf 
from keras import models
from keras.models import Sequential
from keras.applications import VGG19
from keras.layers import Dense, Conv2D, MaxPooling2D

# image files path.
content_img = "content_img.jpg"
style_img = "style_img.jpg"

def load_image(image_path):
	img = load_img(image_path)
	img = VGG19.preprocess_input(img)
	return img

# reading content and style images.
content_img = cv2.resize(cv2.imread(content_img),(int(244),int(244)))
style_img = cv2.resize(cv2.imread(style_img),(int(244),int(244)))

# show images:-
plt.imshow(content_img)
plt.show()
plt.imshow(style_img)
plt.show()

# layers for extracting feature maps:-

#one layer for content_feature_map is needed>>
content_layer = ["block5_conv2"]

#more than one middle layers are needed>>
style_layers =[ 'block1_conv1',
				'block2_conv1',
				'block3_conv1', 
				'block4_conv1', 
				'block5_conv1'
				]

# using VGG19 pretrained model for extracting feature_maps:-
model = VGG19(weights = "imagenet",
			  include_top = False
			  )
# Layers of the model should not be trainable:-
model.trainable = False

#content_loss:-
def content_loss(base_img,target_img): #images are in matrix form(pixels)
    return tf.reduce_mean(tf.square(base_img - target_img))

# main loop:-
c_weights = 1
s_weights = 1e4
for i in range(iterations):
    # content loss
    content_loss = content_loss(features(content_img,"content")[0], features(init_img,"content")[0])
    
    # style loss
    for i in range(len(style_layers)):
        loss = (gram_matrix(features(style_img,"style")[i]) - gram_matrix(features(init_img,"style"))[i])**2
        style_loss +=loss*float(0.25)
        
    #total_loss:-
    total_loss = content_loss*c_weight + style_loss*s_weight
    
    # calulating gradient/slope:-
    grad = cal_gradient(total_loss,base_img)
    # optimize
    opt.apply(grad,base_image)

# style_loss:-

# content_loss:-

# fetch_img():-

