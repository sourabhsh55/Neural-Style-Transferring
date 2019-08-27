
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
from tensorflow.python.keras import models

# for eager_execution:-
tf.enable_eager_execution()

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
def get_model():
    model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    model.trainable = False
    style = [model.get_layer(name).output for name in style_layers]
    content = [model.get_layer(name).output for name in content_layers]
    model_output = content+style
    return models.Model(model.input,model_output)
# Layers of the model should not be trainable:-
model.trainable = False

# feature_extraction :-
def features(img,konsa):
    sudo_model = get_model()
    sudo_t = sudo_model(img)
    sudo_t = {"content":[sudo_t[0]],"style":sudo_t[1:-1]}
    return sudo_t[konsa]

#content_loss:-
def content_loss(base_img,target_img): #images are in matrix form(pixels)
    return tf.reduce_mean(tf.square(base_img - target_img))

# variables:-
iterations = 1
norm_means = np.array([103.939, 116.779, 123.68])
min_value = -norm_means
max_value = 255 - norm_means 
style_loss =0
opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)
init_img = tfe.Variable(content_img,dtype=tf.float32)
c_weight = 1e3
s_weight = 1e-2

# main loop:-
sess = tf.Session()
best_loss = float('inf')
for i in range(100):
    # calulating gradient/slope:-
    grad = cal_slope(init_img)
    # optimize
    opt.apply_gradients([(grad, init_img)])
    


