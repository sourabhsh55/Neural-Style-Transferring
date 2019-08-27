
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

# load_image :-
def load_img(img):
    img = np.asarray(img,dtype='float32')
    content_array = np.expand_dims(img,axis=0)
    content_array[:, :, :, 0] -= 103.939
    content_array[:, :, :, 1] -= 116.779
    content_array[:, :, :, 2] -= 123.68
    content_array=content_array[:, :, :, ::-1]
    # type(img)
    # print(img.shape)
    img=tf.convert_to_tensor(content_array)
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

# computing the loss for feed_forward:-
def compute_losses():
    # content loss
    content_l = content_loss(features(content_img,"content")[0][0], features(init_img,"content")[0][0])
    # style loss
    style_loss =0
    for j in range(len(style_layers)):
        loss = tf.reduce_mean(gram_matrix(features(style_img,"style")[j][0]) - gram_matrix(features(init_img,"style")[j][0]))
        style_loss +=loss*float(j+0.5)
    #total_loss:-
    total_loss = content_l*c_weight + style_loss*s_weight    
    return content_loss, style_loss, total_loss

# for calculating the slope:-
def cal_slope(total_loss, init_img):
    with tf.GradientTape() as tape: 
        pass
    return tf.gradients(total_loss,init_img)
# cal gram_matrix:-

def gram_matrix(input_tensor):
    # We make the image channels first 
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

# loading images:-
content_img = load_img(content_img)
style_img = load_img(style_img)
init_img = content_img

# variables:-
iterations = 1
norm_means = np.array([103.939, 116.779, 123.68])
min_value = -norm_means
max_value = 255 - norm_means 
style_loss =0
opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)
init_img = tfe.Variable(content_img,dtype=tf.float32) # making init_img variable eagerly_executable for updation.
c_weight = 1e3
s_weight = 1e-2

# main loop:-
best_loss = float('inf') # infinitey large float number.
for i in range(100): 
    # calulating gradient/slope:-
    grad = cal_slope(init_img)
    # optimize
    opt.apply_gradients([(grad, init_img)])
	
# OUTPUT:-
image = init_img.numpy()
image.shape
plt.imshow(image[0])
plt.show()
