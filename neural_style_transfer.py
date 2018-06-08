from __future__ import division
import scipy.io
import scipy.misc
import imageio
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

import warnings
warnings.filterwarnings("ignore")

def weights(layer, layer_name, model_layers):
    wb = model_layers[0][layer][0][0][2]
    W = wb[0][0]
    b = wb[0][1]
    layer_name = model_layers[0][layer][0][0][0][0]
    return W, b

def conv2d(previous_layer, layer, layer_name, model_layers):
    W, b = weights(layer, layer_name, model_layers)
    W = tf.constant(W)
    b = tf.constant(np.reshape(b, b.size))
    return tf.nn.conv2d(previous_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b

def relu_conv2d(previous_layer, layer, layer_name, model_layers):
    return tf.nn.relu(conv2d(previous_layer, layer, layer_name, model_layers))

def avg_pool(layer):
    return tf.nn.avg_pool(layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	

def get_vgg16_model(height=512, width=512):
    vgg = scipy.io.loadmat('./imagenet-vgg-verydeep-16.mat')
    layers = vgg['layers']
    model = {}
    model['input'] = tf.Variable(np.zeros((1, height, width, 3)), dtype = 'float32')
    model['conv1_1'] = relu_conv2d(model['input'], 0, 'conv1_1', layers)
    model['conv1_2'] = relu_conv2d(model['conv1_1'], 2, 'conv1_2', layers)
    model['avgpool1'] = avg_pool(model['conv1_2'])
    model['conv2_1'] = relu_conv2d(model['avgpool1'], 5, 'conv2_1', layers)
    model['conv2_2'] = relu_conv2d(model['conv2_1'], 7, 'conv2_2', layers)
    model['avgpool2'] = avg_pool(model['conv2_2'])
    model['conv3_1']  = relu_conv2d(model['avgpool2'], 10, 'conv3_1', layers)
    model['conv3_2']  = relu_conv2d(model['conv3_1'], 12, 'conv3_2', layers)
    model['conv3_3']  = relu_conv2d(model['conv3_2'], 14, 'conv3_3', layers)
    model['avgpool3'] = avg_pool(model['conv3_3'])
    model['conv4_1']  = relu_conv2d(model['avgpool3'], 17, 'conv4_1', layers)
    model['conv4_2']  = relu_conv2d(model['conv4_1'], 19, 'conv4_2', layers)
    model['conv4_3']  = relu_conv2d(model['conv4_2'], 21, 'conv4_3', layers)
    model['avgpool4'] = avg_pool(model['conv4_3'])
    model['conv5_1']  = relu_conv2d(model['avgpool4'], 24, 'conv5_1', layers)
    model['conv5_2']  = relu_conv2d(model['conv5_1'], 26, 'conv5_2', layers)
    model['conv5_3']  = relu_conv2d(model['conv5_2'], 28, 'conv5_3', layers)
    model['avgpool5'] = avg_pool(model['conv5_3'])
    return model

	
def compute_content_cost(a_C, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_C_unrolled = tf.transpose(a_C)
    a_G_unrolled = tf.transpose(a_G)
    
    J_content = (1/ (4* n_H * n_W * n_C)) * tf.reduce_sum(tf.pow((a_G_unrolled - a_C_unrolled), 2))    
    return J_content

def gram_matrix(A):
    GA = tf.matmul(A, tf.transpose(A))    
    return GA	
	
def compute_layer_style_cost(a_S, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    a_S = tf.transpose(tf.reshape(a_S, [n_H*n_W, n_C]))
    a_G = tf.transpose(tf.reshape(a_G, [n_H*n_W, n_C]))

    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    J_style_layer = (1./(4 * n_C**2 * (n_H*n_W)**2)) * tf.reduce_sum(tf.pow((GS - GG), 2))
    return J_style_layer

STYLE_LAYERS  = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]	

def compute_style_cost(model, STYLE_LAYERS):    
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:
        out = model[layer_name]
        a_S = sess.run(out)

        a_G = out
        
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        J_style += coeff * J_style_layer

    return J_style

def total_cost(J_content, J_style, alpha = 20, beta = 30):
    J = alpha * J_content + beta * J_style
    return J
	
	
# Reset the graph
tf.reset_default_graph()

# Start interactive session
sess = tf.Session()	

def reshape_and_normalize_image(image):    
    image = np.reshape(image, ((1,) + image.shape))
    image = image - np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3)) 
    return image
	
content_image = imageio.imread("content_image.jpg")
content_image = cv2.resize(content_image, (512,512))

content_image = reshape_and_normalize_image(content_image)

style_image = imageio.imread("style_image.jpg")
style_image = cv2.resize(style_image, (512,512))

style_image = reshape_and_normalize_image(style_image)

#Hyperparameters
num_iterations = 2000
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
COLOR_CHANNELS = 3
NOISE_RATIO = 0.6

def generate_noise_image(content_image, noise_ratio = NOISE_RATIO):    
    noise_image = np.random.uniform(-20, 20, (1, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS)).astype('float32')
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)    
    return input_image

generated_image = generate_noise_image(content_image)	

model = get_vgg16_model()
sess.run(model['input'].assign(content_image))
out = model['conv4_2']
a_C = sess.run(out)
a_G = out
J_content = compute_content_cost(a_C, a_G)

# Assign the input of the model to be the "style" image # Assign 
sess.run(model['input'].assign(style_image))

# Compute the style cost
J_style = compute_style_cost(model, STYLE_LAYERS)

J = total_cost(J_content, J_style, alpha = 40, beta = 10)

# define
optimizer = tf.train.AdamOptimizer(2.0)

# define traininig step
train_step = optimizer.minimize(J)

with tf.Session() as sess:
    
    # Initialize global variables (you need to run the session on the initializer)
    sess.run(tf.global_variables_initializer())
    
    # Run the noisy input image (initial generated image) through the model. Use assign().
    sess.run(model['input'].assign(generated_image))
    
    for i in range(num_iterations):
    
        # Run the session on the train_step to minimize the total cost
        sess.run(train_step)
        
        # Compute the generated image by running the session on the current model['input']
        generated_image = sess.run(model['input'])

        # Print every 50 iteration.
        if i % 50 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            #print("Iteration " + str(i) + " :", " total cost = " + str(Jt)," content cost = " + str(Jc)," style cost = " + str(Js))
            #print()
            
            # save current generated image in the "/output" directory
            #image = np.clip(generated_image[0], 0, 255).astype('uint8')
            image = generated_image[0]
            output_file = 'temp_output_{}.jpg'.format(i)
            imageio.imwrite(output_file, image)
    image = generated_image[0]
    # save last generated image
    imageio.imwrite('generated_image.jpg', image)
    
    print("Finished")

