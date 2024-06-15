import os
import sys
import scipy.io
import imageio
import scipy.misc
from scipy import io
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np
import pprint                  #to format printing of the vgg mode
import tensorflow as tf
from utils import *
class CONFIG:
    IMAGE_WIDTH = 400
    IMAGE_HEIGHT = 300
    COLOR_CHANNELS = 3
    NOISE_RATIO = 0.4
    MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3)) 
    VGG_MODEL = 'pretrained-model/imagenet-vgg-verydeep-19.mat' # Pick the VGG 19-layer model by from the paper "Very Deep Convolutional Networks for Large-Scale Image Recognition".
    STYLE_IMAGE = 'images/stone_style.jpg' # Style image to use.
    CONTENT_IMAGE = 'images/content300.jpg' # Content image to use.
    OUTPUT_DIR = 'output/'

def generate_noise_image(content_image,noise_ratio=CONFIG.NOISE_RATIO):
    noise=np.random.uniform(-20,20,(CONFIG.IMAGE_HEIGHT,CONFIG.IMAGE_WIDTH,CONFIG.COLOR_CHANNELS))
    noise_image=noise_ratio*noise+(1-noise_ratio)*content_image
    return noise_image
def resize_and_normalize(image):
    image=np.reshape(image,((1,)+image.shape))
    image=image-CONFIG.MEANS
    return image
def content_cost(a_C,a_G):
    m,n_h,n_w,n_c=a_C.shape
    a_c=tf.reshape(a_C,(-1,n_c))
    a_g=tf.reshape(a_G,(-1,n_c))
    subtract=tf.subtract(a_c,a_g)
    J_content=tf.reduce_sum(tf.square(subtract))
    return J_content
def layer_style_cost(a_S,a_G):
    m,n_h,n_w,n_c=a_S.shape
    a_S_unroll=tf.reshape(a_S,(n_c,-1))
    a_G_unroll=tf.reshape(a_G,(n_c,-1))
    a_S_Gram=gram_matrix(a_S_unroll)
    a_G_Gram=gram_matrix(a_G_unroll)
    subtract=tf.subtract(a_S_Gram,a_G_Gram)
    J_style=tf.reduce_sum(tf.square(subtract))
    return J_style
def gram_matrix(A):
    AT=tf.transpose(A,perm=[1,0])
    G=tf.matmul(A,AT)
    return G

STYLE_LAYERS=[
    ('conv1_1',.5),
    ('conv2_1',.5),
    ('conv3_1',.5),
    ('conv4_1',.5),
    ('conv5_1',.5),
    ('conv6_1',.5)
]
def save_image(path, image):
    image = image + CONFIG.MEANS
    image = np.clip(image[0], 0, 255).astype('uint8')
    imageio.imsave(path, image)
def total_cost(content_cost,style_cost,alpha,beta):
    J=alpha*content_cost+beta*style_cost
    return J
