import os
import numpy as np 
import cv2
import requests
import sys
from PIL import Image
from io import BytesIO
from matplotlib import pyplot
import modelCore
import tensorflow as tf
from datetime import datetime

MODEL_DIR = 'pretrained_weights'
global graph
graph = tf.get_default_graph()

class ManTraNet():
    def __init__(self):
        self.model = modelCore.load_pretrain_model_by_index( 4, MODEL_DIR )
        print('ManTraNet Architecture')
        print(self.model.summary(line_length=120))
        IMCFeatex = self.model.get_layer('Featex')
        print('Image Manipulation Classification Network')
        print(IMCFeatex.summary(line_length=120))

    def read_rgb_image(self, image_file) :
        rgb = cv2.imread( image_file, 1 )[...,::-1]
        return rgb
        
    def decode_an_image_array(self, rgb) :
        x = np.expand_dims( rgb.astype('float32')/255.*2-1, axis=0 )
        t0 = datetime.now()
        with graph.as_default():
            y = self.model.predict(x)[0,...,0]
        t1 = datetime.now()
        return y, t1-t0

    def decode_an_image_file(self, image_file) :
        rgb = self.read_rgb_image( image_file )
        mask, ptime = self.decode_an_image_array( rgb )
        return rgb, mask, ptime.total_seconds()

    # Returns original image, segmentation mask, and performance time
    def classify(self, img):
        rgb, mask, ptime = self.decode_an_image_file( img ) 
        return rgb, mask, ptime