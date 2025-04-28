from __future__ import absolute_import
from matplotlib import pyplot as plt

import os
import tensorflow as tf
import numpy as np
import random
import math

# ensures that we run only on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class MLP(tf.keras.Model):
    def __init__(self, classes):
        """
        This model class will contain the architecture for your CNN that
        classifies images. Do not modify the constructor, as doing so
        will break the autograder. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(MLP, self).__init__()

        # Initialize all hyperparameters
        self.loss_list = []
        self.batch_size = 64
        self.input_width = 32
        self.input_height = 32
        self.image_channels = 3
        self.num_classes = classes
        self.hidden_layer_size = 128
        
        self.layer1 = tf.keras.layers.Dense(256, kernel_initializer	="glorot_uniform", activation='leaky_relu')
        self.layer2 = tf.keras.layers.Dense(128, kernel_initializer	="glorot_uniform", activation='leaky_relu')
        self.layer3 = tf.keras.layers.Dense(self.num_classes, kernel_initializer="glorot_uniform", activation='softmax')

    def call(self, inputs, is_training=False):
        
        flat_inputs = tf.keras.layers.Flatten()(inputs)
        
        layer1Output = self.layer1(flat_inputs)
        layer2Output = self.layer2(layer1Output)
        logits = self.layer3(layer2Output)

        return logits
    

    def loss_fn(self, logits, labels):
        #use cross entropy to calculate loss
        cce = tf.keras.losses.CategoricalCrossentropy()
        # print(labels)
        return cce(labels, logits)
	
    def accuracy(self, logits, labels):
        max_logits = tf.argmax(logits,axis=1)
        max_labels = tf.argmax(labels, axis=1)

        comp_pred_true = tf.equal(max_logits, max_labels)
        accuracy = tf.reduce_mean(tf.cast(comp_pred_true, dtype=tf.float32))
        return accuracy
