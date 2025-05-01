from __future__ import absolute_import
from matplotlib import pyplot as plt

import os
import tensorflow as tf
import numpy as np
import random
import math

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class CNN(tf.keras.Model):
    def __init__(self, classes):
        super(CNN, self).__init__()


        # Input shape: (time_steps, channels)

        self.conv1 = tf.keras.layers.Conv1D(256, kernel_size=21, strides=1, padding='same', activation='relu')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPooling1D(pool_size=2)

        self.conv2 = tf.keras.layers.Conv1D(128, kernel_size=11, strides=1, padding='same', activation='relu')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.pool2 = tf.keras.layers.MaxPooling1D(pool_size=2)

        self.conv3 = tf.keras.layers.Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu')
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.pool3 = tf.keras.layers.MaxPooling1D(pool_size=2)

        self.flatten = tf.keras.layers.Flatten()
        self.dropout1 = tf.keras.layers.Dropout(0.4)
        self.fc1 = tf.keras.layers.Dense(256, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.4)
        self.fc2 = tf.keras.layers.Dense(128, activation='relu')
        self.output_layer = tf.keras.layers.Dense(2, activation='softmax')


    def call(self, inputs, is_training=False):

        # x = self.conv1(inputs)
        # x = self.bn1(x, training=is_training)
        # x = self.pool1(x)

        # x = self.conv2(x)
        # x = self.bn2(x, training=is_training)
        # x = self.pool2(x)

        # x = self.conv3(x)
        # x = self.bn3(x, training=is_training)
        # x = self.pool3(x)

        x = self.conv1(inputs)
        x = self.bn1(x, training=is_training)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=is_training)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x, training=is_training)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.dropout1(x, training=is_training)
        x = self.fc1(x)
        #x = self.dropout2(x, training=is_training)
        #x = self.fc2(x)



        # print(f"shape of output {self.output_layer(x).shape}")

        return self.output_layer(x)

    def loss_fn(self, logits, labels):
        #use cross entropy to calculate loss
        cce = tf.keras.losses.CategoricalCrossentropy()
        # print(labels)
        return cce(labels, logits)
	
    def accuracy(self, logits, labels):

        threshold = 0.3
        
        max_logits = tf.cast(logits[:,1] > threshold)
        # max_logits = tf.argmax(logits,axis=1)
        max_labels = tf.argmax(labels, axis=1)

        comp_pred_true = tf.equal(max_logits, max_labels)
        accuracy = tf.reduce_mean(tf.cast(comp_pred_true, dtype=tf.float32))
        return accuracy
