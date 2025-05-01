from __future__ import absolute_import
from matplotlib import pyplot as plt

import tensorflow as tf
import numpy as np

"""
Subclassed tf.keras model for more explicit control
The current state of this file shows our optimized CNN architecture
"""
class CNN(tf.keras.Model):
    def __init__(self, classes):
        super(CNN, self).__init__()


        # Input shape: (time_steps, channels)

        self.conv1 = tf.keras.layers.Conv1D(24, kernel_size=7, strides=1, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv1D(24, kernel_size=7, strides=1, padding='same', activation='relu')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPooling1D(pool_size=2)

        self.conv3 = tf.keras.layers.Conv1D(48, kernel_size=7, strides=1, padding='same', activation='relu')
        self.conv4 = tf.keras.layers.Conv1D(48, kernel_size=7, strides=1, padding='same', activation='relu')
        self.conv5 = tf.keras.layers.Conv1D(48, kernel_size=7, strides=1, padding='same',activation='relu')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.pool2 = tf.keras.layers.MaxPooling1D(pool_size=2)

        self.flatten = tf.keras.layers.Flatten()
        self.dropout1 = tf.keras.layers.Dropout(0.4)
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.4)
        self.fc2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(2, activation='softmax')

    #custom call function
    def call(self, inputs, is_training=False):

        
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.pool1(x)

         
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.bn2(x, training=is_training)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.dropout1(x, training=is_training)
        x = self.fc1(x)
        x = self.dropout2(x, training=is_training)
        x = self.fc2(x)

        return self.output_layer(x)

    """
    In this code we decided to use CCE based on experience in CSCI1470
    BCE using a sigmoid as the last activation function should be less computationally intensive
    Future directions include implementing a custom loss function to penalize false positives and 
    discourage false negatives.
    """
    def loss_fn(self, logits, labels):
        cce = tf.keras.losses.CategoricalCrossentropy()
        return cce(labels, logits)

    """
    Accuracy function used to evaluate the model
    """
    def accuracy(self, logits, labels):

        threshold = 0.3
        
        # max_logits = tf.cast(logits[:,1] > threshold, tf.int64)
        max_logits = tf.argmax(logits,axis=1)
        max_labels = tf.argmax(labels, axis=1)

        comp_pred_true = tf.equal(max_logits, max_labels)
        accuracy = tf.reduce_mean(tf.cast(comp_pred_true, dtype=tf.float32))
        return accuracy
