from __future__ import absolute_import
from matplotlib import pyplot as plt

import os
import tensorflow as tf
import numpy as np
import random
import math

"""
We began implementing LSTM and GRU models to generate further results
However, due to computational limitations we did not
fully explore the viability of these approaches.
"""
class LSTMsolo(tf.keras.Model):
    def __init__(self, input_size, hidden_size=128, 
                 num_layers=1, num_classes=2, 
                 dropout=0.5, recurrent_dropout=0.125):
        super(LSTMsolo, self).__init__()
        self.lstm_layers = [
            tf.keras.layers.LSTM(
                hidden_size,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout
            ) for i in range(num_layers)
        ]
        self.fc = tf.keras.layers.Dense(num_classes)

    def call(self, x, is_training=False):
        # x: (batch, seq_len, input_size)
        for lstm in self.lstm_layers:
            x = lstm(x, training=is_training)
        out = self.fc(x)
        return out
    
    def loss_fn(self, logits, labels):
        cce = tf.keras.losses.CategoricalCrossentropy()
        return cce(labels, logits)


    def accuracy(self, logits, labels):
        max_logits = tf.argmax(logits,axis=1)
        max_labels = tf.argmax(labels, axis=1)

        comp_pred_true = tf.equal(max_logits, max_labels)
        accuracy = tf.reduce_mean(tf.cast(comp_pred_true, dtype=tf.float32))
        return accuracy

class GRUsolo(tf.keras.Model):
    def __init__(self, hidden_size=128, 
                 num_layers=1, num_classes=2, 
                 dropout=0.5, recurrent_dropout=0.125):
        super(GRUsolo, self).__init__()
        self.lstm_layers = [
            tf.keras.layers.GRU(
                hidden_size,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout
            ) for i in range(num_layers)
        ]
        self.fc = tf.keras.layers.Dense(num_classes)

    def call(self, x, is_training=False):
        # x: (batch, seq_len, input_size)
        for lstm in self.lstm_layers:
            x = lstm(x, training=is_training)
        out = self.fc(x)
        return out
    
    def loss_fn(self, logits, labels):
        cce = tf.keras.losses.CategoricalCrossentropy()
        return cce(labels, logits)

    def accuracy(self, logits, labels):
        max_logits = tf.argmax(logits,axis=1)
        max_labels = tf.argmax(labels, axis=1)

        comp_pred_true = tf.equal(max_logits, max_labels)
        accuracy = tf.reduce_mean(tf.cast(comp_pred_true, dtype=tf.float32))
        return accuracy