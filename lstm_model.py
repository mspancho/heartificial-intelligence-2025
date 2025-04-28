from __future__ import absolute_import
from matplotlib import pyplot as plt

import os
import tensorflow as tf
import numpy as np
import random
import math

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
        #use cross entropy to calculate loss
        cce = tf.keras.losses.CategoricalCrossentropy()
        return cce(labels, logits)


    def accuracy(self, logits, labels):
        max_logits = tf.argmax(logits,axis=1)
        max_labels = tf.argmax(labels, axis=1)

        comp_pred_true = tf.equal(max_logits, max_labels)
        accuracy = tf.reduce_mean(tf.cast(comp_pred_true, dtype=tf.float32))
        return accuracy

class GRUsolo(tf.keras.Model):
    def __init__(self, input_size, hidden_size=128, 
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
        #use cross entropy to calculate loss
        cce = tf.keras.losses.CategoricalCrossentropy()
        return cce(labels, logits)

    def accuracy(self, logits, labels):
        max_logits = tf.argmax(logits,axis=1)
        max_labels = tf.argmax(labels, axis=1)

        comp_pred_true = tf.equal(max_logits, max_labels)
        accuracy = tf.reduce_mean(tf.cast(comp_pred_true, dtype=tf.float32))
        return accuracy

# class LSTMLP(tf.keras.Model):
#     def __init__(self, input_size, hidden_size=128, 
#                  num_layers=1, num_classes=2, 
#                  dropout=0.5, recurrent_dropout=0.125):
#         super(LSTMLP, self).__init__()

#         self.lstm_layers = [
#             tf.keras.layers.LSTM(
#                 hidden_size,
#                 dropout=dropout,
#                 recurrent_dropout=recurrent_dropout
#             ) for i in range(num_layers)
#         ]
        
#         self.mlp_layers = [
#             tf.keras.layers.Dense(hidden_size, activation='leaky_relu', kernel_initializer='glorot_uniform', kernel_regularization=tf.keras.regularizers.l2(0.01)),
#             tf.keras.layers.Dropout(dropout),
#             tf.keras.layers.Dense(hidden_size, kernel_initializer='glorot_uniform', activation='softmax')
#         ]

#         self.fc = tf.keras.layers.Dense(num_classes)

#     def call(self, x_ekg, x_static, training=False):
#         # x: (batch, seq_len, input_size)
#         x_lstm = x_ekg
#         for lstm_layer in self.lstm_layers:
#             x_lstm = lstm_layer(x_lstm, training=training)
#         lstm_out = x_lstm
#         mlp_out = x_static
#         for mlp_layer in self.mlp_layers:
#             mlp_out = mlp_layer(mlp_out, training=training)
#         # Concatenate the LSTM output with static features
#         x = tf.concat([lstm_out, mlp_out], axis=-1)
#         # Pass through the final fully connected layer
#         x = tf.reshape(x, (x.shape[0], -1))
#         outs = self.fc(x)
#         return outs
