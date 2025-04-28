from __future__ import absolute_import
from matplotlib import pyplot as plt
from cnn_model import CNN
from lstm_model import LSTMsolo#, LSTMLP

import os
import tensorflow as tf
import numpy as np
import random
import math
from helper_code import *
from preprocess import get_data


def train(model, optimizer, train_inputs, train_labels):
    batch_size = 256
    num_batches = len(train_inputs) // batch_size
    print(f"num batches: {num_batches}")
   
    indices = tf.random.shuffle(tf.range(tf.shape(train_inputs)[0]))
    training_inputs = tf.gather(train_inputs, indices)
    training_labels = tf.gather(train_labels, indices)
   
    train_acc = 0.0
    loss_list = []
    accuracy_list = []   

    for batch_num in range(num_batches):
        batch_inputs = training_inputs[batch_num * batch_size : (batch_num + 1) * batch_size]
        batch_labels = training_labels[batch_num * batch_size : (batch_num + 1) * batch_size]

        with tf.GradientTape() as g:
            logits = model.call(batch_inputs, is_training=True)
            loss = model.loss_fn(logits, batch_labels)
            loss_list.append(loss.numpy())  

        grads = g.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        batch_accuracy = float(model.accuracy(logits, batch_labels))
        accuracy_list.append(batch_accuracy)  
        train_acc += batch_accuracy

    return train_acc / num_batches, loss_list, accuracy_list


def test(model, test_inputs, test_labels):

   batch_size = 64
   num_batches = len(test_inputs)//batch_size
   total_acc = 0.0

   for batch_num in range(num_batches):
      batch_inputs = test_inputs[batch_num * batch_size : (batch_num + 1) * batch_size]
      batch_labels = test_labels[batch_num * batch_size : (batch_num + 1) * batch_size]
    
      logits = model.call(batch_inputs)
      acc = model.accuracy(logits, batch_labels)
      total_acc += acc



   return float(total_acc/num_batches)

def visualize_loss(losses,accuracies):
    x = [i for i in range(len(losses))]
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(x, losses, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx() 
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)  
    ax2.plot(x, accuracies, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Loss and Accuracy per Batch')
    plt.show()


def main():
    LOCAL_TRAIN_FOLDER = 'train_data/'
    LOCAL_TEST_FOLDER = 'test_data/'
   
    train_inputs, train_labels = get_data(LOCAL_TRAIN_FOLDER)
    test_inputs, test_labels = get_data(LOCAL_TEST_FOLDER)
    print("Shape of inputs:", test_inputs.shape)
    print("Shape of one-hot labels:", test_labels.shape)

    model = CNN(2)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    print('about to start training')
    epochs = 5
    all_losses = []
    all_accuracies = []

    for epoch in range(epochs):
        acc, epoch_losses, epoch_accuracies = train(model, optimizer, train_inputs, train_labels)
        print(f"epoch {epoch}: {acc}")
        all_losses.extend(epoch_losses)
        all_accuracies.extend(epoch_accuracies)

    visualize_loss(all_losses, all_accuracies)
    print('done training')

    print(f"test acc: {test(model, test_inputs=test_inputs, test_labels=test_labels)}")

    return


if __name__ == '__main__':
   main()