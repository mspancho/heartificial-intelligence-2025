from __future__ import absolute_import
from matplotlib import pyplot as plt
from cnn_model import CNN
from lstm_model import LSTMsolo, GRUsolo#, LSTMLP
from mlp_model import MLP

import os
import tensorflow as tf
import numpy as np
import random
import math
from helper_code import *
from preprocess import get_data

"""
Training loop recording loss and accuracy for graph generation.
This train method is for one epoch and is therefore looped in main for i epochs
Takes in a model, optimizer, as well as input train_inputs and train_labels
"""

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
      optimizer.build(model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))
      
      batch_accuracy = float(model.accuracy(logits, batch_labels))
      accuracy_list.append(batch_accuracy)  
      train_acc += batch_accuracy

   return train_acc / num_batches, loss_list, accuracy_list


"""
General test function, also records predictions/truth value of all input cases 
to generate a confusion matrix
"""
def test(model, test_inputs, test_labels):

   batch_size = 64
   num_batches = len(test_inputs)//batch_size
   total_acc = 0.0

   all_preds = []
   all_true = []

   for batch_num in range(num_batches):
      batch_inputs = test_inputs[batch_num * batch_size : (batch_num + 1) * batch_size]
      batch_labels = test_labels[batch_num * batch_size : (batch_num + 1) * batch_size]
      
      logits = model.call(batch_inputs)
      acc = model.accuracy(logits, batch_labels)
      total_acc += acc

      preds = tf.argmax(logits, axis=1)
      true = tf.argmax(batch_labels, axis=1)
        
      all_preds.extend(preds.numpy())
      all_true.extend(true.numpy())

      #prints confusion matrix for evaluation
      TP = 0
      FP = 0
      FN = 0
      TN = 0

      for t, p in zip(all_true, all_preds):
        if t == 1 and p == 1:
            TP += 1
        elif t == 0 and p == 1:
            FP += 1
        elif t == 1 and p == 0:
            FN += 1
        elif t == 0 and p == 0:
            TN += 1

   print("\nConfusion Matrix:")
   print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}\n")

   matrix = np.array([[TN, FP],
                       [FN, TP]])

   fig, ax = plt.subplots()
   cax = ax.matshow(matrix, cmap='Blues')
   plt.colorbar(cax)

   ax.set_xticklabels([''] + ['0', '1'])
   ax.set_yticklabels([''] + ['0', '1'])
   plt.xlabel('Predicted')
   plt.ylabel('True')
   plt.title('Confusion Matrix')

   for (i, j), val in np.ndenumerate(matrix):
        color = 'white' if matrix[i, j] > matrix.max() / 2 else 'black'
        ax.text(j, i, f'{val}', ha='center', va='center', color=color, fontsize=14, fontweight='bold')

   plt.show()

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

   mlp = MLP(2)
   model = CNN(2)
   gru = GRUsolo(input_size=12, hidden_size=128, num_layers=1, num_classes=2, dropout=0.5, recurrent_dropout=0.125)
<<<<<<< HEAD
   seq_models = [gru]
   optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
=======
   seq_models = [model]
   optimizer = tf.keras.optimizers.Adam(learning_rate=3e-3)
>>>>>>> 4c99c63578498f4a9d9881633ca58b657877b546

   print('about to start training')
   epochs = 5
   for model in seq_models:
      all_losses = []
      all_accuracies = []

      for epoch in range(epochs):
         acc, epoch_losses, epoch_accuracies = train(model, optimizer, train_inputs, train_labels)
         print(f"epoch {epoch}: {acc}")
         all_losses.extend(epoch_losses)
         all_accuracies.extend(epoch_accuracies)

      visualize_loss(all_losses, all_accuracies)

      # with open('loss.txt', 'w') as f1:
      #    f1.write(str(all_losses))

      # with open('acc.txt', 'w') as f2:
      #    f2.write(str(all_accuracies))

      print('done training')

      print(f"test acc: {test(model, test_inputs=test_inputs, test_labels=test_labels)}")

   return


if __name__ == '__main__':
    main()