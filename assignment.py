from __future__ import absolute_import
from matplotlib import pyplot as plt
from cnn_model import CNN


import os
import tensorflow as tf
import numpy as np
import random
import math
from helper_code import *


# ensures that we run only on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'



def train(model, optimizer, train_inputs, train_labels):
   '''
   Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs
   and labels - ensure that they are shuffled in the same order using tf.gather.
   To increase accuracy, you may want to use tf.image.random_flip_left_right on your
   inputs before doing the forward pass. You should batch your inputs.
   :param model: the initialized model to use for the forward pass and backward pass
   :param train_inputs: train inputs (all inputs to use for training),
   shape (num_inputs, width, height, num_channels)
   :param train_labels: train labels (all labels to use for training),
   shape (num_labels, num_classes)
   :return: None
   '''

   
   batch_size = 256
   num_batches = len(train_inputs)// batch_size
   indices = tf.random.shuffle(tf.range(tf.shape(train_inputs)[0]))
   training_inputs = tf.gather(train_inputs, indices)
   training_labels = tf.gather(train_labels, indices)
   train_acc = 0.0
   loss_list = []
   for batch_num in range(num_batches ):
      batch_inputs = training_inputs[batch_num * batch_size : (batch_num + 1) * batch_size]
      batch_labels = training_labels[batch_num * batch_size : (batch_num + 1) * batch_size]

      batch_inputs = tf.image.random_flip_left_right(batch_inputs)

      with tf.GradientTape() as g:
         logits = model.call(batch_inputs)
         loss = model.loss(logits, batch_labels)
         loss_list.append(loss)



         #if batch_num % 50 ==0:
            #print( f"Batch {batch_num}, accuracy : batch accuracy {batch_accuracy}")
      
      grads = g.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads,model.trainable_variables))
      batch_accuracy = float(model.accuracy(logits, batch_labels))
      train_acc += batch_accuracy


   return train_acc/num_batches, loss_list



def test(model, test_inputs, test_labels):
   """
   Tests the model on the test inputs and labels. You should NOT randomly
   flip images or do any extra preprocessing.
   :param test_inputs: test data (all images to be tested),
   shape (num_inputs, width, height, num_channels)
   :param test_labels: test labels (all corresponding labels),
   shape (num_labels, num_classes)
   :return: test accuracy - this should be the average accuracy across
   all batches
   """

   batch_size = 120
   num_batches = len(test_inputs)//batch_size
   total_acc = 0.0

   for batch_num in range(num_batches):
      batch_inputs = test_inputs[batch_num * batch_size : (batch_num + 1) * batch_size]
      batch_labels = test_labels[batch_num * batch_size : (batch_num + 1) * batch_size]
    
      logits = model.call(batch_inputs, is_testing=True)
      acc = model.accuracy(logits, batch_labels)
      total_acc += acc
   #test_acc = model.accuracy(logits, test_labels)
   #print(f"Accuracy on testing set : {test_acc}")


   return float(total_acc/num_batches)

def visualize_loss(losses):
   """
   Uses Matplotlib to visualize the losses of our model.
   :param losses: list of loss data stored from train. Can use the model's loss_list
   field
   NOTE: DO NOT EDIT
   :return: doesn't return anything, a plot should pop-up
   """
   x = [i for i in range(len(losses))]
   plt.plot(x, losses)
   plt.title('Loss per batch')
   plt.xlabel('Batch')
   plt.ylabel('Loss')
   plt.show()




def visualize_results(image_inputs, logits, image_labels, first_label, second_label):
  
   def plotter(image_indices, label):
       nc = 10
       nr = math.ceil(len(image_indices) / 10)
       fig = plt.figure()
       fig.suptitle(
           f"{label} Examples\nPL = Predicted Label\nAL = Actual Label")
       for i in range(len(image_indices)):
           ind = image_indices[i]
           ax = fig.add_subplot(nr, nc, i+1)
           ax.imshow(image_inputs[ind], cmap="Greys")
           pl = first_label if predicted_labels[ind] == 0.0 else second_label
           al = first_label if np.argmax(
               image_labels[ind], axis=0) == 0 else second_label
           ax.set(title=f"PL: {pl}\nAL: {al}")
           plt.setp(ax.get_xticklabels(), visible=False)
           plt.setp(ax.get_yticklabels(), visible=False)
           ax.tick_params(axis='both', which='both', length=0)


   predicted_labels = np.argmax(logits, axis=1)
   num_images = image_inputs.shape[0]


   # Separate correct and incorrect images
   correct = []
   incorrect = []
   for i in range(num_images):
       if predicted_labels[i] == np.argmax(image_labels[i], axis=0):
           correct.append(i)
       else:
           incorrect.append(i)


   plotter(correct, 'Correct')
   plotter(incorrect, 'Incorrect')
   plt.show()


def main():

   LOCAL_TRAIN_FILE = 'data/train'
   LOCAL_TEST_FILE = 'data/test'


   # TODO: assignment.main() pt 1
   # Load your testing and training data using the get_data function
   train_inputs, train_labels = get_data(LOCAL_TRAIN_FILE, [3, 5])


   test_inputs, test_labels = get_data(LOCAL_TEST_FILE, [3, 5])
   print("Shape of inputs:", test_inputs.shape)
   print("Shape of one-hot labels:", test_labels.shape)



   model = CNN([3,5])

   #could be nice for results to do multiple training rates
   optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)


   print('about to start training')
   epochs = 4
   loss_list = []
   for epoch in range(epochs):
      acc, epoch_loss = train(model,optimizer=optimizer, train_inputs=train_inputs, train_labels=train_labels)
      print(f"epoch {epoch}: {acc}")
      loss_list.append(epoch_loss)

   flat_loss_list = [loss for epoch_loss in loss_list for loss in epoch_loss]
   visualize_loss(flat_loss_list)
   print('done training')

   print(f"test acc: {test(model, test_inputs=test_inputs, test_labels=test_labels)}")

   return

if __name__ == '__main__':
   main()