#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys
import tensorflow as tf

from helper_code import *

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose):
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    records = find_records(data_folder)

    
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Extract the features and labels from the data.
    if verbose:
        print('Extracting features and labels from the data...')

    # features = np.zeros((num_records, 6), dtype=np.float64)
    features = np.zeros((num_records,), dtype=np.ndarray) # array of arrays with waveform data

    labels = np.zeros(num_records, dtype=bool)

    # Iterate over the records.
    for i in range(len(records)):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])
        features[i] = extract_CNN_features(record)
        labels[i] = load_label(record)


    # Train the models.
    if verbose:
        print('Training the model on the data...')


    # ===========
    # This very simple model trains a random forest model with very simple features.

    # Define the parameters for the random forest classifier and regressor.
    # n_estimators = 12  # Number of trees in the forest.
    # max_leaf_nodes = 34  # Maximum number of leaf nodes in each tree.
    # random_state = 56  # Random state; set for reproducibility.

    # # Fit the model.
    # model = RandomForestClassifier(
    #     n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, labels)
    # ===========
    model = init_MLP()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=tf.keras.losses.BinaryCrossentropy())
    # model.fit(features, labels.astype(np.float32), epochs=10, batch_size=64)


    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the model.
    save_model(model_folder, model)

    if verbose:
        print('Done.')
        print()

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def old_load_model(model_folder, verbose):
    model_filename = os.path.join(model_folder, 'model.sav')
    model = joblib.load(model_filename)
    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    # Load the model.
    # model = model['model']

    # Extract the features.
    features = extract_features(record)
    features = features.reshape(1, -1)

    # Get the model outputs.
    # binary_output = model.predict(features)[0]
    # probability_output = model.predict_proba(features)[0][1]
    prob = model.predict(features)[0,0]
    pred = bool(prob >= 0.5)

    return pred, float(prob)

    # return binary_output, probability_output

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract your features.
def extract_6_features(record):
    header = load_header(record)
    age = get_age(header)
    sex = get_sex(header)

    one_hot_encoding_sex = np.zeros(3, dtype=bool)
    if sex == 'Female':
        one_hot_encoding_sex[0] = 1
    elif sex == 'Male':
        one_hot_encoding_sex[1] = 1
    else:
        one_hot_encoding_sex[2] = 1

    signal, fields = load_signals(record)


    # TO-DO: Update to compute per-lead features. Check lead order and update and use functions for reordering leads as needed.

    num_finite_samples = np.size(np.isfinite(signal))
    if num_finite_samples > 0:
        signal_mean = np.nanmean(signal)
    else:
        signal_mean = 0.0
    if num_finite_samples > 1:
        signal_std = np.nanstd(signal)
    else:
        signal_std = 0.0

    features = np.concatenate(([age], one_hot_encoding_sex, [signal_mean, signal_std]))

    return np.asarray(features, dtype=np.float32)

def extract_CNN_features(record):
    
    signal, fields = load_signals(record)

    # normalize 0 to 1?
    # truncate particularly large and small values?

    # print(f"signal type {type(signal)}")
    # print(signal)
    return signal

# Save your trained model.
def old_save_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)


###
def save_model(model_folder, model):
    # os.makedirs(model_folder, exist_ok=True)
    model.save(os.path.join(model_folder, 'model.h5'))

def load_model(model_folder, verbose):
    model_path = os.path.join(model_folder, 'model.h5')
    if verbose:
        print(f"Loading Keras model from {model_path}")
    return tf.keras.models.load_model(model_path)



def init_cnn1d(input_shape):
    model = tf.keras.Sequential([
        # Layer 1
        tf.keras.layers.Conv1D(filters=16, kernel_size=3, strides=1, padding='same', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid'),
        # Layer 2
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid'),
        # Layer 3
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='leaky_relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='leaky_relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    return model


def init_MLP():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(6,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    return model


# write out loss and call fn?


def train(model, train_inputs, train_labels, epochs, batch_size):

    num_batches = train_inputs // batch_size

    for i in range(epochs):
        for batch_num in range(len(num_batches)):

            batch_inputs, batch_labels = get_next_batch(batch_num, train_inputs, train_labels, batch_size)


            with tf.GradientTape() as tape:
                y_pred = model.call(batch_inputs) 
                loss = model.loss_fn(y_pred, batch_labels)

                train_acc = model.accuracy(y_pred, batch_labels)
                print(f"batch {batch_num} training accuracy: {train_acc}")

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))



def get_next_batch(idx, inputs, labels, batch_size=100) -> tuple[np.ndarray, np.ndarray]:
    """
    Given an index, returns the next batch of data and labels. Ex. if batch_size is 5, 
    the data will be a numpy matrix of size 5 * 32 * 32 * 3, and the labels returned will be a numpy matrix of size 5 * 10.
    """
    return (inputs[idx*batch_size:(idx+1)*batch_size], np.array(labels[idx*batch_size:(idx+1)*batch_size]))