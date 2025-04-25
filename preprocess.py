import pickle
import numpy as np
import tensorflow as tf
import os
from helper_code import *

def extract_CNN_features(record):
    
    signal, fields = load_signals(record)

    #normalize 0 to 1?
    #truncate particularly large and small values?

    print(f"signal type {type(signal)}")
    print(signal)
    return signal


def get_data(data_folder):

    print('Finding the Challenge data...')

    

    records = find_records(data_folder)

    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Extract the features and labels from the data.
    print('Extracting features and labels from the data...')

    # features = np.zeros((num_records, 6), dtype=np.float64)
    features = np.zeros((num_records,12), dtype=np.ndarray) # array of arrays with waveform data

    labels = np.zeros(num_records, dtype=bool)

    # Iterate over the records.
    for i in range(len(records)):
        width = len(str(num_records))
        print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])
        features[i] = extract_CNN_features(record)
        labels[i] = load_label(record)

    one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)

    return features, one_hot_labels

