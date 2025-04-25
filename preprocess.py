import pickle
import numpy as np
import tensorflow as tf
import os
from helper_code import *

def extract_CNN_features(record):
    
    signal, fields = load_signals(record)


    #normalize 0 to 1?
    #truncate particularly large and small values?
    #truncate to same lenght

    # print(f"signal type {type(signal)}")
    # print(signal)

    signal = signal[0:2000]

    signal_t = tf.transpose(signal)
    return signal_t


def get_data(data_folder):

    print('Finding the Challenge data...')
    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    print('Extracting features and labels from the data...')

    feature_list = []
    label_list = []

    for i in range(num_records):
        #width = len(str(num_records))
        #print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])
        features = extract_CNN_features(record)  # (12, 4096)
        if features.shape != (12, 2000):
            print(f"skipping {records[i]} due to shape {features.shape}")
            continue

        feature_list.append(features.numpy())  # convert to numpy for stacking

        label = load_label(record)
        label_list.append(label)

    feature_array = np.stack(feature_list)  # shape (num_records, 4096, 12)
    one_hot_labels = tf.one_hot(label_list, depth=2, dtype=tf.float32)

    return feature_array, one_hot_labels

    return features, one_hot_labels

