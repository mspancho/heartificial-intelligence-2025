import pickle
import numpy as np
import tensorflow as tf
import os
from helper_code import *
import pandas as pd

def extract_CNN_features(record):
    
    signal, fields = load_signals(record)

    #normalize 0 to 1?
    #truncate particularly large and small values?
    #truncate to same lenght

    # print(f"signal type {type(signal)}")
    # print(signal)

    signal = signal[0:2000]

    signal_mean = signal.mean()
    signal_std = signal.std()

    signal = (signal - signal_mean) / signal_std

    # print(signal)

    signal_t = tf.transpose(signal)
    return signal_t


def get_data(data_folder):

    print('Finding the Challenge data...')
    records = find_records(data_folder)
    num_records = len(records)

    sample_list = random_sample("something")

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    print('Extracting features and labels from the data...')

    feature_list = []
    label_list = []

    for i in range(num_records):

        record = os.path.join(data_folder, records[i])
        features = extract_CNN_features(record)  # (12, 4096)
        if features.shape != (12, 2000):
            print(f"skipping {records[i]} due to shape {features.shape}")
            continue

        if()
        feature_list.append(features.numpy())  # convert to numpy for stacking
        label = load_label(record)
        print(label)
        label_list.append(label)

    feature_array = np.stack(feature_list)  # shape (num_records, 4096, 12)
    one_hot_labels = tf.one_hot(label_list, depth=2, dtype=tf.float32)

    return feature_array, one_hot_labels


def random_sample(input_file):
    INPUT_FILE = 'code15_input/code15_chagas_labels.csv'   # Replace with your actual input file
    OUTPUT_FILE = 'filtered.csv'  # Output file name


    df = pd.read_csv(INPUT_FILE)

    print("Unique 'chagas' values:", df['chagas'].unique())

    true_df = df[df['chagas'] == True]

    num_true = len(true_df)
    print(f"Found {num_true} true samples.")

    false_df = df[df['chagas'] == False]

    num_false_to_sample = 2 * num_true
    sampled_false_df = false_df.sample(n=num_false_to_sample, random_state=42)

    combined_df = pd.concat([true_df, sampled_false_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Filtered dataset saved to {OUTPUT_FILE} with {len(combined_df)} samples.")

    return combined_df
