import pickle
import numpy as np
import tensorflow as tf
import os
from helper_code import *
import pandas as pd
import matplotlib.pyplot as plt


def extract_CNN_features(record):
    
    signal, fields = load_signals(record)

    #normalize 0 to 1?
    #truncate particularly large and small values?
    #truncate to same lenght

    # print(f"signal type {type(signal)}")
    # print(signal)

    signal = signal[0:2000, :]

    signal_mean = signal.mean()
    signal_std = signal.std()

    signal = (signal - signal_mean) / signal_std

    # print(signal)

    signal_t = tf.convert_to_tensor(signal,dtype=tf.float32)
    return signal_t

def extract_static_features(record):

    header = load_header(record)
    age = get_age(header)
    sex = get_sex(header)
    sex = 1 if sex == 'Male' else 0

    return (age, sex)


def get_data(data_folder, hybrid=False):

    print('Finding the Challenge data...')
    # records = find_records(data_folder)
    # num_records = len(records)

    
    sample_df = random_sample("something")
    sample_exam_ids = set(sample_df['exam_id'].astype(str))

    print('Extracting features and labels from the data...')


    wave_feature_list = []
    static_feature_list = []
    label_list = []

    for file in os.listdir(data_folder): #each file is a folder, e.g. exams_part0
        exam_folder = os.path.join(data_folder, file)
        print(f"exam folder: {exam_folder}")

        # look into each file and add any of its records in set into feature and label list
        # open and check 
        records = find_records(exam_folder)
        num_records = len(records)

        for i in range(num_records):
            record = os.path.join(exam_folder, records[i])
            
            str_record = str(records[i])
            
            if  str_record not in sample_exam_ids:
                continue
            
            features = extract_CNN_features(record)

            if features.shape != (2000,12):
                print(f"skipping {records[i]} due to shape {features.shape}")
                continue

            label = load_label(record)
            wave_feature_list.append(features.numpy())  # convert to numpy for stacking

            label_list.append(label)

            if hybrid:
                age, sex = extract_static_features(record)
                static_feature_list.append([age, sex])

    wave_feature_array = (np.stack(wave_feature_list))  # shape (num_records, 4096, 12)
    one_hot_labels = tf.one_hot(label_list, depth=2, dtype=tf.float32)
    if hybrid:
        static_array = np.array(static_feature_list, dtype=np.float32)
        return wave_feature_array, static_array, one_hot_labels
    else:
        return wave_feature_array, one_hot_labels
    

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


def plot_ecg(signal, chagas_positive=True, sampling_rate=400):
    """
    Plot the 12-lead ECG signal.
    signal: numpy array of shape (time, 12)
    sampling_rate: samples per second, default 400 Hz
    """
    leads = [
        "I", "II", "III", "aVR", "aVL", "aVF",
        "V1", "V2", "V3", "V4", "V5", "V6"
    ]

    if signal.shape[0] < signal.shape[1]:
        # (time, leads) expected; if wrong, transpose
        signal = signal.T

    num_samples = signal.shape[0]
    time_seconds = np.arange(num_samples) / sampling_rate  # time in seconds

    plt.figure(figsize=(10, 8))

    for i in range(12):
        plt.subplot(12, 1, i + 1)
        plt.plot(time_seconds, signal[:, i], linewidth=0.8)
        plt.ylabel(leads[i], fontsize=6)
        if i != 11:
            plt.xticks([])
        else:
            plt.xlabel('Time (s)', fontsize=8)
            # Set ticks every 1 second
            max_time = time_seconds[-1]
            ticks = np.arange(0, max_time + 1, 1)  # ticks at every 1 second
            plt.xticks(ticks, fontsize=6)

        plt.yticks([])

    if chagas_positive:
        title = "ECG Plot - Chagas Positive Case"
    else:
        title = "ECG Plot - Chagas Negative Case"

    plt.suptitle(title, fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def get_true(data_folder):

    # records = find_records(data_folder)
    # num_records = len(records)

    
    sample_df = random_sample("something")
    sample_exam_ids = set(sample_df['exam_id'].astype(str))

    print('Extracting features and labels from the data...')

    for file in os.listdir(data_folder): #each file is a folder, e.g. exams_part0
        exam_folder = os.path.join(data_folder, file)
        print(f"exam folder: {exam_folder}")

        # look into each file and add any of its records in set into feature and label list
        # open and check 
        records = find_records(exam_folder)
        num_records = len(records)

        for i in range(num_records):
            record = os.path.join(exam_folder, records[i])

            label = load_label(record)            
            if  label == 0:
                signal, fields = load_signals(record)
                return signal
        
    print("failed")
    return -1            

def main():
    plot_ecg(get_true("train_data/"), chagas_positive=False)


    

if __name__ == "__main__":
    main()