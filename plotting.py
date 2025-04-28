import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from helper_code import load_signals
import pandas as pd

def extract_ecg_signal(record_path, max_length=2000):
    """
    Load and preprocess ECG signal from a record.
    Normalizes and truncates the signal to max_length.
    """
    signal, fields = load_signals(record_path)

    signal = signal[:max_length]

    signal_mean = signal.mean()
    signal_std = signal.std()
    signal = (signal - signal_mean) / signal_std

    signal = tf.transpose(signal)
    return signal.numpy()

def plot_ecg(signal, record_name=None):
    """
    Plot the 12-lead ECG signal.
    signal: numpy array of shape (12, 2000)
    """
    leads = [
        "I", "II", "III", "aVR", "aVL", "aVF",
        "V1", "V2", "V3", "V4", "V5", "V6"
    ]

    plt.figure(figsize=(20, 15))
    for i in range(12):
        plt.subplot(12, 1, i + 1)
        plt.plot(signal[i], linewidth=1)
        plt.ylabel(leads[i])
        plt.xticks([])
        if i != 11:
            plt.gca().set_xticklabels([])
    plt.xlabel('Time')
    if record_name:
        plt.suptitle(f'ECG Record: {record_name}', fontsize=16)
    plt.tight_layout()
    plt.show()

def main():

    df1 = pd.read_csv('code15_input/code15_chagas_labels.csv')
    df2 = pd.read_csv('code15_input/exams.csv')

    # Merge based on 'exam_id'
    merged_df = pd.merge(df1, df2, on='exam_id', how='inner') 

    # Save the merged file
    merged_df.to_csv('merged_file.csv', index=False)

def random_sample(input_file):
    INPUT_FILE = 'code15_input/code15_chagas_labels.csv' 
    OUTPUT_FILE = 'filtered.csv' 


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

if __name__ == '__main__':
    main()