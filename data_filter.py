import pandas as pd
import numpy as np

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

combined_df.to_csv(OUTPUT_FILE, index=False)

print(f"Filtered dataset saved to {OUTPUT_FILE} with {len(combined_df)} samples.")
