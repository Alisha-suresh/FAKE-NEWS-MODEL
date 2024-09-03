import pandas as pd

# Load your preprocessed dataset
df = pd.read_csv('datasets/combined_dataset.csv')

class_counts = df['labels'].value_counts()

# Print the number of real and fake news samples
print(f"Real news samples (0): {class_counts[0]}")
print(f"Fake news samples (1): {class_counts[1]}")