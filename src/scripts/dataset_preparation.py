import pandas as pd
from sklearn.preprocessing import StandardScaler

# Path to your dataset
mc_fake2_path = '../datasets/MC-Fake2/MC_Fake_dataset.csv'  # Adjust path as necessary
mc_fake2_df = pd.read_csv(mc_fake2_path)\

# Load the FakeNewsNet datasets
politifact_fake_path = '../datasets/politifact_fake.csv'
politifact_real_path = '../datasets/politifact_real.csv'
gossipcop_fake_path = '../datasets/gossipcop_fake.csv'
gossipcop_real_path = '../datasets/gossipcop_real.csv'

politifact_fake_df = pd.read_csv(politifact_fake_path)
politifact_real_df = pd.read_csv(politifact_real_path)
gossipcop_fake_df = pd.read_csv(gossipcop_fake_path)
gossipcop_real_df = pd.read_csv(gossipcop_real_path)

# Function to assign labels based on the dataset
def assign_labels_and_map_columns(df, label):
    df['labels'] = label  # Assign label (0 for fake, 1 for real)
    df['text'] = df['title']  # Use 'title' as 'text' since we don't have a separate 'text' column
    return df[['text', 'title', 'labels']]  # Retain only necessary columns

# Apply the mapping to all datasets
mc_fake2_df = mc_fake2_df[['text', 'title', 'labels']]  # Keep original labels
politifact_fake_df = assign_labels_and_map_columns(politifact_fake_df, label=0)
politifact_real_df = assign_labels_and_map_columns(politifact_real_df, label=1)
gossipcop_fake_df = assign_labels_and_map_columns(gossipcop_fake_df, label=0)
gossipcop_real_df = assign_labels_and_map_columns(gossipcop_real_df, label=1)


# Combine all datasets
combined_df = pd.concat([mc_fake2_df, politifact_fake_df, politifact_real_df, gossipcop_fake_df, gossipcop_real_df], ignore_index=True)

# Handle missing data (if any)
combined_df = combined_df.dropna()

# Normalize numerical features (if needed)
# If you have additional numerical columns from mc_fake2 like 'n_tweets' and 'n_retweets'
numerical_columns = ['n_tweets', 'n_retweets']  # Modify as per available columns in mc_fake2
if set(numerical_columns).issubset(combined_df.columns):
    scaler = StandardScaler()
    combined_df[numerical_columns] = scaler.fit_transform(combined_df[numerical_columns])

# Check the total number of records in the combined dataset
total_records = combined_df.shape[0]
print(f"Total number of records in the combined dataset: {total_records}")


# Save the combined dataset
combined_df.to_csv('combined_dataset.csv', index=False)

print("Combined dataset created successfully. Saved as 'combined_dataset.csv'.")