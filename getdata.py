import pandas as pd
from sklearn.model_selection import train_test_split
from config import SEED, PATH

# Read data
df = pd.read_csv(PATH)

# Train-val-test split
df_train, df_no_train = train_test_split(df, test_size = 0.4, random_state = SEED)
df_val, df_test = train_test_split(df_no_train, test_size = 0.5, random_state = SEED)

# Data preprocessing
labels = [i.split() for i in df['labels'].values.tolist()]
unique_labels = set()
for label in labels:
    for i in label:
        unique_labels.add(i)

print(f'Number of unique labels: {len(unique_labels)}')

labels_to_ids = {k:v for v, k in enumerate(sorted(unique_labels))}
ids_to_labels = {v:k for v, k in enumerate(sorted(unique_labels))}