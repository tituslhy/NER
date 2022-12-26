import pandas as pd
import torch
from transformers import BertForTokenClassification
from sklearn.model_selection import train_test_split

import config as c
from utils import train, evaluate, load_model

SEED = c.SEED

# Read data
df = pd.read_csv('ner.csv')
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

class BertModel(torch.nn.Module):
    def __init__(self):
        super(BertModel, self).__init__()
        self.bert = BertForTokenClassification.from_pretrained('bert-base-uncased',
                                                               num_labels = len(unique_labels))
    def forward(self, input_id, mask, label):
        output = self.bert(input_ids = input_id, 
                           attention_mask = mask,
                           labels = label,
                           return_dict = False)
        return output

model = BertModel()
train(model, df_train, df_val)
print('Final model saved!')

final_model = load_model(BertModel, 'best_model.pt')

evaluate(final_model, df_test)