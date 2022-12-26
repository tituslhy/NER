import json
from transformers import DistilBertTokenizerFast

BATCH_SIZE = 2
EPOCHS = 3
LEARNING_RATE = 0.0001
TOKENIZER = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
UNIQUE_LABELS = 17 #global configuration
SEED = 2022
PATH = 'ner.csv'

with open('ids_to_labels.json','r') as file:
    IDS_TO_LABELS = json.load(file)