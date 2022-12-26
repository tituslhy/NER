from transformers import DistilBertTokenizerFast

BATCH_SIZE = 2
EPOCHS = 3
LEARNING_RATE = 0.0001
TOKENIZER = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
SEED = 2022
PATH = 'ner.csv'