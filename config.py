from transformers import BertTokenizerFast

BATCH_SIZE = 2
EPOCHS = 5
LEARNING_RATE = 0.0001
TOKENIZER = BertTokenizerFast.from_pretrained('bert-base-uncased')
SEED = 2022