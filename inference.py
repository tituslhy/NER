import torch
from transformers import DistilBertForTokenClassification

import config as c
from utils import get_device, load_model

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--txts', 
                    type = str, 
                    required = True,
                    help = "Sentence to run model inference")
args = parser.parse_args()

### Instantiate configuration
SENTENCE = args.txts
TOKENIZER = c.TOKENIZER
UNIQUE_LABELS = c.UNIQUE_LABELS
IDS_TO_LABELS = c.IDS_TO_LABELS

### Functions and model class ###
def align_word_ids(texts, 
                   tokenizer = TOKENIZER,
                   label_all_tokens = False):
    """Aligns labels for text after tokenization

    Args:
        texts (str): Takes in the user given text to run model inference
        tokenizer (class): Defaults to the DistilBertTokenizer.
    """
    tokenized_inputs = tokenizer(texts,
                                 padding = 'max_length',
                                 max_length = 512,
                                 truncation = True)
    word_ids = tokenized_inputs.word_ids()
    previous_word_idx, label_ids = None, []
    
    for word_idx in word_ids:
        
        if word_idx is None:
            label_ids.append(-100)
        
        elif word_idx != previous_word_idx:
            try:
                label_ids.append(1)
            except:
                label_ids.append(-100)
        
        else:
            try:
                label_ids.append(1 if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        
        previous_word_idx = word_idx
    
    return label_ids
    
class DistilBertModel(torch.nn.Module):
    """DistilBertModel class for token classification

    Args:
        unique_labels: Takes in the list of unique labels as input
    """
    def __init__(self, unique_labels = UNIQUE_LABELS):
        super(DistilBertModel, self).__init__()
        self.bert = DistilBertForTokenClassification.from_pretrained('distilbert-base-uncased',
                                                               num_labels = len(unique_labels))
    def forward(self, input_id, mask, label):
        output = self.bert(input_ids = input_id, 
                           attention_mask = mask,
                           labels = label,
                           return_dict = False)
        return output

model = load_model('best.pt', model_class = DistilBertModel)

def evaluate_one_text(model, 
                      sentence = SENTENCE, 
                      tokenizer = TOKENIZER,
                      ids_to_labels = IDS_TO_LABELS):
    """Function evaluates the sentence for named entities

    Args:
        model (class): This is the DistilBertModel
        sentence (str): String for evaluation
        tokenizer (class): Tokenizer for sentence tokenization. 
        ids_to_labels (dictionary): Defaults to ids_to_labels in the ner.csv file.
    """
    device = get_device()
    text = tokenizer(sentence, padding = 'max_length',
                     max_length = 512, truncation = True,
                     return_tensors = 'pt')
    mask = text['attention_mask'].to(device)
    input_id = text['input_ids'].to(device)
    label_ids = torch.Tensor(align_word_ids(sentence)).unsqueeze(0).to(device)
    
    logits = model(input_id, mask, None)
    logits_clean = logits[0][label_ids != -100]
    
    predictions = logits_clean.argmax(dim = 1).tolist()
    prediction_label = [ids_to_labels[i] for i in predictions]
    print(sentence)
    print(prediction_label)    