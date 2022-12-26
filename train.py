import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from transformers import DistilBertForTokenClassification

import config as c
from utils import get_device, load_model

import argparse

#Instantiate parser
parser = argparse.ArgumentParser()

#Required parameters
parser.add_argument("--batch", type = int, required = False)
parser.add_argument("--epochs", type = int, required = False)
parser.add_argument("--lr", type = float, required = False)
parser.add_argument("--path", type = str, required = False)

args = parser.parse_args()

#Instantiate config
if args.batch:
    BATCH_SIZE = args.batch
else:
    BATCH_SIZE = c.BATCH_SIZE

if args.epochs:
    EPOCHS = args.epochs
else:
    EPOCHS = c.EPOCHS

if args.lr:
    LEARNING_RATE = args.lr
else:
    LEARNING_RATE = c.LEARNING_RATE

if args.path:
    PATH = args.path
else:
    PATH = c.PATH

TOKENIZER = c.TOKENIZER
SEED = c.SEED

#Load data
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

### Definitions of functions and models ###
def align_labels(texts, 
                 labels, 
                 labels_to_ids = labels_to_ids, 
                 label_all_tokens = False, 
                 tokenizer = TOKENIZER):
    """
    Align the labels to corresponding words post tokenization.

    Args:
    texts: list of strings
    labels: list of strings
    labels_to_ids: dictionary mapping labels to ids
    label_all_tokens: if True, all tokens are treated as labels
    tokenizer: a function to turn strings into ids

    Returns:
    label_ids
    """
    tokenized_inputs = tokenizer(texts, padding = 'max_length',
                                 max_length = 512, truncation = True)
    word_ids = tokenized_inputs.word_ids()
    previous_word_idx, label_ids = None, []
    
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
            
        elif word_idx != previous_word_idx:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]])
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]] if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        
        previous_word_idx = word_idx
    
    return label_ids

class DataSequence(torch.utils.data.Dataset):
    '''
    Instantiates a custom Torch dataset object for batch loading during model training
    and evaluation.
    '''
    def __init__(self, df):
        '''
        Class constructor. Takes in the dataframe of interest
        The dataframe must contain the 'text' and 'labels' column.
        '''
        lb = [i.split() for i in df['labels'].values.tolist()]
        txt = df['text'].values.tolist()
        self.texts = [TOKENIZER(str(i),
                               padding = 'max_length',
                               max_length = 512,
                               truncation = True,
                               return_tensors = 'pt') for i in txt]
        self.labels = [align_labels(texts,labels) for texts, labels in zip(txt, lb)]
    
    def __len__(self):
        '''
        Returns the length of the dataset.
        '''
        return len(self.labels)
    
    def get_batch_data(self, idx):
        '''
        Returns the batch data for the given index.
        '''
        return self.texts[idx]
    
    def get_batch_labels(self, idx):
        '''
        Returns the batch labels as a torch tensor object
        for the given index
        '''
        return torch.LongTensor(self.labels[idx])
    
    def __getitem__(self, idx):
        '''
        Returns the batch data and labels for a given index
        '''
        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)
        
        return batch_data, batch_labels

class DistilBertModel(torch.nn.Module):
    """DistilBertModel class for token classification

    Args:
        unique_labels: Takes in the list of unique labels as input
    """
    def __init__(self, unique_labels = unique_labels):
        super(DistilBertModel, self).__init__()
        self.bert = DistilBertForTokenClassification.from_pretrained('distilbert-base-uncased',
                                                               num_labels = len(unique_labels))
    def forward(self, input_id, mask, label):
        output = self.bert(input_ids = input_id, 
                           attention_mask = mask,
                           labels = label,
                           return_dict = False)
        return output

def train(model, df_train, df_val, batch_size = BATCH_SIZE,
          epochs = EPOCHS, learning_rate = LEARNING_RATE):
    """Function to finetune BERT model

    Args:
        model (model_class): BERT model
        df_train (dataframe): train dataset
        df_val (dataframe): _test dataset
        batch_size (integer, optional): Batch size for dataloader. Defaults to BATCH_SIZE.
        epochs (integer, optional): Number of epochs to train. Defaults to EPOCHS.
        learning_rate (float, optional): Learning rate for training. Defaults to LEARNING_RATE.
    """
    train_dataset = DataSequence(df_train)
    val_dataset = DataSequence(df_val)
    
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size)
    
    device = get_device()
    print(f'Device: {device}')
    model.to(device)
    
    optimizer = Adam(model.parameters(), lr = learning_rate)
    
    best_acc, best_loss = 0, 1000
    
    for epoch in range(epochs):
        total_acc_train, total_loss_train = 0, 0
        model.train()
        
        for train_data, train_label in tqdm(train_loader):
            train_label = train_label.to(device)
            mask = train_data['attention_mask'].squeeze(1).to(device)
            input_id = train_data['input_ids'].squeeze(1).to(device)
            
            optimizer.zero_grad()
            loss, logits = model(input_id, mask, train_label)
            
            for i in range(logits.shape[0]):
                logits_clean = logits[i][train_label[i] != -100]
                label_clean = train_label[i][train_label[i] != -100]
                
                predictions = logits_clean.argmax(dim = 1)
                acc = (predictions == label_clean).float().mean()
                total_acc_train += acc
                total_loss_train += loss.item()
            
            loss.backward()
            optimizer.step()
        
        model.eval()
        
        total_acc_val, total_loss_val = 0, 0
        
        for val_data, val_label in tqdm(val_loader):
            val_label = val_label.to(device)
            mask = val_data['attention_mask'].squeeze(1).to(device)
            input_id = val_data['input_ids'].squeeze(1).to(device)
            
            loss, logits = model(input_id, mask, val_label)
            
            for i in range(logits.shape[0]):
                logits_clean = logits[i][val_label[i]!= -100]
                label_clean = val_label[i][val_label[i]!= -100]
                
                predictions = logits_clean.argmax(dim = 1)
                acc = (predictions == label_clean).float().mean()
                total_acc_val += acc
                total_loss_val += loss.item()
        
        val_accuracy = total_acc_val/len(df_val)
        val_loss = total_loss_val/len(df_val)
        
        print(f'Epoch: {epoch+1} | Validation Loss: {val_loss:.4f} | Validation accuracy: {val_accuracy:.4f}')
        
        if val_accuracy > best_acc and val_loss < best_loss:
            best_acc, best_loss = val_accuracy, val_loss
            torch.save(model.state_dict(), "best_model.pt")

def evaluate(model, df_test):
    test_dataset = DataSequence(df_test)
    test_loader = DataLoader(test_dataset, batch_size = 1)
    device = get_device()
    print(f'Device used: {device}')
    
    total_acc_test = 0.0
    
    for test_data, test_label in test_loader:
        test_label = test_label.to(device)
        mask = test_data['attention_mask'].squeeze(1).to(device)
        input_id = test_data['input_ids'].squeeze(1).to(device)
        loss, logits = model(input_id, mask, test_label)
        
        for i in range(logits.shape[0]):
            logits_clean = logits[i][test_label[i] != -100]
            label_clean = test_label[i][test_label[i]!= -100]
            predictions = logits_clean.argmax(dim=1)
            acc = (predictions==label_clean).float().mean()
            total_acc_test += acc
    
    print(f'Test accuracy: {total_acc_test/len(df_test):.3f}')

##############
            
model = DistilBertModel()
train(model, df_train, df_val)
print('Final model saved!')

final_model = load_model(DistilBertModel, 'best_model.pt')

evaluate(final_model, df_test)