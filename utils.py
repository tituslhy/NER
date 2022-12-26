import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

import config as c
from getdata import labels_to_ids

BATCH_SIZE = c.BATCH_SIZE
EPOCHS = c.EPOCHS
LEARNING_RATE = c.LEARNING_RATE
TOKENIZER = c.TOKENIZER

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def load_model(model_class, name, device=None):
    if device is None:
        device = get_device()
        
    model = model_class()
    model.load_state_dict(torch.load(name, map_location=get_device()))
    model.to(get_device())
    return model
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
    
    def __init__(self, df):
        lb = [i.split() for i in df['labels'].values.tolist()]
        txt = df['text'].values.tolist()
        self.texts = [TOKENIZER(str(i),
                               padding = 'max_length',
                               max_length = 512,
                               truncation = True,
                               return_tensors = 'pt') for i in txt]
        self.labels = [align_labels(i,j) for i, j in zip(txt, lb)]
    
    def __len__(self):
        return len(self.labels)
    
    def get_batch_data(self, idx):
        return self.texts[idx]
    
    def get_batch_labels(self, idx):
        return torch.LongTensor(self.labels[idx])
    
    def __getitem__(self, idx):
        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)
        
        return batch_data, batch_labels
    
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
        
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            best_loss = val_loss
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