import torch
from transformers import BertForTokenClassification

import config as c
from utils import train, evaluate, load_model
from getdata import df_train, df_val, df_test, unique_labels
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