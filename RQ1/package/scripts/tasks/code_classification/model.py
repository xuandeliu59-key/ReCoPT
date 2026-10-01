

import torch
import torch.nn as nn
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss

class Model(nn.Module):   
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
    
        
        self.dropout = nn.Dropout(args.dropout_probability)

    def forward(self, input_ids=None, labels=None): 
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]

        logits = self.dropout(outputs)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.long()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits