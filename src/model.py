import config
import transformers
import torch.nn as nn
import torch

from transformers import BertModel

# class BertClassifier(nn.Module):
    
#     def __init__(self):  # , bert: BertModel, num_classes: int
#         super(BertClassifier, self).__init__()
#         self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
#         self.dropout = nn.Dropout(0.3)
#         self.linear_out = nn.Linear(self.bert.config.hidden_size, 4)
        
#     def forward(self, input_ids, attention_mask=None, token_type_ids=None, 
#                 position_ids=None, head_mask=None, labels=None):

#         outputs = self.bert(input_ids,
#                                attention_mask=attention_mask,
#                                token_type_ids=token_type_ids,
#                                position_ids=position_ids,
#                                head_mask=head_mask)

#         cls_output = outputs[1] # batch, hidden
#         # cls_output = self.dropout(cls_output)
#         cls_output = self.linear_out(cls_output) # batch, 4
#         cls_output = torch.sigmoid(cls_output)
#         criterion = nn.BCELoss()
#         loss = 0
#         if labels is not None:
#             loss = criterion(cls_output, labels) #.view(-1, 4)
#         return loss, cls_output

class BertClassifier(nn.Module):
    
    def __init__(self, bert: BertModel, 
                        num_classes: int):

        super().__init__()
        self.bert = bert
        # self.dropout = nn.Dropout(0.3)
        self.linear_out = nn.Linear(bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, 
                position_ids=None, head_mask=None, labels=None):

        outputs = self.bert(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)

        cls_output = outputs[1] # batch, hidden
        # cls_output = self.dropout(cls_output)
        cls_output = self.linear_out(cls_output) # batch, 4
        cls_output = torch.sigmoid(cls_output)
        criterion = nn.BCELoss()
        loss = 0
        if labels is not None:
            loss = criterion(cls_output, labels) #.view(-1, 4)
        return loss, cls_output