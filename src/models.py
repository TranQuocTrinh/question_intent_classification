import math
import torch
import torch.nn as nn
import torch.nn.functional as F


from transformers import AutoModel
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ModelBert(nn.Module):
    
    def __init__(self):
        super(ModelBert, self).__init__()
        self.text_encoder = AutoModel.from_pretrained("bert-base-uncased")
        self.linear = nn.Linear(768, 18)
    
    def forward(self, input_ids):
        bert_output = self.text_encoder(input_ids, attention_mask=input_ids != 0)
        text_vec = bert_output[0][:, 0, :]
        x = self.linear(text_vec)
        return x
