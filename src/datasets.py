import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from utils import *


class SequenceDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.maxlen = 350

    def convert_line_uncased(self, text):
        tokens_a = self.tokenizer.tokenize(text)[:self.maxlen]
        one_token = self.tokenizer.convert_tokens_to_ids(
            ["[CLS]"] + tokens_a + ["[SEP]"]
        )
        one_token += [0] * (self.maxlen - len(tokens_a))
        return one_token
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        text = self.data.loc[index, "Utterance"]
        tokens_text = self.convert_line_uncased(text)
        labels = convert_intent_class(self.data.loc[index, "Intent"])

        return torch.tensor(tokens_text), torch.tensor(labels)



if __name__ == "__main__":
    val_df = pd.read_csv("../generate/question_generate_val.csv")
    print("Validation:", val_df.shape)
    val_dataset = SequenceDataset(data=val_df)
    print(val_dataset[0])
