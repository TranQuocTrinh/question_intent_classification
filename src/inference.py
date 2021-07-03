import torch
import torch.nn as nn
import os
from transformers import AutoTokenizer
import torch.nn.functional as F

from utils import *
from models import ModelBert

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")



def convert_line_uncased(text, maxlen=350):
    tokens_a = tokenizer.tokenize(text)[:maxlen]
    one_token = tokenizer.convert_tokens_to_ids(
        ["[CLS]"] + tokens_a + ["[SEP]"]
    )
    one_token += [0] * (maxlen - len(tokens_a))
    return one_token


def inference(model, text):
    model.eval()

    input_ids = convert_line_uncased(text)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
    outputs = model(input_ids)
    outputs = F.softmax(outputs, dim=1)
    _, class_preds = torch.max(outputs, dim=1)
    class_preds = class_preds.cpu().detach().item()
    
    return convert_class_intent(class_preds)


def main():
    model_path = "../output/model_bert_clasify_fold_1.pt"    
    model = ModelBert()
    model.to(device)
    model.load_state_dict(torch.load(model_path))

    text = "What is the contact info?"
    #text = "What can I do in <location> <time>?
    
    print("Text: ", text)
    print("Class: ", inference(model, text))


if __name__ == "__main__":
    main()
