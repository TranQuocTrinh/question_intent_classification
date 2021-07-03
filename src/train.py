import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch import optim
import random
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix

from utils import *
from datasets import SequenceDataset
from models import ModelBert

matplotlib.style.use("ggplot")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(model, train_loader, criterion, optimizer):
    model.train()
    train_running_loss = .0
    correct, total = 0, 0
    bar = tqdm(train_loader, total=len(train_loader), desc="Training...")
    for input_ids, target in bar:
        input_ids, target = input_ids.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        train_running_loss += loss.item()

        outputs = F.softmax(outputs, dim=1)
        _, pred = torch.max(outputs, dim=1)
        correct += torch.sum(pred == target).item()
        total += target.size(0)

        bar.set_postfix(batch_loss=loss.item())

    train_loss = train_running_loss/len(train_loader)
    train_accuracy = correct/total
    return train_loss, train_accuracy


def validate(model, val_loader, criterion, optimizer):
    model.eval()
    val_running_loss = 0
    correct, total = 0, 0
    y_true_val, y_pred_val = [], []
    bar = tqdm(val_loader, total=len(val_loader), desc="Validating...")
    with torch.no_grad():
        for input_ids, target in bar: 
            input_ids, target = input_ids.to(device), target.to(device)
            outputs = model(input_ids)
            loss = criterion(outputs, target)
            val_running_loss += loss.item()

            outputs = F.softmax(outputs, dim=1)
            _, pred = torch.max(outputs, dim=1)
            correct += torch.sum(pred == target).item()
            total += target.size(0)

            y_true_val += target.cpu().tolist()
            y_pred_val += pred.tolist()

            bar.set_postfix(batch_loss=loss.item())
           
    confu = confusion_matrix(y_true_val, y_pred_val)

    val_loss = val_running_loss/len(val_loader)
    val_accuracy = correct/total
    return val_loss, val_accuracy, confu, y_pred_val


def run(train_df,
    val_df,
    n_epochs, 
    batch_size,
    model_output_path,
    confusion_matrix_output_path,
    val_pred_output_path,
    plot_loss_path):

    # train_df = pd.read_csv(train_df_path)
    # val_df = pd.read_csv(val_df_path)
    print("Train:", train_df.shape)
    print("Validation:", val_df.shape)

    train_dataset = SequenceDataset(data=train_df)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = SequenceDataset(data=val_df)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_loss, val_loss = [], []
    max_acc = -np.Inf

    model = ModelBert()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()

    print("Start trainning")
    for epoch in range(1, n_epochs+1):
        print(f"\nEpoch = [{epoch}]/[{n_epochs}]\n")
        train_epoch_loss, train_epoch_acc = train(
            model, train_loader, criterion, optimizer
        )
        train_loss.append(train_epoch_loss)
        print(f"train loss: {train_epoch_loss:.2f} | train acc: {train_epoch_acc*100:.2f}%")

        val_epoch_loss, val_epoch_acc, confusion_matrix, y_pred_val = validate(
            model, val_loader, criterion, optimizer
        )
        val_loss.append(val_epoch_loss)
        print(f"validation loss: {val_epoch_loss:.2f} | val acc: {val_epoch_acc*100:.2f}%")

        # Saving the best weight
        if val_epoch_acc > max_acc:
            max_acc = val_epoch_acc
            torch.save(model.state_dict(), model_output_path)
            np.save(confusion_matrix_output_path, confusion_matrix)
            print("Detected network improvement, saving current model")
            print(f"Best accuracy on validation set is {max_acc*100:.2f}%")

            y_pred_val = [convert_class_intent(class_) for class_ in y_pred_val]
            val_df["IntentPred"] = y_pred_val
            val_df.to_csv(val_pred_output_path)
    plot_train_val_loss(train_loss, val_loss, plot_loss_path)

    return max_acc
        


def main():
    if not os.path.exists("../output/"):
        os.makdir("../output")
    f = open("../output/accuracy.txt", "w")
    
    gpt3_df = pd.read_csv("../data/gpt3-questions.csv")
    for i in range(5):
        print(f"Fold = {i+1}")
        train_df = pd.read_csv(f"../data/train_df_fold_{i+1}.csv")
        val_df = pd.read_csv(f"../data/val_df_fold_{i+1}.csv")

        train_df = pd.concat([train_df, gpt3_df]).reset_index(drop=True)
        fold_acc = run(train_df=train_df, 
            val_df=val_df,
            n_epochs=5, 
            batch_size=16,
            model_output_path=f"../output/model_bert_clasify_fold_{i+1}.pt",
            confusion_matrix_output_path=f"../output/confusion_fold_{i+1}.npy",
            val_pred_output_path=f"../output/pred_val_df_fold_{i+1}.csv",
            plot_loss_path=f"../output/ploss_loss_fold_{i+1}"
        )
        f.write(f"Best accuracy fold {i+1} is: {fold_acc*100:.2f}%\n")
    
    f.close()

if __name__ == "__main__":
    main()
