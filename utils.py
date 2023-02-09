import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import time
import copy
from sklearn.metrics import classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay
from torch.cuda.amp import GradScaler, autocast


def imshow(img, class_names, label):
    un_norm_img = img / 2 + 0.5 # unnormalize
    np_img = un_norm_img.numpy() # convert to numpy objects
    fig = plt.figure(figsize=(10,15))
    plt.imshow(np.transpose(np_img, (1, 2, 0)), cmap='gray')
    # plt.title(f"Label: {label} - {class_names[label]}")
    plt.show()


def train(bs, epochs, train_loader, opt, model, criterion, device):
    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)

    # start.record()

    scaler = GradScaler()

    # Training will be done for affective batch size of:
    # batch_size * gradient_accumulations
    gradient_accumulations = 16

    model.to(device)
    model.train()
    model.zero_grad()
    for epoch in range(epochs):  # loop over the dataset multiple times
        # print('EPOCH ', epoch)
        for batch_idx, batch in enumerate(train_loader):
            # print('BATCH IDX ', batch_idx)
            # get the inputs; data is a list of [images, labels]
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            # forward + backward + optimize
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
    
            scaler.scale(loss / gradient_accumulations).backward()

            if (batch_idx + 1) % gradient_accumulations == 0:
                scaler.step(opt)
                scaler.update()
                
                # zero the parameter gradients
                model.zero_grad()
            
            # images, labels = images.detach(), labels.detach()
            # torch.cuda.empty_cache()

        print(f'[EPOCH:{epoch + 1}, BATCH:{batch_idx + 1:5d}] loss: {loss/gradient_accumulations:.3f}')
        # if i % 2000 == 1999: running_loss = 0.0
        # torch.cuda.empty_cache()
    # end.record()
    print('Finished Training')
    return epoch, loss #, (start, end)


def get_classification_report(model, dataloader, device):

    with torch.no_grad():
        model.eval()
        X_batch, y_batch = next(iter(dataloader))
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        y_test_pred = model(X_batch)
        _, y_pred_tags = torch.max(y_test_pred, dim = 1)
        print(classification_report(y_pred=np.array(y_pred_tags.cpu()), y_true=y_batch.cpu()))
        print('F1 score: ', np.round((f1_score(y_pred=np.array(y_pred_tags.cpu()), y_true=y_batch.cpu()))*100, 2))
        print('Confusion matrix:')
        ConfusionMatrixDisplay(confusion_matrix(y_pred=np.array(y_pred_tags.cpu()), y_true=y_batch.cpu())).plot()


def save_model(model, epoch, loss, path, optimizer, show_model_state_dict=False, show_optimizer_state_dict=False):
    torch.manual_seed(42)
    if show_model_state_dict:
        print("Model's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    if show_optimizer_state_dict:
        print("Optimizer's state_dict:")
        for var_name in optimizer.state_dict():
            print(var_name, "\t", optimizer.state_dict()[var_name])

    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, path)