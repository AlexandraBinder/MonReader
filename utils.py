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
from sklearn.metrics import classification_report
from torch.cuda.amp import GradScaler, autocast

def get_data_csv(path):

    dfs = []
    for i, category in enumerate(os.listdir(path)):
        df = pd.DataFrame()
        df['filename'] = pd.Series(os.listdir(os.path.join(path, category)))
        # df['path'] = pd.Series(df['filename'].apply(lambda x: os.path.join(path, category, str(x))))
        df['category'] = category
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

def imshow(img, class_names, label):
    un_norm_img = img / 2 + 0.5 # unnormalize
    np_img = un_norm_img.numpy() # convert to numpy objects
    fig = plt.figure(figsize=(10,15))
    plt.imshow(np.transpose(np_img, (1, 2, 0)), cmap='gray')
    # plt.title(f"Label: {label} - {class_names[label]}")
    plt.show()

    # img = train_features[0].squeeze()
    # label = train_labels[0]


class Net(nn.Module):

    def __init__(self, device):
        super().__init__()
        # 1 input channel, 6 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        # Max pooling over a (2, 2) window
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # 5x5 from image dimension
        self.fc1 = nn.Linear(110112, 120)   #16*5*5
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)  # 2 classes
        # self.device = device

    def forward(self, x):
        # input -> conv2d -> relu -> maxpool2d -> 
        # conv2d -> relu -> maxpool2d -> 
        # view -> linear -> relu -> linear -> relu -> linear
        # x = x.to(self.device)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # need to flatten the output from the conv layer for the FC layers
        # can use x.view(-1, 16*5*5) or torch.flatten(x, 1)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(bs, epochs, train_loader, opt, model, criterion, device):
    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)

    # start.record()

    scaler = GradScaler()

    # Training will be done for affective batch size of:
    # batch_size * gradient_accumulations = 64
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

        print(f'[EPOCH:{epoch + 1}, BATCH:{batch_idx + 1:5d}] loss: {loss / 2000:.3f}')
        # if i % 2000 == 1999: running_loss = 0.0
        # torch.cuda.empty_cache()
    # end.record()
    print('Finished Training')
    return epoch, loss #, (start, end)


def visualize_model(model, dataloader, class_names, device, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        inputs, labels = next(iter(dataloader))
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        print(inputs)
        print(preds)
        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title(f'predicted: {class_names[preds[j]]}')
            plt.imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                model.train(mode=was_training)
                return
        model.train(mode=was_training)


def get_classification_report(model, dataloader, device):

    with torch.no_grad():
        model.eval()
        X_batch, y_batch = next(iter(dataloader))
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        y_test_pred = model(X_batch)
        _, y_pred_tags = torch.max(y_test_pred, dim = 1)

        print(classification_report(y_pred=np.array(y_pred_tags.cpu().numpy()), y_true=y_batch.cpu()))


def save_model(model, epoch, loss, path, optimizer, show_model_state_dict=False, show_optimizer_state_dict=False):

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