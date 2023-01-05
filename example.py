import os
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision import transforms
import torch.nn as nn
from sklearn.model_selection import train_test_split
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

transform = transforms.Compose([

    # gray scale
    transforms.Grayscale(),

    # resize
    transforms.Resize((128, 128)),

    # converting to tensor
    transforms.ToTensor(),

    # normalize
    transforms.Normalize((0.1307,), (0.3081,)),
])

data_dir = 'data/train/asl_alphabet_train'

# dataset
full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# train & test
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

# splitting
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)

# neural net architecture
Net(
    (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(fc1): Linear(in_features=32768, out_features=128, bias=True)
(fc2): Linear(in_features=128, out_features=29, bias=True)
(dropout): Dropout(p=0.5)
)

loss_fn = nn.CrossEntropyLoss()
# optimizer
opt = optim.SGD(model.parameters(), lr=0.01)


def train(model, train_loader, optimizer, loss_fn, epoch, device):
    # telling pytorch that training mode is on
    model.train()
    loss_epoch_arr = []

    # epochs
    for e in range(epoch):

        # bach_no, data, target
        for batch_idx, (data, target) in enumerate(train_loader):

            # moving to GPU
            # data, target = data.to(device), target.to(device)

            # Making gradints zero
            optimizer.zero_grad()

            # generating output
            output = model(data)

            # calculating loss
            loss = loss_fn(output, target)

            # backward propagation
            loss.backward()

            # stepping optimizer
            optimizer.step()

            # printing at each 10th epoch
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

            # de-allocating memory
            del data, target, output
            # torch.cuda.empty_cache()

        # appending values
        loss_epoch_arr.append(loss.item())

    # plotting loss
    plt.plot(loss_epoch_arr)
    plt.show()


train(model, trainloader, opt, loss_fn, 10, device)
