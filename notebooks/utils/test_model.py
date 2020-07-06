from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()

def test_model(model, testloader):
    correct = 0
    total = 0
    since = time.time()
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        time_elapsed = time.time() - since
        print('Accuracy of the model on {} test images is {:.2f}'.format(len(testloader)*4, 100*correct/total))
        print('Time elapsed : {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed % 60))
