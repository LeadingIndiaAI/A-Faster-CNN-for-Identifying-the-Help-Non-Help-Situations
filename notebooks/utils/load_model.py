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


def load_model(state_dir=None):
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)
    if state_dir:
        print(model.load_state_dict(torch.load(f'../models/{state_dir}')))

    return model