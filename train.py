import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms
from data_pre import LFWPairs
import lightning as L

import torch.nn.functional as F

from model import *


def train():

    dataset = LFWPairs(root="./dataset/LFW",
                       download=True,
                       transform=transforms.Compose([
                           transforms.RandomCrop(128),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                       ]))

    # dataset.targets[dataset.targets == 0] = -1

    # use 10% of training data for validation
    train_set_size = int(len(dataset) * 0.8)
    valid_set_size = int(len(dataset) * 0.1)
    test_set_size = len(dataset) - train_set_size - valid_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(88)
    train_set, valid_set, test_set = data.random_split(dataset, [train_set_size, valid_set_size, test_set_size],
                                                       generator=seed)

    #load image
    train_set_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4, persistent_workers=True)
    valid_set_loader = DataLoader(valid_set, batch_size=64, shuffle=False, num_workers=4, persistent_workers=True)

    model = LitCNN()
    checkpoint = torch.load("./checkpoints/checkpoint.pth.tar", map_location=torch.device('cpu'))
    model.model.load_state_dict(checkpoint['state_dict'])
    trainer = L.Trainer(default_root_dir="./checkpoints", max_epochs=20)
    trainer.fit(model, train_set_loader, valid_set_loader)


if __name__ == '__main__':
    train()
