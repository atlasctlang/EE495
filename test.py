import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms
from data_pre import LFWPairs
import lightning as L

import torch.nn.functional as F

from model import *

threshold = 0.21


def test():
    dataset = LFWPairs(root="./dataset/LFW",
                       download=True,
                       transform=transforms.Compose([
                           transforms.RandomCrop(128),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                       ]))

    # dataset.targets[dataset.targets == 1] = 10

    # use 10% of training data for validation
    train_set_size = int(len(dataset) * 0.8)
    valid_set_size = int(len(dataset) * 0.1)
    test_set_size = len(dataset) - train_set_size - valid_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(88)
    _, _, test_set = data.random_split(dataset, [train_set_size, valid_set_size, test_set_size], generator=seed)

    #load image
    test_set_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    #load model
    model = LitCNN()
    checkpoint = torch.load("./checkpoints/checkpoint.pth.tar", map_location=torch.device('cpu'))
    model.model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # trainer = L.Trainer()
    # test the model
    # trainer.test(model, dataloaders=test_set_loader)

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i, (x1, x2, y) in enumerate(test_set_loader):
        y_hat_1 = model.model(x1)
        y_hat_2 = model.model(x2)
        y_hat = F.cosine_similarity(y_hat_1, y_hat_2)
        if y_hat >= threshold and y == 1:
            tp += 1
        elif y_hat >= threshold and y == -1:
            fp += 1
        elif y_hat < threshold and y == 1:
            fn += 1
        elif y_hat < threshold and y == -1:
            tn += 1
        else:
            print("error")

    print("tp: ", tp)
    print("tn: ", tn)
    print("fp: ", fp)
    print("fn: ", fn)
    print("accuracy: ", (tp + tn) / (tp + tn + fp + fn))
    print("precision: ", tp / (tp + fp))
    print("recall: ", tp / (tp + fn))
    print("f1: ", 2 * tp / (2 * tp + fp + fn))


if __name__ == '__main__':
    test()
