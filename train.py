import torchvision
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
import argparse
import timeit
import os

from datasets import ImagesDataset

def main(args):
    torch.manual_seed(123)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # Data loading code
    train_dataset = ImagesDataset(
            data_dir=args.data_dir,
            transform=transform,
            )

    train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=True,
            drop_last=True)

    net = torchvision.models.resnet50()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(args.epochs):
        epoch_start = timeit.default_timer()

        y_true = torch.FloatTensor()
        y_pred = torch.FloatTensor()
        net.train()
        for index, (images, labels) in enumerate(train_loader):
            print('\rbatch %d/%d' % (index, len(train_loader)), end='')

            # forward pass
            outputs = net(images)
            loss = criterion(outputs, labels)

            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = outputs.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            y_true = torch.cat((y_true, labels))
            y_pred = torch.cat((y_pred, pred))

        train_acc = accuracy_score(y_true, y_pred)
        print('\repoch % 5d train loss %6.4f acc %5.3f' % (epoch+1, loss.item(), train_acc), end='')

        print(' %5.3fsec' % (timeit.default_timer() - epoch_start))

    os.makedirs(args.model_dir, exist_ok=True)
    torch.save(net.state_dict(), '%s/%s.pth' % (args.model_dir, args.model_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=5, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch_size', default=96, type=int, help='batch size')
    parser.add_argument('--data_dir', default='images', type=str)
    parser.add_argument('--name', default='convnet', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--model_dir', default='model', type=str)
    parser.add_argument('--model_name', default='resnet-50', type=str)
    args = parser.parse_args()
    print(vars(args))

    main(args)
