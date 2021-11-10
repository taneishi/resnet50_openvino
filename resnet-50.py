import torchvision
import torch
import torch.nn as nn
import argparse
import timeit
import os

def main(args):
    torch.manual_seed(10)

    device = torch.device('cpu')

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
            root=args.data_dir,
            train=True,
            transform=transform,
            download=True)

    train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True)

    test_dataset = torchvision.datasets.CIFAR10(
            root=args.data_dir,
            train=False,
            transform=transform)

    test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True)

    net = torchvision.models.resnet50()
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr = args.lr, momentum=0.9)

    for epoch in range(args.epochs):
        epoch_start = timeit.default_timer()

        net.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        print('epoch: % 5d train loss: %6.4f' % (epoch, loss.item()), end='')

        net.eval()
        test_loss = 0
        for data, target in test_loader:
            with torch.no_grad():
                output = net(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability

        print(' test average loss: %6.4f' % (test_loss / len(test_loader)), end='')

        print(' %5.3fsec' % (timeit.default_timer() - epoch_start))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ResNet-50 Training')
    parser.add_argument('--data_dir', default='datasets/cifar10', type=str)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    args = parser.parse_args()
    print(vars(args))

    main(args)
