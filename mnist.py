import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import timeit
import argparse
import os

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

def main(args):
    device = torch.device('cpu')

    model = ConvNet()
    model = model.to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    # Data loading code
    train_dataset = torchvision.datasets.MNIST(
           root='./data',
           train=True,
           transform=transforms.ToTensor(),
           download=True)

    train_loader = torch.utils.data.DataLoader(
           dataset=train_dataset,
           batch_size=args.batch_size,
           shuffle=True,
           num_workers=0,
           pin_memory=True)

    total_step = len(train_loader)
    for epoch in range(args.epochs):
        start = timeit.default_timer()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch [% 4d/% 4d], train loss %6.4f, %5.3fsec' % (epoch+1, args.epochs, loss.item(), timeit.default_timer() - start))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=5, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=12, type=int, help='batch size')
    args = parser.parse_args()
    print(vars(args))

    main(args)
