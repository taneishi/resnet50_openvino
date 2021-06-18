import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import timeit
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=2, type=int, metavar='N', help='number of total epochs to run')
    args = parser.parse_args()
    train(0, args)

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

def train(process, args):
    device = torch.device('cpu')

    model = ConvNet()
    model = model.to(device)

    batch_size = 100

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    # Data loading code
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=True)

    start = timeit.default_timer()
    total_step = len(train_loader)
    for epoch in range(args.epochs):
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
            if (i + 1) % 100 == 0 and process == 0:
                print('Epoch [%4d/%4d], Step [%4d/%4d], Loss: %6.4f' %
                        (epoch+1, args.epochs, i+1, total_step, loss.item()))

    if process == 0:
        print('Training complete in: %5.3f' % (timeit.default_timer() - start))

if __name__ == '__main__':
    main()
