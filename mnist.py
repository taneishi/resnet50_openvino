import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import timeit
import argparse
import os

from model import ConvNet

def main(args):
    device = torch.device('cpu')

    net = ConvNet()
    net = net.to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    optimizer = torch.optim.SGD(net.parameters(), 1e-4)

    # Data loading code
    train_dataset = torchvision.datasets.MNIST(
            root='./data',
            train=True,
            transform=transforms.ToTensor(),
            download=True)

    train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True)

    for epoch in range(args.epochs):
        start = timeit.default_timer()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = net(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch [% 4d/% 4d], train loss %6.4f, %5.3fsec' % (epoch+1, args.epochs, loss.item(), timeit.default_timer() - start))

    os.makedirs('model', exist_ok=True)
    torch.save(net.state_dict(), 'model/convnet.pth')
    torch.onnx.export(net, images, 'model/convnet.onnx', verbose=False)
    print('PyTorch and ONNX models exported.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=5, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch_size', default=12, type=int, help='batch size')
    args = parser.parse_args()
    print(vars(args))

    main(args)
