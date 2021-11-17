import torchvision
import torch
import torch.nn as nn
import argparse
import timeit
import os

from model import ConvNet

def main(args):
    torch.manual_seed(123)
    device = torch.device('cpu')

    transform = torchvision.transforms.ToTensor()

    # Data loading code
    train_dataset = torchvision.datasets.MNIST(
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

    test_dataset = torchvision.datasets.MNIST(
            root=args.data_dir,
            train=False,
            transform=transform)

    test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True)

    net = ConvNet()
    net = net.to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), 1e-4)

    for epoch in range(args.epochs):
        epoch_start = timeit.default_timer()

        net.train()
        for index, (images, labels) in enumerate(train_loader):
            print('\rbatch %d/%d' % (index, len(train_loader)), end='')

            # Forward pass
            outputs = net(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('\repoch % 5d train loss %6.4f' % (epoch+1, loss.item()), end='')

        net.eval()
        test_loss = 0
        for images, labels in test_loader:
            with torch.no_grad():
                output = net(images)
            test_loss += criterion(output, labels).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability

        print(' test average loss %6.4f' % (test_loss / len(test_loader)), end='')
        print(' %5.3fsec' % (timeit.default_timer() - epoch_start))

    os.makedirs('model', exist_ok=True)
    torch.save(net.state_dict(), 'model/convnet.pth')
    torch.onnx.export(net, images, 'model/convnet.onnx', verbose=False)
    print('PyTorch and ONNX models exported.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=5, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch_size', default=96, type=int, help='batch size')
    parser.add_argument('--data_dir', default='data', type=str)
    args = parser.parse_args()
    print(vars(args))

    main(args)
