import torchvision
import torch
import torch.nn as nn
import argparse
import timeit
import os

def main(args):
    torch.manual_seed(123)
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
            pin_memory=True,
            drop_last=True)

    test_dataset = torchvision.datasets.CIFAR10(
            root=args.data_dir,
            train=False,
            transform=transform)

    test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=True)

    net = torchvision.models.resnet50()
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)

    for epoch in range(args.epochs):
        epoch_start = timeit.default_timer()

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

        print('\repoch % 5d train loss %6.4f' % (epoch+1, loss.item()), end='')

        net.eval()
        test_loss = 0
        for images, labels in test_loader:
            with torch.no_grad():
                outputs = net(images)
            test_loss += criterion(outputs, labels).item() # sum up batch loss
            pred = outputs.argmax(dim=1, keepdim=True) # get the index of the max log-probability

        print(' test average loss %6.4f' % (test_loss / len(test_loader)), end='')
        print(' %5.3fsec' % (timeit.default_timer() - epoch_start))

    os.makedirs('model', exist_ok=True)
    torch.save(net.state_dict(), 'model/%s.pth' % args.name)
    torch.onnx.export(net, images, 'model/%s.onnx' % args.name, verbose=False)
    print('PyTorch and ONNX models exported.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ResNet-50 Training')
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--batch_size', default=96, type=int)
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--name', default='resnet-50', type=str)
    parser.add_argument('--lr', default=1e-3, type=float)
    args = parser.parse_args()
    print(vars(args))

    main(args)
