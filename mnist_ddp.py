import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import timeit
import argparse
import os

from model import ConvNet

def main(args):
    torch.manual_seed(10)

    world_size = int(os.environ[args.env_size]) if args.env_size in os.environ else 1
    local_rank = int(os.environ[args.env_rank]) if args.env_rank in os.environ else 0

    device = torch.device('cpu')

    if world_size > 1:
        print('rank: {}/{}'.format(local_rank+1, world_size))
        torch.distributed.init_process_group(
                backend='gloo',
                init_method='file://%s' % args.tmpname,
                rank=local_rank,
                world_size=world_size)

    net = ConvNet()
    net = net.to(device)
    # Wrap the model
    if world_size > 1:
        net = nn.parallel.DistributedDataParallel(net)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), 1e-4)

    # Data loading code
    train_dataset = torchvision.datasets.MNIST(
            root=args.data_dir,
            train=True,
            transform=transforms.ToTensor(),
            download=True)

    train_sampler = None
    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=True,
            sampler=train_sampler)

    for epoch in range(args.epochs):
        start = timeit.default_timer()
        net.train()
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

        if local_rank == 0:
            print('epoch [% 4d/% 4d], train loss %6.4f, %5.3fsec' % (epoch+1, args.epochs, loss.item(), timeit.default_timer() - start))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_size', default='WORLD_SIZE', type=str)
    parser.add_argument('--env_rank', default='RANK', type=str)
    parser.add_argument('--tmpname', default='tmpfile', type=str)
    parser.add_argument('--num_threads', default=None, type=int, help='number of threads')
    parser.add_argument('--epochs', default=5, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch_size', default=96, type=int, help='batch size')
    parser.add_argument('--data_dir', default='data', type=str)
    args = parser.parse_args()
    if not args.num_threads:
        args.num_threads = torch.get_num_threads()
    else:
        torch.set_num_threads(args.num_threads)
    print(vars(args))
    print(vars(args))

    main(args)
