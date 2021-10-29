import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from openvino.inference_engine import IECore
import timeit
import argparse
import os

from model import ConvNet

def main(args):
    if args.mode == 'pytorch':
        net = ConvNet()
        net.load_state_dict(torch.load('model/convnet.pth'))
        net.eval()

    elif args.mode == 'openvino':
        model_xml = 'model/convnet.xml'
        model_bin = model_xml.replace('xml', 'bin')

        print('Creating Inference Engine')
        ie = IECore()
        net = ie.read_network(model=model_xml, weights=model_bin)

        # loading model to the plugin
        print('Loading model to the plugin')
        exec_net = ie.load_network(network=net, num_requests=1, device_name='CPU')

        print('Preparing input blobs')
        input_blob = next(iter(net.input_info))
        output_blob = next(iter(net.outputs))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    # Data loading code
    test_dataset = torchvision.datasets.MNIST(
            root=args.data_dir,
            train=False,
            transform=transforms.ToTensor(),
            download=True)

    test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=True)

    for index, (images, labels) in enumerate(test_loader):
        start_time = timeit.default_timer()

        if args.mode == 'pytorch':
            with torch.no_grad():
                outputs = net(images)
        elif args.mode == 'openvino':
            outputs = exec_net.infer(inputs={input_blob: images})
            outputs = torch.from_numpy(outputs[output_blob])

        loss = criterion(outputs, labels)

        print('[% 4d/% 4d], test loss %6.4f, %5.3fsec' % (index, len(test_loader), loss.item(), (timeit.default_timer() - start_time)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=96, type=int)
    parser.add_argument('--mode', choices=['pytorch', 'openvino'], default='pytorch', type=str)
    parser.add_argument('--data_dir', default='data', type=str)
    args = parser.parse_args()
    print(vars(args))

    main(args)
