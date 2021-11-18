import torchvision
import torch
import torch.nn as nn
from openvino.inference_engine import IECore
import argparse
import timeit

def main(args):
    torch.manual_seed(123)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = torchvision.datasets.CIFAR10(
            root=args.data_dir,
            train=False,
            transform=transform,
            download=True)

    test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=True,
            drop_last=True)

    if args.mode == 'pytorch':
        net = torchvision.models.resnet50()
        net.load_state_dict(torch.load('public/resnet-50-pytorch/resnet50-19c8e357.pth'))
        net.eval()

    elif args.mode == 'fp32' or args.mode == 'int8':
        if args.mode == 'fp32':
            model_xml = 'public/resnet-50-pytorch/FP32/resnet-50-pytorch.xml'
        elif args.mode == 'int8':
            model_xml = 'public/resnet-50-pytorch/FP32/resnet-50-pytorch.xml'

        model_bin = model_xml.replace('xml', 'bin')

        print('Creating Inference Engine')
        ie = IECore()
        net = ie.read_network(model=model_xml, weights=model_bin)

        # loading model to the plugin
        print('Loading model to the plugin')
        exec_net = ie.load_network(network=net, num_requests=args.num_requests, device_name='CPU')

        print('Preparing input blobs')
        input_blob = next(iter(net.input_info))
        output_blob = next(iter(net.outputs))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    loss = 0
    start_time = timeit.default_timer()
    for index, (images, labels) in enumerate(test_loader):

        if args.mode == 'pytorch':
            with torch.no_grad():
                outputs = net(images)
        elif args.mode == 'fp32' or args.mode == 'int8':
            outputs = exec_net.infer(inputs={input_blob: images})
            outputs = torch.from_numpy(outputs[output_blob])

        loss += criterion(outputs, labels)

        print('test loss %6.4f, %5.3fsec' % (loss.item() / (index + 1), timeit.default_timer() - start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ResNet-50 Ineference')
    parser.add_argument('--num_requests', default=1, type=int)
    parser.add_argument('--batch_size', default=96, type=int)
    parser.add_argument('--mode', choices=['pytorch', 'fp32'], default='pytorch', type=str)
    parser.add_argument('--data_dir', default='data', type=str)
    args = parser.parse_args()
    print(vars(args))

    main(args)
