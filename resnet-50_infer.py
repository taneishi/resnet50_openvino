import torchvision
import torch
import torch.nn as nn
from openvino.inference_engine import IECore
import argparse
import timeit

def main(args):
    torch.manual_seed(10)

    device = torch.device('cpu')

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
        net = net.to(device)
        net.eval()

    elif args.mode == 'fp32' or args.mode == 'fp16':
        if args.mode == 'fp32':
            model_xml = 'public/resnet-50-pytorch/FP32/resnet-50-pytorch.xml'
        elif args.mode == 'fp16':
            model_xml = 'public/resnet-50-pytorch/FP16/resnet-50-pytorch.xml'
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

    criterion = nn.CrossEntropyLoss()

    test_loss = 0
    for index, (data, target) in enumerate(test_loader):
        start_time = timeit.default_timer()

        if args.mode == 'pytorch':
            with torch.no_grad():
                outputs = net(data)
        elif args.mode == 'fp32' or args.mode == 'fp16':
            outputs = exec_net.infer(inputs={input_blob: data})
            outputs = torch.from_numpy(outputs[output_blob])

        test_loss = criterion(outputs, target)

        print('[% 4d/% 4d] test loss: %6.3f, %5.2fsec' % (index, len(test_loader), test_loss.item(), timeit.default_timer() - start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ResNet-50 Ineference')
    parser.add_argument('--batch_size', default=96, type=int)
    parser.add_argument('--mode', choices=['pytorch', 'fp32', 'fp16'], default='pytorch', type=str)
    parser.add_argument('--data_dir', default='datasets/cifar10', type=str)
    args = parser.parse_args()
    print(vars(args))

    main(args)
