import torchvision
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
from sklearn import metrics
from openvino.inference_engine import IECore
import argparse
import timeit

from datasets import ImageNetDataSet

def main(args):
    torch.manual_seed(123)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # Data loading code
    test_dataset = ImageNetDataSet(
            data_dir='images',
            transform=transform,
            )

    test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=True,
            drop_last=True)

    if args.mode == 'torch':
        net = torchvision.models.resnet50(pretrained=True)
        net.eval()

    elif args.mode == 'fp32' or args.mode == 'int8':
        if args.mode == 'fp32':
            model_xml = 'model/resnet-50.xml'
        elif args.mode == 'int8':
            model_xml = 'model/INT8/resnet-50.xml'

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
    y_true, y_pred = [], []
    for index, (images, labels) in enumerate(test_loader):
        start_time = timeit.default_timer()

        if args.mode == 'torch':
            with torch.no_grad():
                outputs = net(images)
        elif args.mode == 'fp32' or args.mode == 'int8':
            outputs = exec_net.infer(inputs={input_blob: images})
            outputs = torch.from_numpy(outputs[output_blob])

        loss += criterion(outputs, labels)

        y_true += labels
        y_pred += np.argmax(outputs, axis=1)
        acc = metrics.accuracy_score(y_true, y_pred)
        print('[% 3d/% 3d] test acc %5.3f' % (index, len(test_loader), acc), end='')
        print(' test loss %6.4f, %5.3fsec' % (loss.item() / len(test_loader), timeit.default_timer() - start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_requests', default=1, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--mode', choices=['torch', 'fp32', 'int8'], default='torch', type=str)
    parser.add_argument('--data_dir', default='data', type=str)
    args = parser.parse_args()
    print(vars(args))

    main(args)
