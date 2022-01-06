import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from sklearn import metrics
from openvino.inference_engine import IECore, StatusCode
import timeit
import argparse
import os

from datasets import ImagesDataset

def main(args):
    torch.manual_seed(123)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # Data loading code
    test_dataset = ImagesDataset(
            data_dir=args.data_dir,
            transform=transform,
            )

    test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=True,
            drop_last=True)

    model_xml = '%s/INT8/%s.xml' % (args.model_dir, args.model_name)
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

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss(reduction='mean')

    queue = [0] * args.num_requests
    labels = list(range(args.num_requests))
    y_true, y_pred = [], []
    loss = 0

    start_time = timeit.default_timer()
    # wait the latest inference executions
    while len(y_true) < len(test_loader):
        for index in range(args.num_requests):
            if len(y_true) == len(test_loader):
                break

            if queue[index] == 0: 
                images, labels[index] = next(iter(test_loader))
                exec_net.requests[index].async_infer(inputs={input_blob: images})
                queue[index] = 1

            infer_status = exec_net.requests[index].wait(0)

            if infer_status == StatusCode.RESULT_NOT_READY:
                continue

            if infer_status == StatusCode.OK:
                outputs = exec_net.requests[index].output_blobs[output_blob].buffer
                outputs = torch.from_numpy(outputs)
                loss += criterion(outputs, labels[index])
                y_true.append(labels[index])
                y_pred.append(np.argmax(outputs, axis=1))
                queue[index] = 0

    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    acc = metrics.accuracy_score(y_true, y_pred)
    print('test acc %5.3f test loss %6.4f,' % (acc, (loss.item() / len(test_loader))), end='')
    print(' %5.3fsec' % (timeit.default_timer() - start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_requests', default=4, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--data_dir', default='images', type=str)
    parser.add_argument('--model_dir', default='model', type=str)
    parser.add_argument('--model_name', default='resnet-50', type=str)
    args = parser.parse_args()
    print(vars(args))

    main(args)
