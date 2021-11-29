import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from openvino.inference_engine import IECore, StatusCode
import timeit
import argparse
import os

from model import ConvNet

def main(args):
    model_xml = 'model/INT8/convnet.xml'

    model_bin = model_xml.replace('xml', 'bin')

    print('Creating Inference Engine')
    ie = IECore()
    net = ie.read_network(model=model_xml, weights=model_bin)

    print('Preparing input blobs')
    input_blob = next(iter(net.input_info))
    output_blob = next(iter(net.outputs))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    # Data loading code
    test_dataset = torchvision.datasets.CIFAR10(
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

    # loading model to the plugin
    print('Loading model to the plugin')
    exec_net = ie.load_network(network=net, num_requests=len(test_loader), device_name='CPU')

    loss = 0
    start_time = timeit.default_timer()
    for index, (images, labels) in enumerate(test_loader):
        exec_net.requests[index].async_infer(inputs={input_blob: images})

    output_queue = list(range(args.num_requests))

    # wait the latest inference executions
    while True:
        for index in output_queue:
            infer_status = exec_net.requests[index].wait(0)

            if infer_status == StatusCode.RESULT_NOT_READY:
                continue

            if infer_status == StatusCode.OK:
                outputs = exec_net.requests[index].output_blobs[output_blob].buffer
                outputs = torch.from_numpy(outputs)
                loss += criterion(outputs, labels)

                output_queue.remove(index)

        if len(output_queue) == 0:
            break

    print('test loss %6.4f, %5.3fsec' % (loss.item() / len(test_loader), (timeit.default_timer() - start_time)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_requests', default=1, type=int)
    parser.add_argument('--batch_size', default=96, type=int)
    parser.add_argument('--data_dir', default='data', type=str)
    args = parser.parse_args()
    print(vars(args))

    main(args)
