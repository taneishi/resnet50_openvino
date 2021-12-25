import torchvision
import torch
import argparse
import os

def main(args):
    torch.manual_seed(123)

    dummy_input = torch.randn(args.batch_size, 3, 224, 224)

    net = torchvision.models.resnet50(pretrained=True)
    net.train(False)

    torch_out = net(dummy_input)

    os.makedirs('model', exist_ok=True)
    torch.onnx.export(net, dummy_input, 'model/resnet-50.onnx', verbose=False)
    print('ONNX models exported.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    args = parser.parse_args()
    print(vars(args))

    main(args)
