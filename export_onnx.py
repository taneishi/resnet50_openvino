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
    torch.onnx.export(net, dummy_input, '%s/%s.onnx' % (args.model_dir, args.model_name), verbose=False)
    print('ONNX models exported.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    parser.add_argument('--model_dir', default='model', type=str)
    parser.add_argument('--model_name', default='resnet-50', type=str)
    args = parser.parse_args()
    print(vars(args))

    main(args)
