import torchvision
import torch
import numpy as np
import cv2
import os

def main():
    transform = torchvision.transforms.ToTensor()

    data_dir = 'data/cifar10'
    image_dir = '%s/images' % data_dir
    os.makedirs(image_dir, exist_ok=True)

    test_dataset = torchvision.datasets.CIFAR10(
            root='data',
            train=False,
            transform=transform)

    test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=True)

    with open('%s/val.txt' % data_dir, 'w') as f:
        for index, (image, label) in enumerate(test_loader):

            image = np.array(image * 256, dtype=np.uint8)
            image = image.reshape((3, 32, 32)).transpose(1, 2, 0)

            cv2.imwrite(os.path.join(image_dir, '%d.png' % (index)), image)

            f.write('%d\n' % (label))

if __name__ == '__main__':
    main()
