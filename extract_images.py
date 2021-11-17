import numpy as np
import cv2
import os

def main():
    data_dir = 'data/MNIST'
    image_dir = '%s/images' % data_dir
    if not os.path.exists(image_dir):
        os.makedirs(image_dir, exist_ok=True)

    train_image = 'raw/train-images-idx3-ubyte'
    train_label = 'raw/train-labels-idx1-ubyte'
    test_image = 'raw/t10k-images-idx3-ubyte'
    test_label = 'raw/t10k-labels-idx1-ubyte'

    with open(os.path.join(data_dir, test_image), 'rb') as f:
        images = f.read()

    images = [d for d in images[16:]]
    images = np.array(images, dtype=np.uint8)
    images = images.reshape((-1, 28, 28))

    for index, image in enumerate(images):
        cv2.imwrite(os.path.join(image_dir, '%d.png' % (index)), image)
    
    with open(os.path.join(data_dir, test_label), 'rb') as f:
        labels = f.read()

    labels = labels[8:]

    with open('%s/val.txt' % data_dir, 'w') as f:
        for label in labels:
            f.write('%d\n' % (label))

if __name__ == '__main__':
    main()
