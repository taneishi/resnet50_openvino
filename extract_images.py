import numpy as np
import cv2
import os

def main():
    data_dir = 'data/MNIST/raw'
    image_dir = 'images'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir, exist_ok=True)

    train_image = 'train-images-idx3-ubyte'
    train_label = 'train-labels-idx1-ubyte'
    test_image = 't10k-images-idx3-ubyte'
    test_label = 't10k-labels-idx1-ubyte'

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

    with open('val.txt', 'w') as f:
        for label in labels:
            f.write('%d\n' % (label))

if __name__ == '__main__':
    main()
