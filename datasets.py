from torch.utils.data import Dataset
from PIL import Image
import os

class ImageNetDataSet(Dataset):
    def __init__(self, data_dir, transform=None):

        image_names = []
        labels = []

        tags = []
        for l in open('imagenet_classes.txt'):
            tags.append(l.strip())

        for filename in os.listdir(data_dir):
            if filename.endswith('JPEG'):
                name, label = filename.split('.')[0].split('_', maxsplit=1)
                image_names.append(os.path.join(data_dir, filename))
                label = tags.index(label.replace('_', ' '))
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_names)
