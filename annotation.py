from accuracy_checker.annotation_converters.convert import main
from accuracy_checker.annotation_converters.format_converter import BaseFormatConverter, ConverterReturn
from accuracy_checker.representation import ClassificationAnnotation
from accuracy_checker.config import PathField
import os

class ImagesConverter(BaseFormatConverter):
    __provider__ = 'images'
    annotation_types = (ClassificationAnnotation,)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'data_dir': PathField(is_directory=True, description='Path to sample dataset root directory.')
            })
        return parameters

    def configure(self):
        self.data_dir = self.config['data_dir']

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        # read and convert annotation

        tags = []
        for l in open('imagenet_classes.txt'):
            tags.append(l.strip())
        
        annotations= []
        labels = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith('JPEG'):
                name, label = filename.split('.')[0].split('_', maxsplit=1)
                label = label.replace('_', ' ')
                labels.append(label)
                label = tags.index(label)
                annotations.append(ClassificationAnnotation(filename, label))

        meta = {}
        meta['labels'] = labels
        meta['label_map'] = dict(enumerate(labels))

        return ConverterReturn(annotations, meta, None)

if __name__ == '__main__':
    main()
