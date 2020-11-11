import numpy as np
import random

from meta_domain_transfer.utils import project_root
from meta_domain_transfer.utils.serialization import json_load
from meta_domain_transfer.dataset.base_dataset import BaseDataset

DEFAULT_INFO_PATH = project_root / 'advent/dataset/cityscapes_list/info.json'

class RandomHorizontalFlip(object):
    def __call__(self, image, label):
        if random.random() < 0.5:
            image = np.flip(image,axis=-1)
            label = np.flip(label,axis=-1)
        return image.copy(), label.copy()

class CityscapesDataSet(BaseDataset):
    def __init__(self, root, list_path, set='val',
                 max_iters=None,
                 crop_size=(321, 321), mean=(128, 128, 128),
                 load_labels=True,
                 info_path=DEFAULT_INFO_PATH, labels_size=None,
                 random_horizontal_flip=False):
        super().__init__(root, list_path, set, max_iters, crop_size, labels_size, mean)

        self.load_labels = load_labels
        self.info = json_load(info_path)
        self.class_names = np.array(self.info['label'], dtype=np.str)
        self.mapping = np.array(self.info['label2train'], dtype=np.int)
        self.map_vector = np.zeros((self.mapping.shape[0],), dtype=np.int64)
        for source_label, target_label in self.mapping:
            self.map_vector[source_label] = target_label
        self.random_horizontal_flip = random_horizontal_flip

    def get_metadata(self, name):
        img_file = self.root / 'leftImg8bit' / self.set / name
        label_name = name.replace("leftImg8bit", "gtFine_labelIds")
        label_file = self.root / 'gtFine' / self.set / label_name
        return img_file, label_file

    def map_labels(self, input_):
        return self.map_vector[input_.astype(np.int64, copy=False)]

    def __getitem__(self, index):
        img_file, label_file, name = self.files[index]
        label = self.get_labels(label_file)
        label = self.map_labels(label).copy()
        image = self.get_image(img_file)
        image = self.preprocess(image)
        if self.random_horizontal_flip is True:
            image, label = RandomHorizontalFlip()(image, label)
        return image.copy(), label, np.array(image.shape), name
