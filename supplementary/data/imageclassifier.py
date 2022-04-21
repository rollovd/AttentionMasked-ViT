import torch
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Tuple
from abc import ABC, abstractmethod

from supplementary.data.basic import BasicDataset
from supplementary.helpers.helpers import ImagePadding, square_crop

IMAGE_MODE_NUMPY = 'np'
IMAGE_MODE_PIL = 'pil'
DATAFRAME_SOURCE_TYPE_CSV = 'csv'
COLUMNS = ['frame_filename', 'class', 'aspect_ratio']

class ImageClassifierDataset(BasicDataset, ABC):

    filenames_with_labels: List[Tuple[str, str]] = None
    classes = None
    image_mode = None

    @abstractmethod
    def _collect_filenames_with_labels(self):
        pass

    def __init__(self, cfg, augment, **kwargs):
        super(ImageClassifierDataset, self).__init__(cfg=cfg, augment=augment, **kwargs)

        self.classes = cfg.get('classes', None)
        self.image_mode = cfg.get('image_mode', IMAGE_MODE_PIL)
        self._collect_filenames_with_labels()

        if self.mask:
            self._image_padding = ImagePadding(patch_size=16,
                                               padding_type='one_side',
                                               square_image_size=256)

    def __len__(self):
        return len(self.filenames_with_labels)

    def __getitem__(self, index):
        data = self.filenames_with_labels[index]

        image_path = data[0]
        label = data[1]

        if self.mask:
            proportion_class = data[3]
            cropped_image, mask, num_channels = self._image_padding._aspect_ratio_resize(image_path)
            cropped_image = np.asarray(cropped_image)
            if num_channels == 4:
                cropped_image = cropped_image[..., :-1]

            image = self.augment(image=cropped_image)['image']
            image = self._image_padding._pad_image(image) / 255

            return image, mask, label, proportion_class, image_path


        else:
            image = Image.open(image_path)
            image = square_crop(image)
            cropped_image = np.asarray(image.resize((224, 224), Image.NEAREST))
            _, _, num_channels = cropped_image.shape
            if num_channels == 4:
                cropped_image = cropped_image[..., :-1]

            image = self.augment(image=cropped_image)['image']
            image = image.type(torch.float32) / 255.

            return image, label


class DataFrameImageClassifierDataset(ImageClassifierDataset):
    datasource_type: str = None
    datasource: str = None
    separator: str = ','

    def _encode_proportions(self, data):
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        le.fit(data.aspect_ratio + data['class'])

        data['proportion_class'] = le.transform(data['aspect_ratio'] + data['class'])
        return data

    def _collect_filenames_with_labels(self):
        if self.datasource_type == DATAFRAME_SOURCE_TYPE_CSV:
            try:
                data = pd.read_csv(self.datasource, low_memory=False)
                data['frame_filename'] = data['frame_filename'].apply(lambda x: '/mnt/data/opt/labeler/' + x)
                data = data[COLUMNS]
                data = data.sample(frac=1)

                if self.mask:
                    data = self._encode_proportions(data)
                    self._labels = data['class'].values
                    self._proportion_classes = data.proportion_class.values

                    self.filenames_with_labels = list(
                        data[[self.filename_column, self.label_column, self.aspect_ratio_column,
                              self.proportion_class]].itertuples(
                            index=False))
                    self.filenames_with_labels = [tuple(x) for x in self.filenames_with_labels]

                else:
                    self.filenames_with_labels = list(
                        data[[self.filename_column, self.label_column]].itertuples(index=False))
                    self.filenames_with_labels = [tuple(x) for x in self.filenames_with_labels]

            except Exception as e:
                raise RuntimeError(f'failed to read csv data: {e}')
        else:
            raise NotImplementedError('that source is not implemented yet')


