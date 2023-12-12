import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A


class ImageClassificationDataset(Dataset):

    def __init__(self, image_ids, image_paths, image_types, targets=None, transforms=None, zoom_normalization_probability=0):

        self.image_ids = image_ids
        self.image_paths = image_paths
        self.image_types = image_types
        self.targets = targets
        self.transforms = transforms
        self.random_crop = A.RandomCrop(height=512, width=512, p=zoom_normalization_probability)

    def __len__(self):

        """
        Get the length of the dataset

        Returns
        -------
        length: int
            Length of the dataset
        """

        return len(self.image_paths)

    def __getitem__(self, idx):

        """
        Get the idxth element in the dataset

        Parameters
        ----------
        idx: int
            Index of the sample (0 <= idx < length of the dataset)

        Returns
        -------
        image_id: int
            Image identifier

        image_type: str
            Image type

        image: torch.FloatTensor of shape (channel, height, width)
            Image tensor

        target: torch.Tensor of shape (1)
            Target tensor
        """

        image_id = self.image_ids[idx]
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_type = self.image_types[idx]

        if image_type == 'wsi':
            image = self.random_crop(image=image)['image']

        if self.targets is not None:

            target = self.targets[idx]
            target = torch.as_tensor(target, dtype=torch.long)

            if self.transforms is not None:
                image = self.transforms(image=image)['image'].float()
            else:
                image = torch.as_tensor(image, dtype=torch.float)

            return image_id, image_type, image, target

        else:

            if self.transforms is not None:
                image = self.transforms(image=image)['image'].float()
            else:
                image = torch.as_tensor(image, dtype=torch.float)

            return image_id, image_type, image


def prepare_classification_data(df):

    """
    Prepare data for classification dataset

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with image_id, image_type, image_path and target columns

    Returns
    -------
    image_ids: numpy.ndarray of shape (n_samples)
        Array of image identifiers

    image_types: numpy.ndarray of shape (n_samples)
        Array of image types

    image_paths: numpy.ndarray of shape (n_samples)
        Array of image paths

    targets: dict of numpy.ndarray of shape (n_samples)
        Array of targets
    """

    image_ids = df['image_id'].values
    image_paths = df['image_path'].values
    image_types = df['image_type'].values
    df['target'] = df['label'].map({
        'HGSC': 0,
        'EC': 1,
        'CC': 2,
        'LGSC': 3,
        'MC': 4,
        'Other': 5
    }).astype(np.uint8)
    targets = df['target'].values

    return image_ids, image_types, image_paths, targets


def collate_fn(batch):

    """
    Collate function for crop dataset

    Parameters
    ----------
    batch: list (batch size) of tuples (images and masks) of tensors (n_crops, 1 or 3, height, width)
        Batch that the data loader generates on each iteration

    Returns
    -------
    images: torch.FloatTensor of shape (batch_size, channel, height, width)
        Images tensor

    targets: torch.FloatTensor of shape (batch_size, 1)
        Targets tensor
    """

    _, _, images, targets = zip(*batch)
    images = torch.cat(images, dim=0)
    targets = torch.cat(targets, dim=0)

    return images, targets
