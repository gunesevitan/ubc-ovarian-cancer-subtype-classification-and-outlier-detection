import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A


class ImageClassificationDataset(Dataset):

    def __init__(self, image_paths, image_types, targets=None, transforms=None, zoom_normalization_probability=0):

        self.image_paths = image_paths
        self.image_types = image_types
        self.targets = targets
        self.transforms = transforms
        self.random_crop = A.RandomCrop(height=512, width=512, p=zoom_normalization_probability)

    def __len__(self):

        """
        Get the length the dataset

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
        image: torch.FloatTensor of shape (channel, height, width)
            Image tensor

        target: torch.Tensor of shape (1)
            Target tensor
        """

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

            return image, target

        else:

            if self.transforms is not None:
                image = self.transforms(image=image)['image'].float()
            else:
                image = torch.as_tensor(image, dtype=torch.float)

            return image


def prepare_classification_data(df, dataset_type):

    """
    Prepare data for classification dataset

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with image_path and target columns

    dataset_type: str
        Type of the dataset

    Returns
    -------
    image_paths: numpy.ndarray of shape (n_samples)
        Array of image paths

    image_types: numpy.ndarray of shape (n_samples)
        Array of image types

    targets: dict of numpy.ndarray of shape (n_samples)
        Array of targets
    """

    if dataset_type == 'image_dataset':
        image_paths = df['image_path'].values
    elif dataset_type == 'instance_dataset':
        image_paths = df['image_path'].apply(lambda image_path: '/'.join(str(image_path).split('/')[:-2]) + f'/instances/{str(image_path).split("/")[-1]}')
        image_paths = image_paths.apply(lambda x: [f'{"/".join(x.split("/")[:-1])}/{str(x).split("/")[-1].split(".")[0]}_{q}.png' for q in range(1, 17)]).values
    else:
        raise ValueError(f'Invalid dataset type {dataset_type}')

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

    return image_paths, image_types, targets


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

    images, targets = zip(*batch)
    images = torch.cat(images, dim=0)
    targets = torch.cat(targets, dim=0)

    return images, targets
