import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class ImageClassificationDataset(Dataset):

    def __init__(self, image_paths, crop_size, n_crop, targets=None, transforms=None):

        self.image_paths = image_paths
        self.crop_size = crop_size
        self.n_crop = n_crop
        self.targets = targets
        self.transforms = transforms

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
        images: torch.FloatTensor of shape (n_crops, channel, height, width)
            Images tensor

        targets: torch.Tensor of shape (n_crops, 1)
            Targets tensor
        """

        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images = []

        image_height, image_width = image.shape[:2]
        start_xs = np.random.randint(0, image_width - self.crop_size, self.n_crop)
        start_ys = np.random.randint(0, image_height - self.crop_size, self.n_crop)

        if self.targets is not None:

            target = self.targets[idx]
            target = torch.as_tensor(target, dtype=torch.long)
            targets = []

            for (start_x, start_y) in zip(start_xs, start_ys):
                image_crop = image[start_y:start_y + self.crop_size, start_x:start_x + self.crop_size]
                if self.transforms is not None:
                    transformed = self.transforms(image=image_crop)
                    images.append(transformed['image'])
                else:
                    images.append(torch.as_tensor(image_crop, dtype=torch.float))

                targets.append(target)

            images = torch.stack(images, dim=0)
            targets = torch.stack(targets, dim=0)

            return images, targets

        else:

            for (start_x, start_y) in zip(start_xs, start_ys):
                image_crop = image[start_y:start_y + self.crop_size, start_x:start_x + self.crop_size]
                if self.transforms is not None:
                    transformed = self.transforms(image=image_crop)
                    images.append(transformed['image'])
                else:
                    images = torch.as_tensor(images, dtype=torch.float)

            images = torch.stack(images, dim=0)

            return images


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

    df['target'] = df['label'].map({
        'HGSC': 0,
        'EC': 1,
        'CC': 2,
        'LGSC': 3,
        'MC': 4,
        'Other': 5
    }).astype(np.uint8)
    targets = df['target'].values

    return image_paths, targets


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

    idx = torch.randperm(images.shape[0])
    images = images[idx]
    targets = targets[idx]

    return images, targets
