import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class ImageClassificationDataset(Dataset):

    def __init__(self, image_paths, targets=None, transforms=None):

        self.image_paths = image_paths
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
        image: torch.FloatTensor of shape (channel, height, width)
            Image tensor

        target: torch.Tensor
            Target tensor
        """

        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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


class InstanceClassificationDataset(Dataset):

    def __init__(self, image_paths, n_instances, targets=None, transforms=None):

        self.image_paths = image_paths
        self.n_instances = n_instances
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
        images: torch.FloatTensor of shape (channel, n_instance, height, width)
            Image tensor

        targets: torch.Tensor
            Target tensor
        """

        images = []
        for image_path in self.image_paths[idx][:self.n_instances]:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)

        if self.targets is not None:

            target = self.targets[idx]
            target = torch.as_tensor(target, dtype=torch.long)

            if self.transforms is not None:
                augmentation_inputs = {f'image_{image_idx}': image for image_idx, image in enumerate(images[1:], start=2)}
                augmentation_inputs.update({'image': images[0]})
                transformed = self.transforms(**augmentation_inputs)
                images = torch.stack(list(transformed.values()), dim=1)
            else:
                images = [torch.as_tensor(image, dtype=torch.float) for image in images]
                images = torch.stack(images, dim=1)
                images = torch.permute(images, dims=(0, 3, 1, 2))

            return images, target

        else:

            if self.transforms is not None:
                augmentation_inputs = {f'image_{image_idx}': image for image_idx, image in enumerate(images[1:], start=2)}
                augmentation_inputs.update({'image': images[0]})
                transformed = self.transforms(**augmentation_inputs)
                images = torch.stack(list(transformed.values()), dim=1)
            else:
                images = [torch.as_tensor(image, dtype=torch.float) for image in images]
                images = torch.stack(images, dim=0)
                images = torch.permute(images, dims=(0, 3, 1, 2))

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
