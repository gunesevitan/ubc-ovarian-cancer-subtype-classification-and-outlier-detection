import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_classification_transforms(**transform_parameters):

    """
    Get transforms for classification dataset

    Parameters
    ----------
    transform_parameters: dict
        Dictionary of transform parameters

    Returns
    -------
    transforms: dict
        Transforms for training and inference
    """

    training_transforms = A.Compose([
        A.Resize(
            height=transform_parameters['resize_height'],
            width=transform_parameters['resize_width'],
            interpolation=cv2.INTER_NEAREST,
            always_apply=True
        ),
        A.HorizontalFlip(p=transform_parameters['horizontal_flip_probability']),
        A.VerticalFlip(p=transform_parameters['vertical_flip_probability']),
        A.RandomRotate90(p=transform_parameters['random_rotate_90_probability']),
        A.ShiftScaleRotate(
            shift_limit=transform_parameters['shift_limit'],
            scale_limit=transform_parameters['scale_limit'],
            rotate_limit=transform_parameters['rotate_limit'],
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            value=255,
            p=transform_parameters['shift_scale_rotate_probability']
        ),
        A.ColorJitter(
            brightness=transform_parameters['brightness'],
            contrast=transform_parameters['contrast'],
            saturation=transform_parameters['saturation'],
            hue=transform_parameters['hue'],
            p=transform_parameters['color_jitter_probability']
        ),
        A.GaussianBlur(
            blur_limit=transform_parameters['blur_limit'],
            sigma_limit=transform_parameters['sigma_limit'],
            p=transform_parameters['gaussian_blur_probability']
        ),
        A.CoarseDropout(
            max_holes=transform_parameters['max_holes'],
            max_height=transform_parameters['max_height'],
            max_width=transform_parameters['max_width'],
            min_holes=transform_parameters['min_holes'],
            min_height=transform_parameters['min_height'],
            min_width=transform_parameters['min_width'],
            fill_value=0,
            p=transform_parameters['coarse_dropout_probability']
        ),
        A.Normalize(
            mean=transform_parameters['normalize_mean'],
            std=transform_parameters['normalize_std'],
            max_pixel_value=transform_parameters['normalize_max_pixel_value'],
            always_apply=True
        ),
        ToTensorV2(always_apply=True)
    ])

    inference_transforms = A.Compose([
        A.Resize(
            height=transform_parameters['resize_height'],
            width=transform_parameters['resize_width'],
            interpolation=cv2.INTER_NEAREST,
            always_apply=True
        ),
        A.Normalize(
            mean=transform_parameters['normalize_mean'],
            std=transform_parameters['normalize_std'],
            max_pixel_value=transform_parameters['normalize_max_pixel_value'],
            always_apply=True
        ),
        ToTensorV2(always_apply=True)
    ])

    classification_transforms = {'training': training_transforms, 'inference': inference_transforms}
    return classification_transforms
