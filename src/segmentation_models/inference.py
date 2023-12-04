import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.append('..')
import settings
from models import SegModel


def load_segmentation_model(model_directory, model_file_name, device):

    """
    Load model and pretrained weights from the given model directory

    Parameters
    ----------
    model_directory: pathlib.Path
        Path of the model directory

    model_file_name: str
        Name of the model weights file

    device: torch.device
        Location of the model

    Returns
    -------
    model: torch.nn.Module
        Model with weights loaded
    """

    model = SegModel(num_classes=1)
    model.load_state_dict(torch.load(model_directory / model_file_name), strict=True)
    model.to(device)
    model.eval()
    settings.logger.info(f'{model.__class__.__name__} model\'s weights are loaded from {str(model_directory / model_file_name)}')

    return model


if __name__ == '__main__':

    segmentation_size = 384
    segmentation_device = torch.device('cuda')
    segmentation_amp = True

    segmentation_transforms = A.Compose([
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ])

    image_directory = settings.HDD / 'ubc_ocean' / 'train_thumbnails'

    output_mask_directory = settings.DATA / 'model_datasets' / 'ubc_ocean' / 'masks'
    output_mask_directory.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(settings.DATA / 'model_datasets' / 'ubc_ocean' / 'metadata.csv')
    segmentation_model = load_segmentation_model(
        model_directory=settings.MODELS / 'segmentation',
        model_file_name='maxvit_tiny_512_v1_final_epoch_13.pt',
        device=segmentation_device
    )

    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):

        if row['image_type'] == 'tma':
            continue

        image_thumbnail = cv2.imread(str(image_directory / f'{row["image_id"]}_thumbnail.png'))
        image_thumbnail = cv2.cvtColor(image_thumbnail, cv2.COLOR_BGR2RGB)

        image_thumbnail_height, image_thumbnail_width = image_thumbnail.shape[:2]
        thumbnail_height_padding = segmentation_size - image_thumbnail_height % segmentation_size
        thumbnail_width_padding = segmentation_size - image_thumbnail_width % segmentation_size
        image_thumbnail_padded = np.pad(image_thumbnail, ((0 + 64, thumbnail_height_padding + 64), (0 + 64, thumbnail_width_padding + 64), (0, 0)))
        mask_padded = np.zeros((image_thumbnail_height + thumbnail_height_padding, image_thumbnail_width + thumbnail_width_padding), dtype=np.float32)

        for height in range((image_thumbnail_height + thumbnail_height_padding) // segmentation_size):
            height_1, height_2 = height * segmentation_size, (height + 1) * segmentation_size
            for width in range((image_thumbnail_width + thumbnail_width_padding) // segmentation_size):
                width_1, width_2 = width * segmentation_size, (width + 1) * segmentation_size

                image_thumbnail_tile = image_thumbnail_padded[height_1:height_2 + 128, width_1:width_2 + 128]
                inputs = segmentation_transforms(image=image_thumbnail_tile)['image'].to(segmentation_device)
                inputs = torch.stack((
                    inputs,
                    torch.flip(inputs, dims=(1,)),
                    torch.flip(inputs, dims=(2,)),
                    torch.flip(inputs, dims=(1, 2))
                ), dim=0)

                with torch.no_grad():
                    if segmentation_amp:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            outputs = segmentation_model(inputs)
                    else:
                        outputs = segmentation_model(inputs)

                outputs = outputs.cpu()
                outputs = torch.stack((
                    outputs[0],
                    torch.flip(outputs[1], dims=(1,)),
                    torch.flip(outputs[2], dims=(2,)),
                    torch.flip(outputs[3], dims=(1, 2)),
                ), dim=0)
                outputs = torch.mean(outputs, dim=0).squeeze().cpu().numpy()
                mask_padded[height_1:height_2, width_1:width_2] = outputs[64:448, 64:448]

        mask = mask_padded[:image_thumbnail_height, :image_thumbnail_width]
        mask = np.uint8(mask * 255)
        cv2.imwrite(output_mask_directory / f'{row["image_id"]}.png', mask)

    image_thumbnail = cv2.imread(str(settings.HDD / 'ubc_ocean' / 'test_thumbnails' / '41_thumbnail.png'))
    image_thumbnail = cv2.cvtColor(image_thumbnail, cv2.COLOR_BGR2RGB)

    image_thumbnail_height, image_thumbnail_width = image_thumbnail.shape[:2]
    thumbnail_height_padding = segmentation_size - image_thumbnail_height % segmentation_size
    thumbnail_width_padding = segmentation_size - image_thumbnail_width % segmentation_size
    image_thumbnail_padded = np.pad(image_thumbnail, ((0 + 64, thumbnail_height_padding + 64), (0 + 64, thumbnail_width_padding + 64), (0, 0)))
    mask_padded = np.zeros((image_thumbnail_height + thumbnail_height_padding, image_thumbnail_width + thumbnail_width_padding), dtype=np.float32)

    for height in range((image_thumbnail_height + thumbnail_height_padding) // segmentation_size):
        height_1, height_2 = height * segmentation_size, (height + 1) * segmentation_size
        for width in range((image_thumbnail_width + thumbnail_width_padding) // segmentation_size):
            width_1, width_2 = width * segmentation_size, (width + 1) * segmentation_size

            image_thumbnail_tile = image_thumbnail_padded[height_1:height_2 + 128, width_1:width_2 + 128]
            inputs = segmentation_transforms(image=image_thumbnail_tile)['image'].to(segmentation_device)
            inputs = torch.stack((
                inputs,
                torch.flip(inputs, dims=(1,)),
                torch.flip(inputs, dims=(2,)),
                torch.flip(inputs, dims=(1, 2))
            ), dim=0)

            with torch.no_grad():
                if segmentation_amp:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        outputs = segmentation_model(inputs)
                else:
                    outputs = segmentation_model(inputs)

            outputs = outputs.cpu()
            outputs = torch.stack((
                outputs[0],
                torch.flip(outputs[1], dims=(1,)),
                torch.flip(outputs[2], dims=(2,)),
                torch.flip(outputs[3], dims=(1, 2)),
            ), dim=0)
            outputs = torch.mean(outputs, dim=0).squeeze().cpu().numpy()
            mask_padded[height_1:height_2, width_1:width_2] = outputs[64:448, 64:448]

    mask = mask_padded[:image_thumbnail_height, :image_thumbnail_width]
    mask = np.uint8(mask * 255)
    cv2.imwrite(str(output_mask_directory / '41.png'), mask)
