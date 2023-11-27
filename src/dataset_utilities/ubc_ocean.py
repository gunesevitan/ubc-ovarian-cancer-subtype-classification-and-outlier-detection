import sys
import os
from tqdm import tqdm
import numpy as np
import pandas as pd

os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(pow(2, 40))
import cv2

sys.path.append('..')
import settings
import image_utilities


if __name__ == '__main__':

    raw_image_directory = settings.HDD / 'ubc_ocean' / 'train_images'
    raw_mask_directory = settings.DATA / 'raw_datasets' / 'ubc_ocean' / 'masks'
    mask_file_names = sorted(os.listdir(raw_mask_directory), key=lambda x: int(str(x).split('.')[0]))
    df = pd.read_csv(settings.DATA / 'raw_datasets' / 'ubc_ocean' / 'train.csv')

    output_image_directory = settings.DATA / 'model_datasets' / 'ubc_ocean' / 'images'
    output_image_directory.mkdir(parents=True, exist_ok=True)
    output_mask_directory = settings.DATA / 'model_datasets' / 'ubc_ocean' / 'masks'
    output_mask_directory.mkdir(parents=True, exist_ok=True)

    wsi_longest_edge_size = 8192
    metadata = []

    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):

        image_type = 'tma' if row['is_tma'] else 'wsi'
        file_name = f'{row["image_id"]}.png'

        image_path = str(raw_image_directory / file_name)
        image = cv2.imread(image_path)
        raw_image_shape = image.shape[:2]

        if file_name in mask_file_names:
            mask_path = str(raw_mask_directory / file_name)
            mask = cv2.imread(mask_path)
        else:
            mask = None

        if image_type == 'tma':

            # Drop low standard deviation rows and columns (white areas with less tissue)
            image = image_utilities.drop_low_std(image=image, threshold=10)

        elif image_type == 'wsi':

            if mask is not None:
                # Crop image and mask by mask's bounding box
                non_zero_idx = np.where(np.any(mask != 0, axis=-1))
                mask_bounding_box = [
                    int(np.min(non_zero_idx[1])),
                    int(np.min(non_zero_idx[0])),
                    int(np.max(non_zero_idx[1])),
                    int(np.max(non_zero_idx[0]))
                ]

                image = image[
                    mask_bounding_box[1]:mask_bounding_box[3] + 1,
                    mask_bounding_box[0]:mask_bounding_box[2] + 1,
                    :
                ]

                mask = mask[
                    mask_bounding_box[1]:mask_bounding_box[3] + 1,
                    mask_bounding_box[0]:mask_bounding_box[2] + 1,
                    :
                ]

            # Crop the largest contour on WSIs
            largest_contour_bounding_box = image_utilities.get_largest_contour(image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), threshold=20)
            image = image[
                largest_contour_bounding_box[1]:largest_contour_bounding_box[3] + 1,
                largest_contour_bounding_box[0]:largest_contour_bounding_box[2] + 1,
                :
            ]

            # Standardize the background
            image[np.all(image == 0, axis=-1)] = 255
            # Resize the longest edge to given size
            image = image_utilities.resize_with_aspect_ratio(image=image, longest_edge=wsi_longest_edge_size, interpolation=cv2.INTER_AREA)

            if mask is not None:
                mask = mask[
                    largest_contour_bounding_box[1]:largest_contour_bounding_box[3] + 1,
                    largest_contour_bounding_box[0]:largest_contour_bounding_box[2] + 1,
                    :
                ]
                mask = image_utilities.resize_with_aspect_ratio(image=mask, longest_edge=wsi_longest_edge_size, interpolation=cv2.INTER_AREA)
        else:
            raise ValueError(f'Invalid image type {image_type}')

        cv2.imwrite(str(output_image_directory / file_name), image)
        if mask is not None:
            cv2.imwrite(str(output_mask_directory / file_name), mask)

        processed_image_shape = image.shape[:2]
        settings.logger.info(f'Image ID {row["image_id"]} - Raw Shape: {raw_image_shape} Processed Shape: {processed_image_shape}')

        metadata.append({
            'image_id': row['image_id'],
            'label': row['label'],
            'image_width': processed_image_shape[1],
            'image_height': processed_image_shape[0],
            'image_type': image_type,
            'organ': 'ovary',
            'cancer_subtype': np.nan,
            'dataset': 'ubc_ocean',
            'image_path': output_image_directory / file_name,
            'mask_path': str(output_mask_directory / file_name) if mask is not None else np.nan
        })

    df_metadata = pd.DataFrame(metadata)
    df_metadata.to_csv(settings.DATA / 'model_datasets' / 'ubc_ocean' / 'metadata.csv', index=False)
