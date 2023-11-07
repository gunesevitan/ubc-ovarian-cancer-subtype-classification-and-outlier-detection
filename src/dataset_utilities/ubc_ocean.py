import sys
import os
from tqdm import tqdm
import pandas as pd

os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(pow(2, 40))
import cv2

sys.path.append('..')
import settings
import image_utilities


if __name__ == '__main__':

    raw_image_directory = settings.HDD / 'ubc_ocean' / 'train_images'
    df_train = pd.read_csv(settings.DATA / 'raw_datasets' / 'ubc_ocean' / 'train.csv')

    output_directory = settings.DATA / 'model_datasets' / 'ubc_ocean' / 'images'
    output_directory.mkdir(parents=True, exist_ok=True)

    wsi_longest_edge = 4096

    for idx, row in tqdm(df_train.iterrows(), total=df_train.shape[0]):

        image_type = 'TMA' if row['is_tma'] else 'WSI'

        if row['is_tma'] is False:
            continue

        image_path = str(raw_image_directory / f'{row["image_id"]}.png')
        image = cv2.imread(image_path)
        raw_image_shape = image.shape[:2]

        if image_type == 'TMA':

            # Drop low standard deviation rows and columns (white areas without information)
            image = image_utilities.drop_low_std(image=image, threshold=10)

        elif image_type == 'WSI':

            # Crop the largest contour for WSIs
            largest_contour_bounding_box = image_utilities.get_largest_contour(image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), threshold=20)
            image = image[
                largest_contour_bounding_box[1]:largest_contour_bounding_box[3] + 1,
                largest_contour_bounding_box[0]:largest_contour_bounding_box[2] + 1,
                :
            ]
            image = image_utilities.resize_with_aspect_ratio(image=image, longest_edge=wsi_longest_edge)

        else:
            raise ValueError(f'Invalid image type {image_type}')

        processed_image_shape = image.shape[:2]
        settings.logger.info(f'Image ID {row["image_id"]} - Raw Shape: {raw_image_shape} Processed Shape: {processed_image_shape}')

        cv2.imwrite(str(output_directory / f'{row["image_id"]}.png'), image)
