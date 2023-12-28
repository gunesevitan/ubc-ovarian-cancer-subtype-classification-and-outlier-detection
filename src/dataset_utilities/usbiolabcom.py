import sys
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2

sys.path.append('..')
import settings


if __name__ == '__main__':

    dataset_name = 'usbiolabcom'
    raw_image_directory = settings.DATA / 'raw_datasets' / dataset_name / 'images'
    raw_image_paths = glob(str(raw_image_directory / '*'))

    output_dataset_directory = settings.DATA / 'model_datasets' / dataset_name
    output_image_directory = output_dataset_directory / 'images'
    output_image_directory.mkdir(parents=True, exist_ok=True)

    metadata = []

    for image_id, image_path in enumerate(tqdm(raw_image_paths)):

        image = cv2.imread(image_path)

        image_file_name = image_path.split('/')[-1]
        image_metadata = image_file_name.split('_')
        output_image_path = str(output_image_directory / f'{image_id}.png')
        image_width = image.shape[1]
        image_height = image.shape[0]

        metadata.append({
            'image_id': f'{dataset_name}_{image_id}',
            'label': image_metadata[1].split('.')[0],
            'image_width': image_width,
            'image_height': image_height,
            'image_type': 'tma',
            'organ': 'ovary',
            'cancer_subtype': np.nan,
            'dataset': dataset_name,
            'image_path': output_image_path
        })

        cv2.imwrite(output_image_path, image)

    df_metadata = pd.DataFrame(metadata)
    df_metadata.to_csv(output_dataset_directory / 'metadata.csv', index=False)
