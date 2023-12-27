import sys
from glob import glob
from tqdm import tqdm
import pandas as pd
import cv2

sys.path.append('..')
import settings


CLASS_MAPPING = {
    'Serous': 'HGSC',
    'Mucinous': 'MC',
    'Clear_Cell': 'CC',
    'Endometri': 'EC'
}


if __name__ == '__main__':

    dataset_name = 'kztymsrjx9'
    raw_image_directory = settings.DATA / 'raw_datasets' / dataset_name / '*'
    raw_image_paths = glob(str(raw_image_directory / '*'))

    output_dataset_directory = settings.DATA / 'model_datasets' / dataset_name
    output_image_directory = output_dataset_directory / 'images'
    output_image_directory.mkdir(parents=True, exist_ok=True)

    metadata = []

    for image_id, image_path in enumerate(tqdm(raw_image_paths)):

        cancer_subtype = image_path.split('/')[-2]
        if cancer_subtype == 'Non_Cancerous':
            continue

        image = cv2.imread(image_path)
        image_file_name = image_path.split('/')[-1]

        output_image_path = str(output_image_directory / f'{image_id}.jpg')
        image_width = image.shape[1]
        image_height = image.shape[0]

        metadata.append({
            'image_id': f'{dataset_name}_{image_id}',
            'label': CLASS_MAPPING[cancer_subtype],
            'image_width': image_width,
            'image_height': image_height,
            'image_type': 'tma',
            'organ': 'ovary',
            'cancer_subtype': cancer_subtype,
            'dataset': dataset_name,
            'image_path': output_image_path
        })

        cv2.imwrite(output_image_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    df_metadata = pd.DataFrame(metadata)
    df_metadata.to_csv(output_dataset_directory / 'metadata.csv', index=False)
