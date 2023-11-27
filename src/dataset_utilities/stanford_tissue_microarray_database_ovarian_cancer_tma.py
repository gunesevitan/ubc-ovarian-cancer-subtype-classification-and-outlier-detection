import sys
from glob import glob
from tqdm import tqdm
import pandas as pd
import cv2

sys.path.append('..')
import settings


CLASS_MAPPING = {
    'fibroma of ovary spindle cell fibroma of ovary': 'Other',
    'carcinoma papillary serous': 'HGSC',
    'carcinoma endometrioid': 'EC',
    'lymphoma precursor B lymphoblastic': 'Other',
    'carcinoma adeno': 'HGSC',
    'carcinoma clear cell': 'CC',
    'carcinoma mucinous': 'MC',
    'carcinoma adeno mucinous': 'MC',
    'seminoma dysgerminoma': 'Other'
}


if __name__ == '__main__':

    dataset_name = 'stanford_tissue_microarray_database_ovarian_cancer_tma'
    raw_image_directory = settings.DATA / 'raw_datasets' / dataset_name / 'images'
    raw_image_paths = glob(str(raw_image_directory / '*'))

    output_dataset_directory = settings.DATA / 'model_datasets' / dataset_name
    output_image_directory = output_dataset_directory / 'images'
    output_image_directory.mkdir(parents=True, exist_ok=True)

    metadata = []

    for image_path in tqdm(raw_image_paths):

        image = cv2.imread(image_path)

        image_file_name = image_path.split('/')[-1]
        image_metadata = image_file_name.split('_')
        image_id = f'{dataset_name}_{image_metadata[0]}_{image_metadata[1]}'
        organ = image_metadata[2]
        cancer_subtype = image_metadata[-1].split('.')[0]
        output_image_path = str(output_image_directory / f'{image_id}.jpg')
        image_width = image.shape[1]
        image_height = image.shape[0]

        metadata.append({
            'image_id': image_id,
            'label': CLASS_MAPPING[cancer_subtype],
            'image_width': image_width,
            'image_height': image_height,
            'image_type': 'tma',
            'organ': organ,
            'cancer_subtype': cancer_subtype,
            'dataset': dataset_name,
            'image_path': output_image_path
        })

        cv2.imwrite(output_image_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    df_metadata = pd.DataFrame(metadata)
    df_metadata.to_csv(output_dataset_directory / 'metadata.csv', index=False)
