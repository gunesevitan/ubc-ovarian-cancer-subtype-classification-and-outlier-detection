import sys
import os
from shutil import copyfile
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.append('..')
import settings
import annotation_utilities


if __name__ == '__main__':

    annotation_directory = settings.DATA / 'raw_datasets' / 'tma_object_detection'

    yolo_dataset_directory = settings.DATA / 'model_datasets' / 'yolo_roi_detection'
    yolo_dataset_directory.mkdir(parents=True, exist_ok=True)

    df_annotations = []

    for annotation_file_name in tqdm(os.listdir(annotation_directory)):

        dataset_name = annotation_file_name.split('_annotations')[0]
        df_annotation = pd.read_csv(annotation_directory / annotation_file_name)

        df_annotation['dataset'] = dataset_name
        if dataset_name == 'ubc_ocean':
            df_annotation['image_path'] = df_annotation['image_name'].apply(lambda image_file_name: str(settings.DATA / 'raw_datasets' / 'ubc_ocean' / 'train_images' / image_file_name))
        elif dataset_name == 'stanford_tissue_microarray_database_ovarian_cancer_tma':
            df_annotation['image_path'] = df_annotation['image_name'].apply(lambda image_file_name: str(settings.DATA / 'raw_datasets' / 'stanford_tissue_microarray_database_ovarian_cancer_tma' / 'images' / image_file_name))

        df_annotations.append(df_annotation)

    df_annotations = pd.concat(df_annotations, axis=0, ignore_index=True)

    category_ids = {'object': 0}
    categories = [{'name': name, 'id': category_id} for name, category_id in category_ids.items()]

    fold = 2

    if fold is not None:

        skf = StratifiedKFold(n_splits=2, random_state=42, shuffle=True)

        for fold_idx, (training_idx, validation_idx) in enumerate(skf.split(df_annotations, df_annotations['dataset']), start=1):

            if fold_idx != fold:
                continue

            training_directory = yolo_dataset_directory / 'train'
            training_directory.mkdir(parents=True, exist_ok=True)

            for _, row in tqdm(df_annotations.loc[training_idx].iterrows(), total=df_annotations.loc[training_idx].shape[0]):

                copyfile(row['image_path'], training_directory / row['image_name'])

                with open(training_directory / f'{row["image_name"].split(".")[0]}.txt', 'w') as text_file:
                    label = category_ids[row['label_name']]
                    bounding_box = annotation_utilities.coco_to_yolo_bounding_box([row['bbox_x'], row['bbox_y'], row['bbox_width'], row['bbox_height']])
                    bounding_box = np.array(bounding_box, dtype=np.float32)
                    bounding_box[0] /= row['image_width']
                    bounding_box[1] /= row['image_height']
                    bounding_box[2] /= row['image_width']
                    bounding_box[3] /= row['image_height']
                    text_file.write(f'{label} {" ".join(map(str, bounding_box))}\n')

            validation_directory = yolo_dataset_directory / 'val'
            validation_directory.mkdir(parents=True, exist_ok=True)

            for _, row in tqdm(df_annotations.loc[validation_idx].iterrows(), total=df_annotations.loc[validation_idx].shape[0]):
                copyfile(row['image_path'], validation_directory / row['image_name'])

                with open(validation_directory / f'{row["image_name"].split(".")[0]}.txt', 'w') as text_file:
                    label = category_ids[row['label_name']]
                    bounding_box = annotation_utilities.coco_to_yolo_bounding_box([row['bbox_x'], row['bbox_y'], row['bbox_width'], row['bbox_height']])
                    bounding_box = np.array(bounding_box, dtype=np.float32)
                    bounding_box[0] /= row['image_width']
                    bounding_box[1] /= row['image_height']
                    bounding_box[2] /= row['image_width']
                    bounding_box[3] /= row['image_height']
                    text_file.write(f'{label} {" ".join(map(str, bounding_box))}\n')

            settings.logger.info(f'Written YOLO datasets for fold {fold_idx}')
