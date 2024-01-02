import os.path
import sys
import argparse
from glob import glob
from pathlib import Path
from tqdm import tqdm
import yaml
import json
import numpy as np
import pandas as pd
import cv2
import torch

sys.path.append('..')
import settings
import metrics
import image_models
import torch_utilities
import transforms



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model_directory', type=str)
    parser.add_argument('image_directory', type=str)
    args = parser.parse_args()

    config = yaml.load(open(settings.MODELS / args.model_directory / 'config.yaml', 'r'), Loader=yaml.FullLoader)
    settings.logger.info(f'Running inference on {args.image_directory}')

    model_root_directory = Path(settings.MODELS / args.model_directory)
    models, config = image_models.load_classification_model(
        model_directory=model_root_directory,
        model_file_names=[
            'model_fold1_epoch_6.pt',
            'model_fold2_epoch_1.pt',
            'model_fold3_epoch_1.pt',
            'model_fold4_epoch_7.pt',
            'model_fold5_epoch_9.pt',
        ],
        device=torch.device('cuda')
    )
    dataset_transforms = transforms.get_classification_transforms(**config['transforms'])
    image_paths = glob(f'{args.image_directory}/*')

    # Set model, device and seed for reproducible results
    torch_utilities.set_seed(config['training']['random_state'], deterministic_cudnn=config['training']['deterministic_cudnn'])
    device = torch.device(config['training']['device'])
    amp = True
    tta = True

    metadata_path = '/'.join(args.image_directory.split('/')[:-1]) + '/metadata.csv'
    if os.path.exists(metadata_path):
        df = pd.read_csv(metadata_path)

    predictions = []
    image_ids = []

    for image_path in tqdm(image_paths):

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        tma_predictions = torch.zeros(1, 6)

        inputs = dataset_transforms['inference'](image=image)['image']
        inputs = inputs.to(device)

        if tta:
            inputs = torch.stack((
                inputs,
                torch.flip(inputs, dims=(1,)),
                torch.flip(inputs, dims=(2,)),
                torch.flip(inputs, dims=(1, 2))
            ), dim=0)
        else:
            inputs = torch.unsqueeze(inputs, dim=0)

        for model_idx, models in enumerate([models]):
            for model in models.values():
                with torch.no_grad():
                    if amp:
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            outputs = model(inputs)
                    else:
                        outputs = model(inputs)

                outputs = outputs.cpu()
                if tta:
                    outputs = torch.mean(outputs, dim=0)
                else:
                    outputs = torch.squeeze(outputs, dim=0)

                tma_predictions += outputs / len(models)

        tma_predictions = torch.softmax(tma_predictions, dim=-1).numpy()
        predictions.append(tma_predictions)
        image_ids.append(np.array([int(image_path.split('/')[-1].split('.')[0])]))

    predictions = np.argmax(np.concatenate(predictions), axis=-1)
    image_ids = np.concatenate(image_ids)
    df_predictions = pd.DataFrame()
    df_predictions['image_id'] = image_ids
    df_predictions['prediction'] = predictions

    df['image_id'] = df['image_id'].apply(lambda x: int(x.split('_')[-1]))
    df = df.merge(df_predictions, on='image_id', how='left')

    label_mapping = {
        0: 'HGSC',
        1: 'EC',
        2: 'CC',
        3: 'LGSC',
        4: 'MC',
        5: 'Other',
    }
    df['prediction_label'] = df['prediction'].map(label_mapping)

    scores = metrics.multiclass_classification_scores(y_true=df['label'], y_pred=df['prediction_label'])
    settings.logger.info(json.dumps(scores, indent=2))

    for label in ['HGSC', 'EC', 'CC', 'LGSC', 'MC', 'Other']:

        df_label = df.loc[df['label'] == label, :].reset_index(drop=True)
        label_scores = metrics.multiclass_classification_scores(y_true=df_label['label'], y_pred=df_label['prediction_label'])
        settings.logger.info(f'Label: {label} (N={df_label.shape[0]})\n{json.dumps(label_scores, indent=2)}')
