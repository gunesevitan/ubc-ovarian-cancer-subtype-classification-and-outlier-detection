import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import settings


def create_folds(df, stratify_columns, n_splits, shuffle=True, random_state=42, verbose=True):

    """
    Create columns of folds on given dataframe

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with given stratify columns

    stratify_columns: list
        Names of columns to be stratified

    n_splits: int
        Number of folds (2 <= n_splits)

    shuffle: bool
        Whether to shuffle before split or not

    random_state: int
        Random seed for reproducible results

    verbose: bool
        Verbosity flag

    Returns
    -------
    df: pandas.DataFrame
        Dataframe with created fold columns
    """

    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    for fold, (training_idx, validation_idx) in enumerate(mskf.split(X=df, y=df[stratify_columns]), 1):
        df.loc[training_idx, f'fold{fold}'] = 0
        df.loc[validation_idx, f'fold{fold}'] = 1
        df[f'fold{fold}'] = df[f'fold{fold}'].astype(np.uint8)

    if verbose:

        settings.logger.info(f'Dataset split into {n_splits} folds')

        for fold in range(1, n_splits + 1):
            df_fold = df[df[f'fold{fold}'] == 1]
            stratify_columns_value_counts = {}
            for column in stratify_columns:
                stratify_column_value_counts = df_fold[column].value_counts().to_dict()
                stratify_columns_value_counts[column] = stratify_column_value_counts
            settings.logger.info(f'Fold {fold} {df_fold.shape} - {stratify_columns_value_counts}')

    return df


if __name__ == '__main__':

    df_train = pd.read_csv(settings.DATA / 'raw_datasets' / 'ubc_ocean' / 'train.csv')
    settings.logger.info(f'Train Dataset Shape: {df_train.shape} - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')

    n_splits = 5
    df_train = create_folds(
        df=df_train,
        stratify_columns=['label', 'is_tma'],
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
        verbose=True
    )

    df_train[['image_id'] + [f'fold{fold}' for fold in range(1, n_splits + 1)]].to_csv(settings.DATA / 'folds.csv', index=False)
    settings.logger.info(f'folds.csv is saved to {settings.DATA}')

    # Read and concatenate metadata from dataset directories
    dataset_names = [
        'ubc_ocean',
        'stanford_tissue_microarray_database_ovarian_cancer_tma',
        'tissuearraycom',
        'kztymsrjx9',
        'usbiolabcom',
        'human_protein_atlas'
    ]
    df = []
    for dataset_name in dataset_names:
        df.append(pd.read_csv(settings.DATA / 'model_datasets' / dataset_name / 'metadata.csv'))

    df = pd.concat(df, axis=0).reset_index(drop=True)
    df = df.groupby('image_id').first().reset_index()
    df['dataset'] = df['dataset'].fillna('ubc_ocean')
    df = create_folds(
        df=df,
        stratify_columns=['label', 'image_type', 'dataset'],
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
        verbose=True
    )
    df[['image_id', 'dataset'] + [f'fold{fold}' for fold in range(1, n_splits + 1)]].to_csv(settings.DATA / 'folds_2.csv', index=False)
