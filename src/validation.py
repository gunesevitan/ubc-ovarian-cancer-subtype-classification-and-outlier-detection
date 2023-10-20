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

    df_train = pd.read_csv(settings.DATA / 'ubc_ocean' / 'train.csv')
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
