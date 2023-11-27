import sys
import argparse
import yaml
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.optim as optim

sys.path.append('..')
import settings
import metrics
import visualization
import torch_datasets
import torch_modules
import torch_utilities
import transforms


def train(training_loader, model, criterion, optimizer, device, scheduler=None, amp=False):

    """
    Train given model on given data loader

    Parameters
    ----------
    training_loader: torch.utils.data.DataLoader
        Training set data loader

    model: torch.nn.Module
        Model to train

    criterion: torch.nn.Module
        Loss function

    optimizer: torch.optim.Optimizer
        Optimizer for updating model weights

    device: torch.device
        Location of the model and inputs

    scheduler: torch.optim.LRScheduler or None
        Learning rate scheduler

    amp: bool
        Whether to use auto mixed precision or not

    Returns
    -------
    training_losses: dict
        Dictionary of training losses after model is fully trained on training set data loader
    """

    model.train()
    progress_bar = tqdm(training_loader)

    running_loss = 0.0

    if amp:
        grad_scaler = torch.cuda.amp.GradScaler()
    else:
        grad_scaler = None

    for step, (inputs, targets) in enumerate(progress_bar):

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        if amp:
            with torch.cuda.amp.autocast():
                outputs = model(inputs.half())
        else:
            outputs = model(inputs)

        loss = criterion(outputs, targets)

        if amp:
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.detach().item() * len(inputs)
        lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
        progress_bar.set_description(f'lr: {lr:.8f} - training loss: {running_loss / len(training_loader.sampler):.4f}')

    training_loss = running_loss / len(training_loader.sampler)

    training_losses = {
        'loss': training_loss
    }

    return training_losses


def validate(validation_loader, model, criterion, device, amp=False):

    """
    Validate given model on given data loader

    Parameters
    ----------
    validation_loader: torch.utils.data.DataLoader
        Validation set data loader

    model: torch.nn.Module
        Model to validate

    criterion: torch.nn.Module
        Loss function

    device: torch.device
        Location of the model and inputs

    amp: bool
        Whether to use auto mixed precision or not

    Returns
    -------
    validation_losses: dict
        Dictionary of validation losses after model is fully validated on validation set data loader
    """

    model.eval()
    progress_bar = tqdm(validation_loader)

    running_loss = 0.0

    for step, (inputs, targets) in enumerate(progress_bar):

        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            if amp:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs.half())
            else:
                outputs = model(inputs)

        loss = criterion(outputs, targets)
        running_loss += loss.detach().item() * len(inputs)
        progress_bar.set_description(f'validation loss: {running_loss / len(validation_loader.sampler):.4f}')

    validation_loss = running_loss / len(validation_loader.sampler)

    validation_losses = {
        'loss': validation_loss
    }

    return validation_losses


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model_directory', type=str)
    parser.add_argument('mode', type=str)
    args = parser.parse_args()

    config = yaml.load(open(settings.MODELS / args.model_directory / 'config.yaml', 'r'), Loader=yaml.FullLoader)
    settings.logger.info(f'Running {config["persistence"]["model_directory"]} model in {args.mode} mode')

    # Create directory for models and results
    model_root_directory = Path(settings.MODELS / args.model_directory)
    model_root_directory.mkdir(parents=True, exist_ok=True)

    dataset_type = config['dataset']['dataset_type']
    crop_size = config['dataset']['crop_size']
    n_crop = config['dataset']['n_crop']
    dataset_names = config['dataset']['dataset_names']
    df = []
    for dataset_name in dataset_names:
        df.append(pd.read_csv(settings.DATA / 'model_datasets' / dataset_name / 'metadata.csv'))
    df = pd.concat(df, axis=0).reset_index(drop=True)

    df_folds = pd.read_csv(settings.DATA / 'folds.csv')
    df = df.merge(df_folds, on='image_id', how='left').fillna(0)
    del df_folds

    if args.mode == 'training':

        dataset_transforms = transforms.get_classification_transforms(**config['transforms'])
        training_metadata = defaultdict(dict)

        for fold in config['training']['folds']:

            training_idx, validation_idx = df.loc[df[fold] != 1].index, df.loc[df[fold] == 1].index
            # Validate on training set if validation is set is not specified
            if len(validation_idx) == 0:
                validation_idx = training_idx

            # Create training and validation inputs and targets
            training_image_paths, training_targets = torch_datasets.prepare_classification_data(df=df.loc[training_idx], dataset_type=dataset_type)
            validation_image_paths, validation_targets = torch_datasets.prepare_classification_data(df=df.loc[validation_idx], dataset_type=dataset_type)

            settings.logger.info(
                f'''
                Fold: {fold}
                Training: {len(training_image_paths)} ({len(training_image_paths) // config["training"]["training_batch_size"] + 1} steps)
                Validation {len(validation_image_paths)} ({len(validation_image_paths) // config["training"]["test_batch_size"] + 1} steps)
                '''
            )

            # Create training and validation datasets and dataloaders
            training_dataset = torch_datasets.ImageClassificationDataset(
                image_paths=training_image_paths,
                crop_size=crop_size,
                n_crop=n_crop,
                targets=training_targets,
                transforms=dataset_transforms['training']
            )
            training_loader = DataLoader(
                training_dataset,
                batch_size=config['training']['training_batch_size'],
                sampler=RandomSampler(training_dataset, replacement=False),
                pin_memory=False,
                drop_last=False,
                num_workers=config['training']['num_workers'],
                collate_fn=torch_datasets.collate_fn
            )
            validation_dataset = torch_datasets.ImageClassificationDataset(
                image_paths=validation_image_paths,
                crop_size=crop_size,
                n_crop=n_crop,
                targets=validation_targets,
                transforms=dataset_transforms['inference']
            )
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=config['training']['test_batch_size'],
                sampler=SequentialSampler(validation_dataset),
                pin_memory=False,
                drop_last=False,
                num_workers=config['training']['num_workers'],
                collate_fn=torch_datasets.collate_fn
            )

            # Set model, device and seed for reproducible results
            torch_utilities.set_seed(config['training']['random_state'], deterministic_cudnn=config['training']['deterministic_cudnn'])
            device = torch.device(config['training']['device'])

            criterion = getattr(torch_modules, config['training']['loss_function'])(**config['training']['loss_function_args'])
            model = getattr(torch_modules, config['model']['model_class'])(**config['model']['model_args'])
            if config['model']['model_checkpoint_path'] is not None:
                model.load_state_dict(torch.load(config['model']['model_checkpoint_path']), strict=False)
            model.to(device)

            # Set optimizer, learning rate scheduler and stochastic weight averaging
            optimizer = getattr(torch.optim, config['training']['optimizer'])(model.parameters(), **config['training']['optimizer_args'])
            scheduler = getattr(optim.lr_scheduler, config['training']['lr_scheduler'])(optimizer, **config['training']['lr_scheduler_args'])
            amp = config['training']['amp']

            best_epoch = 1
            early_stopping = False
            training_history = {
                'training_loss': [],
                'validation_loss': []
            }

            for epoch in range(1, config['training']['epochs'] + 1):

                if early_stopping:
                    break

                training_losses = train(
                    training_loader=training_loader,
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                    scheduler=scheduler,
                    amp=amp
                )

                validation_losses = validate(
                    validation_loader=validation_loader,
                    model=model,
                    criterion=criterion,
                    device=device,
                    amp=amp
                )

                settings.logger.info(
                    f'''
                    Epoch {epoch}
                    Training Loss: {json.dumps(training_losses, indent=2)}
                    Validation Loss: {json.dumps(validation_losses, indent=2)}
                    '''
                )

                if epoch in config['persistence']['save_epoch_model']:
                    # Save model if current epoch is specified to be saved
                    model_name = f'model_{fold}_epoch{epoch}.pt'
                    torch.save(model.state_dict(), model_root_directory / model_name)
                    settings.logger.info(f'Saved {model_name} to {model_root_directory}')

                best_validation_loss = np.min(training_history['validation_loss']) if len(training_history['validation_loss']) > 0 else np.inf
                last_validation_loss = validation_losses['loss']
                if last_validation_loss < best_validation_loss:
                    # Save model if validation loss improves
                    model_name = f'model_{fold}_best.pt'
                    torch.save(model.state_dict(), model_root_directory / model_name)
                    settings.logger.info(f'Saved {model_name} to {model_root_directory} (validation loss decreased from {best_validation_loss:.6f} to {last_validation_loss:.6f})\n')

                training_history['training_loss'].append(training_losses['loss'])
                training_history['validation_loss'].append(validation_losses['loss'])

                best_epoch = np.argmin(training_history['validation_loss'])
                if config['training']['early_stopping_patience'] > 0:
                    # Trigger early stopping if early stopping patience is greater than 0
                    if len(training_history['validation_loss']) - best_epoch > config['training']['early_stopping_patience']:
                        settings.logger.info(
                            f'''
                            Early Stopping (validation loss didn\'t improve for {config['training']["early_stopping_patience"]} epochs)
                            Best Epoch ({best_epoch + 1}) Validation Loss: {training_history["validation_loss"][best_epoch]:.4f}
                            '''
                        )
                        early_stopping = True

            training_metadata[fold] = {
                'best_epoch': int(best_epoch),
                'training_loss': float(training_history['training_loss'][best_epoch]),
                'validation_loss': float(training_history['validation_loss'][best_epoch]),
                'training_history': training_history
            }

            visualization.visualize_learning_curve(
                training_losses=training_metadata[fold]['training_history']['training_loss'],
                validation_losses=training_metadata[fold]['training_history']['validation_loss'],
                best_epoch=training_metadata[fold]['best_epoch'],
                path=model_root_directory / f'learning_curve_{fold}.png'
            )

            with open(model_root_directory / 'training_metadata.json', mode='w') as f:
                json.dump(training_metadata, f, indent=2, ensure_ascii=False)

    elif args.mode == 'test':

        dataset_transforms = transforms.get_classification_transforms(**config['transforms'])
        df_scores = []

        folds = config['test']['folds']
        model_file_names = config['test']['model_file_names']

        for fold, model_file_name in zip(folds, model_file_names):

            # Create validation inputs and targets
            validation_idx = df.loc[df[fold] == 1].index
            validation_image_paths, validation_targets = torch_datasets.prepare_classification_data(df=df.loc[validation_idx], dataset_type=dataset_type)

            settings.logger.info(
                f'''
                Fold: {fold} ({model_file_name})
                Validation {len(validation_image_paths)} ({len(validation_image_paths) // config["training"]["test_batch_size"] + 1} steps)
                '''
            )

            # Create validation datasets and dataloaders
            validation_dataset = torch_datasets.ImageClassificationDataset(
                image_paths=validation_image_paths,
                crop_size=crop_size,
                n_crop=n_crop,
                targets=validation_targets,
                transforms=dataset_transforms['inference']
            )
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=config['training']['test_batch_size'],
                sampler=SequentialSampler(validation_dataset),
                pin_memory=False,
                drop_last=False,
                num_workers=config['training']['num_workers'],
                collate_fn=torch_datasets.collate_fn
            )

            # Set model, device and seed for reproducible results
            torch_utilities.set_seed(config['training']['random_state'], deterministic_cudnn=config['training']['deterministic_cudnn'])
            device = torch.device(config['training']['device'])
            amp = config['training']['amp']

            model = getattr(torch_modules, config['model']['model_class'])(**config['model']['model_args'])
            model.load_state_dict(torch.load(model_root_directory / model_file_name))
            model.to(device)
            model.eval()

            validation_predictions = []

            for inputs, _ in tqdm(validation_loader):

                inputs = inputs.to(device)

                with torch.no_grad():
                    outputs = model(inputs)

                outputs = outputs.cpu()

                if config['test']['tta']:

                    inputs = inputs.to('cpu')
                    tta_flip_dimensions = config['test']['tta_flip_dimensions']

                    tta_outputs = []

                    for dimensions in tta_flip_dimensions:

                        augmented_inputs = torch.flip(inputs, dims=dimensions).to(device)

                        with torch.no_grad():
                            augmented_outputs = model(augmented_inputs)

                        tta_outputs.append(augmented_outputs.cpu())

                    outputs = torch.stack(([outputs] + tta_outputs), dim=-1)
                    outputs = torch.mean(outputs, dim=-1)

                validation_predictions += [outputs]

            validation_predictions = torch.cat(validation_predictions, dim=0)
            validation_predictions = validation_predictions.view(validation_predictions.shape[0] // n_crop, n_crop, -1).mean(dim=1)
            validation_predictions = np.argmax(torch.softmax(validation_predictions, dim=-1).numpy(), axis=1)

            df.loc[validation_idx, 'prediction'] = validation_predictions
            df.loc[validation_idx, 'target'] = validation_targets
            validation_scores = metrics.multiclass_classification_scores(y_true=df.loc[validation_idx, 'target'], y_pred=df.loc[validation_idx, 'prediction'])
            df_scores.append(validation_scores)
            settings.logger.info(f'{fold} Validation Scores: {json.dumps(validation_scores, indent=2)}')

        df['prediction'] = df['prediction'].astype(int)
        df['target'] = df['target'].astype(int)

        df_scores = pd.DataFrame(df_scores)
        settings.logger.info(
            f'''
            Mean Validation Scores
            {json.dumps(df_scores.mean(axis=0).to_dict(), indent=2)}
            and Standard Deviations
            ±{json.dumps(df_scores.std(axis=0).to_dict(), indent=2)}
            '''
        )

        tma_mask = df['image_type'] == 'tma'
        tma_oof_scores = metrics.multiclass_classification_scores(y_true=df.loc[tma_mask, 'target'], y_pred=df.loc[tma_mask, 'prediction'])
        settings.logger.info(f'TMA OOF Scores: {json.dumps(tma_oof_scores, indent=2)}')

        with open(model_root_directory / 'tma_oof_scores.json', mode='w') as f:
            json.dump(tma_oof_scores, f, indent=2, ensure_ascii=False)

        wsi_mask = df['image_type'] == 'wsi'
        wsi_oof_scores = metrics.multiclass_classification_scores(y_true=df.loc[wsi_mask, 'target'], y_pred=df.loc[wsi_mask, 'prediction'])
        settings.logger.info(f'WSI OOF Scores: {json.dumps(wsi_oof_scores, indent=2)}')

        with open(model_root_directory / 'wsi_oof_scores.json', mode='w') as f:
            json.dump(wsi_oof_scores, f, indent=2, ensure_ascii=False)

        oof_scores = metrics.multiclass_classification_scores(y_true=df.loc[:, 'target'], y_pred=df.loc[:, 'prediction'])
        settings.logger.info(f'OOF Scores: {json.dumps(oof_scores, indent=2)}')

        with open(model_root_directory / 'oof_scores.json', mode='w') as f:
            json.dump(oof_scores, f, indent=2, ensure_ascii=False)

        visualization.visualize_scores(
            df_scores=df_scores,
            title=f'Scores of {len(folds)} Model(s)',
            path=model_root_directory / 'scores.png'
        )

        visualization.visualize_predictions(
            y_true=df['target'],
            y_pred=df['prediction'],
            title=f'Predictions of {len(folds)} Model(s)',
            plot_type='histogram',
            path=model_root_directory / f'oof_predictions_histogram.png'
        )

        df[['image_id', 'prediction']].to_csv(model_root_directory / 'oof_predictions.csv', index=False)
