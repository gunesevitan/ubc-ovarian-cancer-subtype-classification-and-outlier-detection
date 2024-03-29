import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_categorical_column_distribution(df, column, title, path=None):

    """
    Visualize distribution of the given categorical column in the given dataframe

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with given categorical column

    column: str
        Name of the categorical column

    title: str
        Title of the plot

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    value_counts = df[column].value_counts()

    fig, ax = plt.subplots(figsize=(24, df[column].value_counts().shape[0] + 4), dpi=100)
    ax.bar(
        x=np.arange(len(value_counts)),
        height=value_counts.values,
    )
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks(
        np.arange(len(value_counts)),
        [
            f'{value} ({count:,})' for value, count in value_counts.to_dict().items()
        ]
    )
    ax.tick_params(axis='x', labelsize=15, pad=10)
    ax.tick_params(axis='y', labelsize=15, pad=10)
    ax.set_title(title, size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_continuous_column_distribution(df, column, title, path=None):

    """
    Visualize distribution of the given continuous column in the given dataframe

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with given continuous column

    column: str
        Name of the continuous column,

    title: str
        Title of the plot

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    fig, ax = plt.subplots(figsize=(24, 6), dpi=100)
    ax.hist(df[column], bins=16)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(
        title + f'''
        {column}
        Mean: {np.mean(df[column]):.2f} Std: {np.std(df[column]):.2f}
        Min: {np.min(df[column]):.2f} Max: {np.max(df[column]):.2f}
        ''',
        size=15,
        pad=12.5,
        loc='center',
        wrap=True
    )

    if path is None:
        plt.show()
    else:
        plt.savefig(path, bbox_inches='tight')
        plt.close(fig)


def visualize_image(image, title, mask=None, path=None):

    """
    Visualize the given image

    Parameters
    ----------
    image: numpy.ndarray of shape (height, width, channel)
        Image array

    title: str
        Title of the plot

    mask: numpy.ndarray of shape (height, width)
        Mask array



    path: str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    if mask is not None:
        ax.imshow(mask, alpha=0.5)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='x', labelsize=15, pad=10)
    ax.tick_params(axis='y', labelsize=15, pad=10)
    ax.set_title(title, size=15, pad=12.5, loc='center', wrap=True)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_learning_curve(training_losses, validation_losses, best_epoch, validation_scores=None, path=None):

    """
    Visualize learning curves of the models

    Parameters
    ----------
    training_losses: list of shape (n_epochs)
        List of training losses

    validation_losses: list of shape (n_epochs)
        List of validation losses

    best_epoch: int or None
        Epoch with the best validation loss

    validation_scores: list of shape (n_scores, n_epochs)
        List of multiple validation scores

    path: str or None
        Path of the output file (if path is None, plot is displayed with selected backend)
    """

    if validation_scores is not None:

        fig, axes = plt.subplots(figsize=(18, 18), nrows=2, dpi=100)
        axes[0].plot(np.arange(1, len(training_losses) + 1), training_losses, '-o', linewidth=2, label=f'training_loss (best: {training_losses[best_epoch]:.4f})')
        axes[0].plot(np.arange(1, len(validation_losses) + 1), validation_losses, '-o', linewidth=2, label=f'validation_loss (best: {validation_losses[best_epoch]:.4f})')
        axes[0].axvline(best_epoch + 1, color='r', label=f'Best Epoch: {best_epoch + 1}')

        if validation_scores is not None:
            for metric, scores in validation_scores.items():
                axes[1].plot(np.arange(1, len(scores) + 1), scores, '-o', linewidth=2, label=f'{metric} (best: {scores[best_epoch]:.4f})')

            axes[1].axvline(best_epoch + 1, color='r', label=f'Best Epoch: {best_epoch + 1}')

        for i in range(2):
            axes[i].set_xlabel('Epochs/Steps', size=15, labelpad=12.5)
            axes[i].set_ylabel('Losses/Metrics', size=15, labelpad=12.5)
            axes[i].set_xticks(np.arange(1, len(validation_losses) + 1), np.arange(1, len(validation_losses) + 1))

            axes[i].tick_params(axis='x', labelsize=12.5, pad=10)
            axes[i].tick_params(axis='y', labelsize=12.5, pad=10)
            axes[i].legend(prop={'size': 18})

        axes[0].set_title('Learning Curve (Losses)', size=20, pad=15)
        axes[0].set_title('Learning Curve (Metrics)', size=20, pad=15)

    else:
        fig, ax = plt.subplots(figsize=(18, 8), dpi=100)
        ax.plot(np.arange(1, len(training_losses) + 1), training_losses, '-o', linewidth=2, label=f'training_loss (best: {training_losses[best_epoch]:.4f})')
        ax.plot(np.arange(1, len(validation_losses) + 1), validation_losses, '-o', linewidth=2, label=f'validation_loss (best: {validation_losses[best_epoch]:.4f})')
        ax.axvline(best_epoch + 1, color='r', label=f'Best Epoch: {best_epoch + 1}')

        ax.set_xlabel('Epochs/Steps', size=15, labelpad=12.5)
        ax.set_ylabel('Losses/Metrics', size=15, labelpad=12.5)
        ax.set_xticks(np.arange(1, len(validation_losses) + 1), np.arange(1, len(validation_losses) + 1))

        ax.tick_params(axis='x', labelsize=12.5, pad=10)
        ax.tick_params(axis='y', labelsize=12.5, pad=10)
        ax.legend(prop={'size': 18})
        ax.set_title('Learning Curve (Losses)', size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_scores(df_scores, title, path=None):

    """
    Visualize scores of models

    Parameters
    ----------
    df_scores: pandas.DataFrame of shape (n_splits, n_metrics)
        Dataframe with one or multiple scores and metrics

    title: str
        Title of the plot

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    # Create mean and std of scores for error bars
    df_scores = df_scores.T
    column_names = df_scores.columns.to_list()
    df_scores['mean'] = df_scores[column_names].mean(axis=1)
    df_scores['std'] = df_scores[column_names].std(axis=1).fillna(0)

    fig, ax = plt.subplots(figsize=(32, 8))
    ax.barh(
        y=np.arange(df_scores.shape[0]),
        width=df_scores['mean'],
        xerr=df_scores['std'],
        align='center',
        ecolor='black',
        capsize=10
    )
    ax.set_yticks(np.arange(df_scores.shape[0]))
    ax.set_yticklabels([
        f'{metric}\n{mean:.4f} (±{std:.4f})' for metric, mean, std in zip(
            df_scores.index,
            df_scores['mean'].values,
            df_scores['std'].values
        )
    ])
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelsize=15, pad=10)
    ax.tick_params(axis='y', labelsize=15, pad=10)
    ax.set_title(title, size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_predictions(y_true, y_pred, title, plot_type, path=None):

    """
    Visualize labels and predictions as histograms

    Parameters
    ----------
    y_true: numpy.ndarray of shape (n_samples)
        Ground-truth labels

    y_pred: numpy.ndarray of shape (n_samples)
        Predicted labels

    title: str
        Title of the plot

    plot_type: str
        Type of the plot

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    fig, ax = plt.subplots(figsize=(16, 6))
    if plot_type == 'histogram':
        ax.hist(y_true, 16, alpha=0.5, label=f'Labels - Mean: {np.mean(y_true):.4f}')
        ax.hist(y_pred, 16, alpha=0.5, label=f'Predictions - Mean: {np.mean(y_pred):.4f}')
        ax.legend(prop={'size': 14})
    elif plot_type == 'scatter':
        ax.scatter(x=y_true, y=y_pred)
    elif plot_type == 'line':
        ax.plot(y_true, '-o', linewidth=2, label=f'Labels - Mean: {np.mean(y_true):.4f}')
        ax.plot(y_pred, '-o', linewidth=2, label=f'Predictions - Mean: {np.mean(y_pred):.4f}')
        ax.legend(prop={'size': 14})

    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_title(title, size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_confusion_matrix(y_true, y_pred, title, path=None):

    """
    Visualize confusion matrix of predictions

    Parameters
    ----------
    y_true: array-like of shape (n_samples)
        Array of ground-truth values

    y_pred: array-like of shape (n_samples)
        Array of prediction values

    title: str
        Title of the plot

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    cf = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize='true')
    label_mapping = {
        'HGSC': 0,
        'EC': 1,
        'CC': 2,
        'LGSC': 3,
        'MC': 4,
    }

    fig, ax = plt.subplots(figsize=(20, 20), dpi=100)
    ax = sns.heatmap(
        cf,
        annot=True,
        square=True,
        cmap='coolwarm',
        annot_kws={'size': 12},
        fmt='.2f'
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=15)
    ax.set_xticks(np.arange(len(label_mapping)) + 0.5, list(label_mapping.keys()))
    ax.set_yticks(np.arange(len(label_mapping)) + 0.5, list(label_mapping.keys()))
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_title(title, size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)
