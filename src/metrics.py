from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
)


def multiclass_classification_scores(y_true, y_pred):

    """
    Calculate multiclass metric scores from given ground-truth and predictions

    Parameters
    ----------
    y_true: array-like of shape (n_samples)
        Array of ground-truth values

    y_pred: array-like of shape (n_samples)
        Array of prediction values

    Returns
    -------
    scores: dict
        Dictionary of calculated multiclass classification metric scores
    """

    scores = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'micro_precision': float(precision_score(y_true, y_pred, average='micro')),
        'macro_precision': float(precision_score(y_true, y_pred, average='macro')),
        'micro_recall': float(recall_score(y_true, y_pred, average='micro')),
        'macro_recall': float(recall_score(y_true, y_pred, average='macro')),
        'micro_f1': float(f1_score(y_true, y_pred, average='micro')),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro')),
    }

    return scores
