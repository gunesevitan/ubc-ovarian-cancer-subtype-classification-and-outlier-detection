# UBC Ovarian Cancer Subtype Classification and Outlier Detection (UBC-OCEAN)

This was an interesting competition and I would like to thank everyone involved with the organization of it.
This is a simple textbook strong label solution that heavily relies on external TMA data.



## Raw Dataset

### WSI

Masks of WSIs are resized to thumbnail sizes.
Tiles of WSIs and masks are extracted from their thumbnails with stride of 384 and they are padded to 512.
A MaxViT Tiny FPN model is trained on those padded tiles and masks.
Segmentation model outputs are activated with sigmoid and 3x TTA (horizontal, vertical and diagonal flip) are applied after the activation.

Final segmentation mask prediction is blocky since the model was trained on tiles and merged later.

![seg1](https://i.ibb.co/jg24x1H/Screenshot-from-2024-01-04-09-28-01.png)

Segmentation mask predictions are cast to 8-bit integer and upsampled to original WSI size with nearest neighbor interpolation.

![seg2](https://i.ibb.co/ZHjtfmY/Screenshot-from-2024-01-04-09-31-42.png)

* WSI and their mask predictions are cropped maximum number of times with stride of 1024.
* Crops are sorted based on their mask areas in descending order
* Top 16 crops are taken and WSI label is assigned to them

### TMA

Rows and columns with low standard deviation are dropped on TMAs with the function below.
The purpose of this preprocessing is removing white regions and making WSIs and TMAs as similar as possible.
Using higher values of threshold were dropping areas in the tissue region so the standard deviation threshold is set to 10.

```
def drop_low_std(image, threshold):

    """
    Drop rows and columns that are below the given standard deviation threshold

    Parameters
    ----------
    image: numpy.ndarray of shape (height, width, 3)
        Image array

    threshold: int
        Standard deviation threshold

    Returns
    -------
    image: numpy.ndarray of shape (cropped_height, cropped_width, 3)
        Cropped image array
    """

    vertical_stds = image.std(axis=(1, 2))
    horizontal_stds = image.std(axis=(0, 2))
    cropped_image = image[vertical_stds > threshold, :, :]
    cropped_image = cropped_image[:, horizontal_stds > threshold, :]

    return cropped_image
```

![seg2](https://i.ibb.co/8jCyhgG/4134-crop.png)

## Validation

Multi-label stratified kfold is used as the cross-validation scheme.
Dataset is split into 5 folds.
label and is_tma columns are used for stratification.

## Models

EfficientNetV2 small model is used as the backbone with a regular classification head.

## Training

CrossEntropyLoss with class weights are used as the loss function.
Class weights are calculated as n / n ith class.

AdamW optimizer is used with 0.0001 learning rate.
Cosine annealing scheduler is used with 0.00001 minimum learning rate.

AMP is also used for faster training and regularization.

Each fold is trained for 15 epochs and epochs with highest balanced accuracy are selected.
