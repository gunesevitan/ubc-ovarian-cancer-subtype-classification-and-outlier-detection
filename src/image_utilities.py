import numpy as np
import cv2



def resize_with_aspect_ratio(image, longest_edge):

    """
    Resize image while preserving its aspect ratio

    Parameters
    ----------
    image: numpy.ndarray of shape (height, width, 3)
        Image array

    longest_edge: int
        Desired number of pixels on the longest edge

    Returns
    -------
    image: numpy.ndarray of shape (resized_height, resized_width, 3)
        Resized image array
    """

    height, width = image.shape[:2]
    scale = longest_edge / max(height, width)
    image = cv2.resize(image, dsize=(int(np.ceil(width * scale)), int(np.ceil(height * scale))), interpolation=cv2.INTER_AREA)

    return image


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


def get_largest_contour(image, threshold):

    """
    Get the largest contour from the image

    Parameters
    ----------
    image: numpy.ndarray of shape (height, width)
        Image array

    threshold: int
        Binarization threshold

    Returns
    -------
    bounding_box: list of shape (4)
        Bounding box with x1, y1, x2, y2 values
    """

    image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        x1 = 0
        x2 = image.shape[1] + 1
        y1 = 0
        y2 = image.shape[0] + 1
    else:
        contour = max(contours, key=cv2.contourArea)
        mask = np.zeros(image.shape, np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)

        y1, y2 = np.min(contour[:, :, 1]), np.max(contour[:, :, 1])
        x1, x2 = np.min(contour[:, :, 0]), np.max(contour[:, :, 0])

        x1 = int(0.999 * x1)
        x2 = int(1.001 * x2)
        y1 = int(0.999 * y1)
        y2 = int(1.001 * y2)

    bounding_box = [x1, y1, min(x2, image.shape[1]), min(y2, image.shape[0])]

    return bounding_box
