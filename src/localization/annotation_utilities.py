def coco_to_voc_bounding_box(bounding_box):

    """
    Convert bounding box annotation from VOC to COCO format

    Parameters
    ----------
    bounding_box: list of shape (4)
        Bounding box with x1, y1, width, height values

    Returns
    -------
    bounding_box: list of shape (4)
        Bounding box with x1, y1, x2, y2 values
    """

    x1 = bounding_box[0]
    y1 = bounding_box[1]
    x2 = x1 + bounding_box[2]
    y2 = y1 + bounding_box[3]

    return x1, y1, x2, y2


def coco_to_yolo_bounding_box(bounding_box):

    """
    Convert bounding box annotation from COCO to YOLO format

    Parameters
    ----------
    bounding_box: list of shape (4)
        Bounding box with x1, y1, width, height values

    Returns
    -------
    bounding_box: list of shape (4)
        Bounding box with x_center, y_center, width, height values
    """

    x1 = bounding_box[0]
    y1 = bounding_box[1]
    width = bounding_box[2]
    height = bounding_box[3]
    x_center = x1 + (width // 2)
    y_center = y1 + (height // 2)

    return x_center, y_center, width, height
