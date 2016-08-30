import numpy as np


def mask_to_label_images(data, labels_axis=1):
    """Take a 4D array (n_images, 1, width, height) and use the single dimension axis to transform into
    a multiple dimensions axis that contain 0 or 1 according to the presence of a label. The labels are
    guessed using np.unique(data).
    """

    # Get all the labels in the dataset
    labels = np.unique(data)

    new_dims = list(data.shape)
    new_dims[labels_axis] = len(labels)

    data_labels = np.zeros(new_dims)
    data_labels = data_labels.swapaxes(0, labels_axis)

    for i, label in enumerate(labels):
        mask = (data == label).squeeze(axis=labels_axis)
        data_labels[i][mask] = 1

    data_labels = data_labels.swapaxes(0, labels_axis)

    return data_labels


def label_images_to_mask(data, labels_axis=1):
    """Revert mask_to_label_images()
    """

    new_dims = list(data.shape)
    new_dims[labels_axis] = 1

    data_mask = np.zeros(new_dims)
    data_mask = data_mask.swapaxes(0, labels_axis)

    data = data.swapaxes(0, labels_axis)

    for i in range(data.shape[0]):
        data_mask[0][data[i] == 1] = i

    data = data.swapaxes(0, labels_axis)
    data_mask = data_mask.swapaxes(0, labels_axis)

    return data_mask
