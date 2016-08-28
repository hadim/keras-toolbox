from scipy.ndimage.interpolation import map_coordinates, affine_transform
from scipy.ndimage.filters import gaussian_filter
import numpy as np

from tqdm import tnrange, tqdm_notebook


def augment(X, Y, n=5, elastic_alpha_range=20, rotation_range=30,
            shift_x_range=0.1, shift_y_range=0.1):
    """I admit this code is very ugly (much worse than Keras ImageDataGenerator API
       Note to myself : dont reinvent the wheel...
    """


    X_transformed = np.zeros([X.shape[0] * n * 3] + list(X.shape[1:]))
    Y_transformed = np.zeros([X.shape[0] * n * 3] + list(X.shape[1:]))

    z = 0
    for i in tnrange(n):

        for j, (XX, YY) in tqdm_notebook(enumerate(zip(X, Y)), total=len(X), disable=False, leave=False):
            X_transformed[z], Y_transformed[z] = elastic_transform([XX, YY], elastic_alpha_range, sigma=10)
            z += 1

        for j, (XX, YY) in tqdm_notebook(enumerate(zip(X, Y)), total=len(X), disable=False, leave=False):
            X_transformed[z], Y_transformed[z] = random_rotation([XX, YY], rg=rotation_range,
                                                                  row_index=1, col_index=2,
                                                                  channel_index=0,
                                                                  fill_mode='nearest', cval=0.)
            z += 1

        for j, (XX, YY) in tqdm_notebook(enumerate(zip(X, Y)), total=len(X), disable=False, leave=False):
            X_transformed[z], Y_transformed[z] = random_shift([XX, YY], shift_x_range, shift_y_range,
                                                               row_index=1, col_index=2,
                                                               channel_index=0,
                                                               fill_mode='nearest', cval=0.)
            z += 1

    return X_transformed, Y_transformed


def elastic_transform(images, alpha_range=200, sigma=10, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """

    alpha = np.random.uniform(0, alpha_range)

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = images[0].shape
    if len(shape) == 3:
        shape = images[0].shape[1:]

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

    results = []
    for image in images:

        if len(images[0].shape) == 3:
            im = np.zeros(image.shape)
            for i, c_image in enumerate(image):
                im[i] = map_coordinates(c_image, indices, order=1).reshape(shape)
        else:
            im = map_coordinates(image, indices, order=1).reshape(shape)

        results.append(im)

    return results


def random_rotation(xx, rg, row_index=1, col_index=2, channel_index=0,
                    fill_mode='nearest', cval=0.):
    """xx is a list numpy matrices.
    """
    theta = np.pi / 180 * np.random.uniform(-rg, rg)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = xx[0].shape[row_index], xx[0].shape[col_index]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    xx = [apply_transform(x, transform_matrix, channel_index, fill_mode, cval) for x in xx]
    return xx


def random_shift(xx, wrg, hrg, row_index=1, col_index=2, channel_index=0,
                 fill_mode='nearest', cval=0.):
    """xx is a list numpy matrices.
    """
    h, w = xx[0].shape[row_index], xx[0].shape[col_index]
    tx = np.random.uniform(-hrg, hrg) * h
    ty = np.random.uniform(-wrg, wrg) * w
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])

    transform_matrix = translation_matrix  # no need to do offset
    xx = [apply_transform(x, transform_matrix, channel_index, fill_mode, cval) for x in xx]
    return xx


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, channel_index=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [affine_transform(x_channel, final_affine_matrix,
                      final_offset, order=0, mode=fill_mode, cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x
