import os

import numpy as np
from ipywidgets import interact
import matplotlib.pyplot as plt

from tqdm import trange


def make_mosaic(im, nrows, ncols, border=1):
    """From http://nbviewer.jupyter.org/github/julienr/ipynb_playground/blob/master/keras/convmnist/keras_cnn_mnist.ipynb
    """
    import numpy.ma as ma

    nimgs = len(im)
    imshape = im[0].shape

    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                            dtype=np.float32)

    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    im
    for i in range(nimgs):

        row = int(np.floor(i / ncols))
        col = i % ncols

        mosaic[row * paddedh:row * paddedh + imshape[0],
               col * paddedw:col * paddedw + imshape[1]] = im[i]

    return mosaic


def get_weights_mosaic(model, layer_id, n=64):
    """
    """

    # Get Keras layer
    layer = model.layers[layer_id]

    # Check if this layer has weight values
    if not hasattr(layer, "W"):
        raise Exception("The layer {} of type {} does not have weights.".format(layer.name,
                                                           layer.__class__.__name__))

    weights = layer.W.get_value()

    # For now we only handle Conv layer like with 4 dimensions
    if weights.ndim != 4:
        raise Exception("The layer {} has {} dimensions which is not supported.".format(layer.name, weights.ndim))

    # n define the maximum number of weights to display
    if weights.shape[0] < n:
        n = weights.shape[0]

    # Create the mosaic of weights
    nrows = int(np.round(np.sqrt(n)))
    ncols = int(nrows)

    if nrows ** 2 < n:
        ncols +=1

    mosaic = make_mosaic(weights[:n, 0], nrows, ncols, border=1)

    return mosaic


def plot_weights(model, layer_id, n=64, ax=None, **kwargs):
    """Plot the weights of a specific layer. ndim must be 4.
    """
    import matplotlib.pyplot as plt

    # Set default matplotlib parameters
    if not 'interpolation' in kwargs.keys():
        kwargs['interpolation'] = "none"

    if not 'cmap' in kwargs.keys():
        kwargs['cmap'] = "gray"

    layer = model.layers[layer_id]

    mosaic = get_weights_mosaic(model, layer_id, n=64)

    # Plot the mosaic
    if not ax:
        fig = plt.figure()
        ax = plt.subplot()

    im = ax.imshow(mosaic, **kwargs)
    ax.set_title("Layer #{} called '{}' of type {}".format(layer_id, layer.name, layer.__class__.__name__))

    plt.colorbar(im, ax=ax)

    return ax


def plot_all_weights(model, n=64, n_columns=3, **kwargs):
    """
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Set default matplotlib parameters
    if not 'interpolation' in kwargs.keys():
        kwargs['interpolation'] = "none"

    if not 'cmap' in kwargs.keys():
        kwargs['cmap'] = "gray"

    layers_to_show = []

    for i, layer in enumerate(model.layers):
        if hasattr(layer, "W"):
            weights = layer.W.get_value()
            if weights.ndim == 4:
                layers_to_show.append((i, layer))

    n_mosaic = len(layers_to_show)
    nrows = n_mosaic // n_columns
    ncols = n_columns

    if ncols ** 2 < n_mosaic:
        nrows +=1

    fig_w = 15
    fig_h = nrows * fig_w / ncols

    fig = plt.figure(figsize=(fig_w, fig_h))

    for i, (layer_id, layer) in enumerate(layers_to_show):

        mosaic = get_weights_mosaic(model, layer_id=layer_id, n=n)

        ax = fig.add_subplot(nrows, ncols, i+1)

        im = ax.imshow(mosaic, **kwargs)
        ax.set_title("Layer #{} called '{}' \nof type {}".format(layer_id, layer.name, layer.__class__.__name__))

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)

        ax.axis('off')

        for sp in ax.spines.values():
            sp.set_visible(False)
        if ax.is_first_row():
            ax.spines['top'].set_visible(True)
        if ax.is_last_row():
            ax.spines['bottom'].set_visible(True)
        if ax.is_first_col():
            ax.spines['left'].set_visible(True)
        if ax.is_last_col():
            ax.spines['right'].set_visible(True)

    #fig.tight_layout()
    return fig


def plot_feature_map(model, layer_id, X, n=256, n_columns=3, ax=None, **kwargs):
    """
    """
    import keras.backend as K
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    layer = model.layers[layer_id]

    try:
        get_activations = K.function([model.layers[0].input, K.learning_phase()], [layer.output,])
        activations = get_activations([X, 0])[0]
    except:
        # Ugly catch, a cleaner logic is welcome here.
        raise Exception("This layer cannot be plotted.")

    # For now we only handle feature map with 4 dimensions
    if activations.ndim != 4:
        raise Exception("Feature map of '{}' has {} dimensions which is not supported.".format(layer.name,
                                                                                             activations.ndim))

    # Set default matplotlib parameters
    if not 'interpolation' in kwargs.keys():
        kwargs['interpolation'] = "none"

    if not 'cmap' in kwargs.keys():
        kwargs['cmap'] = "gray"

    # Compute nrows and ncols for images
    n_mosaic = len(activations)
    nrows = n_mosaic // n_columns
    ncols = n_columns

    if ncols ** 2 < n_mosaic:
        nrows +=1

    # Compute nrows and ncols for mosaics
    if activations[0].shape[0] < n:
        n = activations[0].shape[0]

    nrows_inside_mosaic = int(np.round(np.sqrt(n)))
    ncols_inside_mosaic = int(nrows_inside_mosaic)

    if nrows_inside_mosaic ** 2 < n:
        ncols_inside_mosaic += 1

    fig_w = 15
    fig_h = nrows * fig_w / ncols

    fig = plt.figure(figsize=(fig_w, fig_h))

    for i, feature_map in enumerate(activations):

        mosaic = make_mosaic(feature_map[:n], nrows_inside_mosaic, ncols_inside_mosaic, border=1)

        ax = fig.add_subplot(nrows, ncols, i+1)

        im = ax.imshow(mosaic, **kwargs)
        ax.set_title("Feature map #{} \nof layer#{} \ncalled '{}' \nof type {} ".format(i, layer_id,
                                                                                  layer.name,
                                                                                  layer.__class__.__name__))

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)

        ax.axis('off')

        for sp in ax.spines.values():
            sp.set_visible(False)
        if ax.is_first_row():
            ax.spines['top'].set_visible(True)
        if ax.is_last_row():
            ax.spines['bottom'].set_visible(True)
        if ax.is_first_col():
            ax.spines['left'].set_visible(True)
        if ax.is_last_col():
            ax.spines['right'].set_visible(True)

    # fig.tight_layout()
    return fig


def plot_all_feature_maps(model, X, n=256, ax=None, **kwargs):
    """
    """

    figs = []

    for i, layer in enumerate(model.layers):

        try:
            fig = plot_feature_map(model, i, X, n=n, ax=ax, **kwargs)
        except:
            pass
        else:
            figs.append(fig)

    return figs


def browse_images(images, manual_update=False):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from ipywidgets import interact

    n = len(images[list(images.keys())[0]])

    def nice_imshow(data, title, ax):
        im = ax.imshow(data, interpolation="none", cmap="gray")
        ax.set_title(title, fontsize=18)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)

    def view_image(i):

        fig = plt.figure(figsize=(16, 8))

        n_ax = len(images)

        for j, (label, data) in enumerate(images.items()):

            ax = plt.subplot(1, n_ax, j+1)

            if data[i].ndim == 3:
                nice_imshow(data[i, 0], label, ax)
            else:
                nice_imshow(data[i], label, ax)

    interact(view_image, i=(0, n - 1) , __manual=manual_update)
