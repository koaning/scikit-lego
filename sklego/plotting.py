import matplotlib.pyplot as plt
import numpy as np


def plot_loess_windows(x, y, loess_smoother):
    """
    TODO: Use matplotlib.animation to create video
    Plot the data used in each window and it's accompanying focal point. Useful for debugging
    methods used in the self._get_window_indices function
    :param x:
    :param y:
    :param loess_smoother: fitted sklego.loess.LoessSmoother object
    """
    # for readability of the code abbreviate
    indices = loess_smoother.indices.copy()

    assert loess_smoother.x_focal_base.size != 0, ('Seems LoessSmoother has not been fit yet. First'
                                                   ' fit model before creating plots')

    for index in indices.keys():
        ax = plt.axes(xlim=(np.min(x), np.max(x)), ylim=(np.min(y), np.max(y)))
        ax.scatter(x[indices[index]], y[indices[index]])
        ax.scatter(loess_smoother.x_focal_base[index], loess_smoother.y_focal_base[index], c='r')
        ax.set_title(f'Window for {loess_smoother.x_focal_base[index]}')
        plt.pause(0.01)
        plt.show()