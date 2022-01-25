import numpy as np
import seaborn as sns

def plot_cov(data, **kwargs):
    vmax = np.max(np.abs(data))
    kwargs.setdefault('vmin', -vmax)
    kwargs.setdefault('vmax', vmax)
    kwargs.setdefault('cmap', 'BrBG')
    ax = sns.heatmap(data, **kwargs)
    ax.set_aspect(1)
    ax.tick_params(left=False,
                   bottom=False,
                   labelleft=False,
                   labelbottom=False)
    return ax
