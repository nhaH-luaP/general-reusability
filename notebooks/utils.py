import matplotlib.pyplot as plt
import numpy as np


def plot_conf(conf, labels, vmax, fig, ax, idx=0, headers=None, colorbar=False):
    # Plot conf matrix
    conf = np.array(conf)
    avg_conf = np.array([[conf[i][j] for i in range(conf.shape[0]) if i != j] for j in range(conf.shape[1])])
    avg_conf = np.array([int(i) for i in np.mean(avg_conf, axis=1)]).reshape(1,-1)

    im = ax[0][idx].imshow(conf, vmin=0, vmax=vmax)
    if colorbar:
        fig.colorbar(im, orientation='vertical')

    # Show all ticks and label them with the respective list entries
    ax[0][idx].set_xticks(np.arange(len(labels)), labels=labels)
    if idx == 0:
        ax[0][idx].set_yticks(np.arange(len(labels)), labels=labels)
    else:
        ax[0][idx].tick_params(left = False, labelleft = False)

    # Let the horizontal axes labeling appear on top.
    ax[0][idx].tick_params(top=True, bottom=False,
                labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax[0][idx].get_xticklabels(), rotation=-30, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax[0][idx].text(j, i, conf[i, j] if i != j else "0", size=11,
                        ha="center", va="center", color="w")

    # Average Below
    ax[1][idx].imshow(avg_conf, vmin=0, vmax=vmax)
    for j in range(len(labels)):
        text = ax[1][idx].text(j, 0, avg_conf[0, j], size=11,
                        ha="center", va="center", color="w")

    ax[1][idx].tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False)

    ax[0][idx].set_anchor('W')
    ax[1][idx].set_anchor('W')

    if headers:
        ax[0][idx].set_title(headers[idx])