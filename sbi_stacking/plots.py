import numpy as np
import matplotlib.pyplot as plt

from . import stacking

def plot_ranks(ranks, nb_models_to_plot, nb_params_to_plot, mode="separate", bins=20, title=None, dist_type="cvm"):
    assert ranks.ndim == 3 # (nb_models, nb_samples, nb_params)
    assert nb_models_to_plot <= ranks.shape[0]
    assert nb_params_to_plot <= ranks.shape[2]

    if mode == "separate":
        fig, axs = plt.subplots(nb_models_to_plot, nb_params_to_plot, figsize=(4*nb_params_to_plot, 4*nb_models_to_plot), sharey=True, sharex=True)
        if nb_models_to_plot == 1:
            axs = axs.reshape(1, -1)
        for i in range(nb_models_to_plot):
            for j in range(nb_params_to_plot):
                axs[i, j].hist(ranks[i, :, j].cpu(), bins=bins, density=True, histtype='step')
                dist = stacking.distance_to_uniform(ranks[i, :, j], dist_type=dist_type).item()
                axs[i, j].text(0.05, 0.95, f"{dist_type} dist to uni = {dist:.4f}", transform=axs[i, j].transAxes, va='top', ha='left')
                axs[-1, j].set_xlabel("Rank")
                axs[0, j].set_title(f"$\\theta_{j+1}$")
            axs[i, 0].set_ylabel("Density")
    elif mode == "stacked":
        fig, axs = plt.subplots(1, nb_params_to_plot, figsize=(4*nb_params_to_plot, 4), sharey=True, sharex=True)
        for i in range(nb_models_to_plot):
            for j in range(nb_params_to_plot):
                axs[j].hist(ranks[i, :, j].cpu(), label=f"Model {i+1}", bins=bins, density=True, histtype='step')
                axs[j].set_xlabel("Rank")
                axs[j].set_title(f"$\\theta_{j+1}$")
        axs[0].set_ylabel("Density")
        axs[0].legend()
    else:
        raise ValueError("Invalid mode")
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
