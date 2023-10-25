import wandb
import numpy as np
import os
import torch

from . import utils

def get_sweep_runs(nb_best_models, sweep_name):
    """Get the best runs from a sweep"""
    sweep_id = utils.get_sweep_id(sweep_name + ".yaml")
    wandb_api = wandb.Api()

    sweep = wandb_api.sweep(f"{utils.wb_entity}/{utils.wb_project}/{sweep_id}")
    runs_all = sorted(sweep.runs, key=lambda run: run.summary.get("best_validation_log_probs", 0), reverse=True)
    runs = []
    for run in runs_all:
        if 'best_validation_log_probs' in run.summary.keys():
            runs.append(run)
            if len(runs) == nb_best_models:
                break
    if len(runs) < nb_best_models:
        raise Exception("Not enough models!")
    return runs

def get_sweep_runs_models(runs, epoch=None, device="cpu"):
    """Get the models for each run"""
    output_dir = runs[0].summary['output_directory']
    posteriors = []
    best_val_logprobs = []
    for run in runs:
        network_fname = run.summary['output_filename']
        best_val_logprobs.append(run.summary['best_validation_log_probs'])
        if epoch is not None:
            network_fname = network_fname.replace('.pt', f'_{epoch}.pt')
        posteriors.append(torch.load(os.path.join(output_dir, network_fname), map_location=torch.device(device)))
    for posterior in posteriors:
        posterior._device = device
        posterior.potential_fn.device = device
    return posteriors, best_val_logprobs
