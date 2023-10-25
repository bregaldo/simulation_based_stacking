# Stacking for Simulation-Based Inference

Code to reproduce the stacking experiments of the paper *[Simulation based stacking](https://arxiv.org/abs/#####)*.

## Install

For this work, we worked with a Python 3.10.10 environment and PyTorch v2.0.1. First follow PyTorch installation instructions [here](https://pytorch.org/).

Then, within the relevant environment, run:
```bash
pip install wandb
pip install https://github.com/eventlet/eventlet/archive/master.zip
pip install sbi
pip install sbibm
pip install altair==v4.2.2
pip install -e .
```

To use the SIR model, also follow [this installation procedure](https://github.com/sbi-benchmark/diffeqtorch#installation).

## General overview

We do not provide here the neural posterior inferences used in the paper or the different quantities we can extract from them (log probabilities, intervals, ranks, etc). However, we provide the necessary material to recompute this on the sbibm examples.

For neural posterior estimation, create a [WandB sweep configuration file](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration) as in [scripts/sweeps_yaml/](scripts/sweeps_yaml/) and run the sweep using [scripts/sweep_scheduler.ipynb](scripts/sweep_scheduler.ipynb).

Then, for each neural posterior estimation, estimate statistics using `compute_stats` function from [sbi_stacking/stats.py](sbi_stacking/stats.py) file.

Once these statistics are computed, stacking experiments can be conducted using the functions of the [sbi_stacking/stacking.py](sbi_stacking/stacking.py) file. Notebooks from [nb/](nb/) directory allow to reproduce the stacking experiments of the paper.
