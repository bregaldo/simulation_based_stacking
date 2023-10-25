import os
import json
import yaml
import wandb
import torch

# To be adapted to your own system
base_proj_dir = "~/simulation_based_stacking/sbi_stacking/"
base_out_dir = "~/simulation_based_stacking/output/"

# WandB variables
wb_entity = "#entity"
wb_project = "stacking"

def get_output_dir(system=None):
    """Get output directory"""
    ret_dir = base_out_dir
    if system is not None:
        ret_dir = os.path.join(ret_dir, system)
    return ret_dir

def get_sweep_id(wb_config_name):
    """Get sweep id"""
    sweeps_dir = os.path.join(base_proj_dir, "scripts")
    sweep_id_dict_name = os.path.join(sweeps_dir, "sweep_ids.json")
    with open(sweep_id_dict_name) as f:
        sweep_dict = json.load(f)
        if wb_config_name in sweep_dict:
            sweep_id = sweep_dict[wb_config_name]
        else:
            # Initialize sweep by passing in config
            config = yaml.load(open(os.path.join(sweeps_dir, "sweeps_yaml", wb_config_name)), Loader=yaml.Loader)
            sweep_id = wandb.sweep(sweep=config,
                                   project=wb_project,
                                   entity=wb_entity)
            sweep_dict[wb_config_name] = sweep_id
            with open(sweep_id_dict_name, 'w') as f:
                json.dump(sweep_dict, f, indent=4)
            print(f"New sweep_id created for config {wb_config_name}: {sweep_id}. Added to the sweep_ids dictionary!")
            print("Exiting...")
            os._exit(00)
    return sweep_id

def get_stats(system, stats_name):
    """Get stats"""
    output_dir = get_output_dir(system)
    return torch.load(os.path.join(output_dir, stats_name + ".pt"))

def smooth_indicator(x, eps=0.1):
    return (torch.tanh(x/eps) + 1)/2
