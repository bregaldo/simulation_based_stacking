import os, time
import sbibm
import sbi_stacking.sbi as sbi
import sbi_stacking.utils as utils
from pyro import distributions as pdist
import wandb
import argparse
from pathlib import Path
import torch


def get_args(interactive=False):
    parser = argparse.ArgumentParser('NPE Script', add_help=False)

    # General settings
    parser.add_argument('--task', default='two_moons', type=str)
    parser.add_argument('--output_dir', default=utils.get_output_dir("two_moons"), type=str)
    parser.add_argument('--num_simulations', default=10000, type=int)
    parser.add_argument('--epochs', default=None, type=int) # if None, we do early stopping
    parser.add_argument('--save_every', default=None, type=int) # if None, we only save the best model

    # WandB parameters
    parser.add_argument('--wb_project', default='stacking', type=str)
    parser.add_argument('--wb_job_type', default='npe', type=str)
    parser.add_argument('--wb_group', default=None, type=str)
    parser.add_argument('--wb_name', default=None, type=str)
    parser.add_argument('--wb_sweep_id', default='', type=str)
    parser.add_argument('--wb_sweep_cnt', default=1, type=int)

    # NPE parameters
    parser.add_argument('--density_estimator', default='maf', type=str)
    parser.add_argument('--num_transforms', default=3, type=int)
    parser.add_argument('--num_bins', default=None, type=int)
    parser.add_argument('--tail_bound', default=None, type=float)
    parser.add_argument('--hidden_features', default=256, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--dropout_probability', default=0.0, type=float)
    parser.add_argument('--use_batch_norm', default=True, type=bool)

    # Training parameters
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--clip_max_norm', default=5.0, type=float)

    if interactive:
        args = parser.parse_args('')
    else:
        args = parser.parse_args()
    return args

def main(args):
    # General settings
    num_simulations = args.num_simulations
    device = "cuda:0"

    # Get task, prior, simulator
    task = sbibm.get_task(args.task)
    prior = task.get_prior()
    simulator = task.get_simulator()
    prior_params = task.prior_params.copy()
    for key in prior_params:
        prior_params[key] = prior_params[key].to(device)
    if args.task in ['two_moons', 'slcp']:
        prior_dist = pdist.Uniform(**prior_params).to_event(1)
    elif args.task in ['sir']:
        prior_dist = pdist.LogNormal(**prior_params).to_event(1)
    else:
        raise NotImplementedError
    prior_dist = prior_dist.set_default_validate_args(False)

    print("Task name: ", task.name)
    print("Task display name: ", task.name_display)
    print("Dimensionality data: ", task.dim_data)
    print("Dimensionality parameters: ", task.dim_parameters)
    print("Number of observations: ", task.num_observations)

    # Wandb initialization
    if len(args.wb_sweep_id) > 0: # if we are running a sweep
        wandb.init(group=args.wb_group,
                   job_type=args.wb_job_type)

        # We update the args with the sweep config
        for key, value in wandb.config.items():
            print(f"Updating {key} to {value}")
            setattr(args, key, value)
    else:
        wandb.init(project=args.wb_project,
                   group=args.wb_group,
                   job_type=args.wb_job_type,
                   entity=utils.wb_entity,
                   name=args.wb_name,
                   config=args)
    
    # We update the wandb run name
    run_name = 'npe' \
                + f'_{args.density_estimator}' \
                + f'_nt_{args.num_transforms}' \
                + f'_hf_{args.hidden_features}' \
                + f'_nb_{args.num_blocks}' \
                + f'_lr_{args.lr:.6f}' \
                + f'_bs_{args.batch_size}'
    wandb.run.name = run_name

    # Simulate data
    theta = prior(num_samples=num_simulations)
    x = simulator(theta)

    #
    # NPE
    #

    start_time = time.time()

    # Train NPE
    dens_est = sbi.get_density_estimator(density_estimator=args.density_estimator,
                                         num_transforms=args.num_transforms,
                                         num_bins=args.num_bins,
                                         tail_bound=args.tail_bound,
                                         num_blocks=args.num_blocks,
                                         hidden_features=args.hidden_features,
                                         use_batch_norm=args.use_batch_norm,
                                         dropout_probability=args.dropout_probability)
    posterior, npe = sbi.npe(x.to(device), theta.to(device), prior_dist, dens_est,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        clip_max_norm=args.clip_max_norm,
                        log_fn=wandb.log,
                        device=device,
                        epochs=args.epochs,
                        save_every=args.save_every,
                        output_path=os.path.join(args.output_dir, run_name),
                        ret_npe=True)
    
    # Save posterior
    fname = f'{run_name}.pt'
    torch.save(posterior, os.path.join(args.output_dir, fname))

    # Log
    nb_epochs = npe._summary['epochs_trained'][0]
    wandb.log({'output_directory': os.path.join(args.output_dir),
               'output_filename': fname})
    wandb.log({'nb_params': sum(p.numel() for p in npe._neural_net.parameters()),
               'nb_trainable_params': sum(p.numel() for p in npe._neural_net.parameters() if p.requires_grad)})
    wandb.log({'best_validation_log_probs': npe._summary['best_validation_log_prob'][0],
               'epochs': nb_epochs,
               'training_time': (time.time() - start_time)})

    print('\nBest validation log prob:', npe._summary['best_validation_log_prob'])
    print(f'Training took: {(time.time() - start_time):.1f}s')

if __name__ == '__main__':
    args = get_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if len(args.wb_sweep_id) > 0:
        if args.output_dir:
            args.output_dir = os.path.join(args.output_dir, args.wb_sweep_id)
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        wandb.agent(args.wb_sweep_id,
                    function=lambda: main(args),
                    count=args.wb_sweep_cnt,
                    entity=utils.wb_entity,
                    project=args.wb_project)
    else:
        main(args)
