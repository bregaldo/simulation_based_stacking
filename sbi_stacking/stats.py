import torch
from tqdm import tqdm


def compute_stats(x,
                  theta,
                  posteriors,
                  stats_list=["log_probs", "ranks", "intervals", "means", "stds"],
                  nb_samples=1000,
                  alpha=0.1):
    """Compute statistics for each model"""
    assert len(posteriors) > 0
    device = x.device
    nb_models = len(posteriors)
    nb_simulations = x.shape[0]
    nb_params = theta.shape[-1]

    log_probs = torch.zeros(nb_models, nb_simulations, device=device)
    ranks = torch.zeros(nb_models, nb_simulations, nb_params, device=device)
    intervals = torch.zeros(nb_models, nb_simulations, nb_params, 2, device=device)
    means = torch.zeros(nb_models, nb_simulations, nb_params, device=device)
    stds = torch.zeros(nb_models, nb_simulations, nb_params, device=device)

    # Check if we need to sample from the posterior
    need_posterior_sampling = False
    if "ranks" in stats_list or "intervals" in stats_list or "means" in stats_list or "stds" in stats_list:
        need_posterior_sampling = True

    # Compute statistics
    for i, model in enumerate(posteriors):
        # Log prob
        if "log_probs" in stats_list:
            log_prob = model.posterior_estimator.log_prob(theta, x).detach()
            log_probs[i] = log_prob 
        
        # Posterior statistics
        if need_posterior_sampling:
            for j in tqdm(range(nb_simulations)):
                theta_samples = model.sample((nb_samples,), x=x[j], show_progress_bars=False)

                # Ranks
                ranks[i, j] = torch.sum(theta_samples > theta[j], dim=0) / nb_samples

                # Central confidence intervals
                intervals[i, j, :, 0] = torch.quantile(theta_samples, alpha/2, dim=0)
                intervals[i, j, :, 1] = torch.quantile(theta_samples, 1-alpha/2, dim=0)

                # Posterior mean and std
                means[i, j] = torch.mean(theta_samples, dim=0)
                stds[i, j] = torch.std(theta_samples, dim=0)
        
    ret = {}
    if "log_probs" in stats_list:
        ret["log_probs"] = log_probs
    if "ranks" in stats_list:
        ret["ranks"] = ranks
    if "intervals" in stats_list:
        ret["intervals"] = intervals
    if "means" in stats_list:
        ret["means"] = means
    if "stds" in stats_list:
        ret["stds"] = stds

    return ret
