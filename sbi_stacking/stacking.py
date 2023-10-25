import torch.nn.functional as F
import torch
import numpy as np

from . import utils


#
# Density mixture for average KL divergence
#

def ensemble_log_prob(log_probs, weights):
    """Compute the log probability of the ensemble"""
    return torch.log(torch.sum(weights.unsqueeze(-1) * torch.exp(log_probs), dim=0))

def stacking_log_prob(log_probs,
                      n_it,
                      weights_init=None,
                      auto_init_type='uniform',
                      ret_weights_evol=False):
    """Find optimal weights for additive mixture of models"""
    device = log_probs.device
    nb_models = log_probs.shape[0]

    # Weights initialization
    if weights_init is None:
        if auto_init_type == 'uniform':
            weights_init = torch.ones(nb_models, device=device) / nb_models
        elif auto_init_type == 'random':
            weights_init = F.softmax(torch.normal(0, 1, size=(nb_models,), device=device), dim=0)
    
    # Optimize weights
    weights = weights_init.clone()
    weights_evol = []
    for i in range(n_it):
        weights_evol.append(weights.detach())
        weights = weights.detach().requires_grad_(True)
        log_prob_ensemble = ensemble_log_prob(log_probs, weights)
        loss = log_prob_ensemble.mean()
        loss.backward()
        with torch.no_grad():
            weights = weights.detach() * weights.grad # Multiplicative gradient method (cf https://arxiv.org/pdf/2207.13198.pdf)
    weights_evol.append(weights.detach())

    if ret_weights_evol:
        return weights, torch.stack(weights_evol)
    else:
        return weights

#
# Density mixture for moments stacking (Mean and Variance only)
#

def mixture_means(means, weights):
    assert means.ndim == 3 or means.ndim == 2
    assert weights.ndim == 1

    if means.ndim == 2:
        weights = weights.unsqueeze(-1)
    if means.ndim == 3:
        weights = weights.unsqueeze(-1).unsqueeze(-1)

    return torch.sum(weights*means, dim=0)

def mixture_var(means, stds, weights):
    assert means.ndim == 3 or means.ndim == 2
    assert stds.ndim == 3 or stds.ndim == 2
    assert means.ndim == stds.ndim
    assert weights.ndim == 1

    if means.ndim == 2:
        weights = weights.unsqueeze(-1)
    if means.ndim == 3:
        weights = weights.unsqueeze(-1).unsqueeze(-1)

    mmeans = torch.sum(weights*means, dim=0)
    var = torch.sum(weights*stds**2, dim=0) + torch.sum(weights*(means - mmeans)**2, dim=0)
    return var

def moments_loss(means, stds, thetas, weights):
    assert means.ndim == 3
    assert stds.ndim == 3
    assert thetas.ndim == 2
    assert weights.ndim == 1

    weights = torch.softmax(weights, dim=0)

    weights = weights.unsqueeze(-1).unsqueeze(-1)

    mmeans = torch.sum(weights*means, dim=0)
    var = torch.sum(weights*stds**2, dim=0) + torch.sum(weights*(means - mmeans)**2, dim=0)
    return torch.mean(torch.log(var) + (thetas - mmeans)**2/var)

def stacking_moments(means,
                     stds,
                     thetas,
                     n_it=2000,
                     lr=0.01):
    """Find optimal mixture weights for moments stacking (Mean and Variance only + treating parameters indepedently)."""
    assert means.ndim == 3
    assert stds.ndim == 3
    assert thetas.ndim == 2

    # General settings
    device = means.device
    nb_models = means.shape[0]

    # Weights initialization
    weights = torch.randn(nb_models, device=device)
    weights.requires_grad = True

    # Optimizer
    optimizer = torch.optim.Adam([weights], lr=lr)

    # Optimize weights
    losses = torch.zeros(n_it)
    for i in range(n_it):
        def closure():
            optimizer.zero_grad()
            l = moments_loss(means, stds, thetas, weights)
            l.backward()
            losses[i] = l.item()
            return l
        optimizer.step(closure)

    opt_weights = torch.softmax(weights, dim=0).detach()
    return opt_weights, losses

#
# Density mixture for hybrid log prob/ moments stacking (Mean and Variance only)
#

def stacking_hybrid_log_prob_moments(log_probs,
                                     means,
                                     stds,
                                     thetas,
                                     lambd=1.0,
                                     n_it=2000,
                                     lr=0.01):
    """Find optimal mixture weights for hybrid log_prob / moments stacking (Mean and Variance only + treating parameters indepedently)."""
    assert means.ndim == 3
    assert stds.ndim == 3
    assert thetas.ndim == 2

    # General settings
    device = means.device
    nb_models = means.shape[0]

    # Weights initialization
    weights = torch.randn(nb_models, device=device)
    weights.requires_grad = True

    # Optimizer
    optimizer = torch.optim.Adam([weights], lr=lr)

    # Optimize weights
    losses = torch.zeros(n_it)
    weights_evol = []
    for i in range(n_it):
        def closure():
            weights_evol.append(torch.softmax(weights.detach(), dim=0))
            optimizer.zero_grad()
            l = moments_loss(means, stds, thetas, weights)
            log_prob_ensemble = ensemble_log_prob(log_probs, torch.softmax(weights, dim=0))
            l += -lambd*log_prob_ensemble.mean()
            l.backward()
            losses[i] = l.item()
            return l
        optimizer.step(closure)
    weights_evol.append(torch.softmax(weights.detach(), dim=0))

    opt_weights = torch.softmax(weights, dim=0).detach()
    return opt_weights, losses, torch.stack(weights_evol)


#
# Density mixture for calibration
#

def w2_distance_to_uniform(ranks):
    """ Wasserstein-2 distance to uniform distribution. """
    x = torch.linspace(0, 1, int(np.sqrt(ranks.shape[0])), device=ranks.device)

    rw_quantiles = torch.quantile(ranks, x, dim=0)
    uf_quantiles = x

    w2_distance = torch.mean((rw_quantiles - uf_quantiles) ** 2, dim=0)

    return w2_distance

def cramer_von_mises_distance_to_uniform(ranks):
    """ Cramer-von Mises distance to uniform distribution. """
    x = torch.linspace(0, 1, int(np.sqrt(ranks.shape[0])), device=ranks.device)
    ranks_ecdf = torch.mean(utils.smooth_indicator(x - ranks.unsqueeze(-1), eps=0.01), dim=0)

    cvm_distance = torch.mean((ranks_ecdf - x) ** 2)

    return cvm_distance

def moments_distance_to_uniform(ranks, p=10):
    """ Moments distance to uniform distribution. """
    moments = torch.arange(1, p+1, device=ranks.device)

    ranks_moments = torch.mean(ranks.unsqueeze(-1)**moments, dim=0) ** (1/moments)

    moments_distance = torch.mean((ranks_moments - (1 / (moments + 1))**(1/moments)) ** 2)

    return moments_distance

def distance_to_uniform(ranks, weights=None, dist_type="cvm"):
    """ Compute distance to uniform distribution. """
    if weights is not None:
        assert ranks.shape[0] == weights.shape[0]
        assert ranks.ndim == 2
    else:
        assert ranks.ndim == 1

    if weights is None:
        ranks_weighted = ranks
    else:
        ranks_weighted = torch.sum(torch.softmax(weights, dim=0).unsqueeze(-1) * ranks, dim=0)

    if dist_type == "w2":
        return w2_distance_to_uniform(ranks_weighted)
    elif dist_type == "cvm":
        return cramer_von_mises_distance_to_uniform(ranks_weighted)
    elif dist_type == "moments":
        return moments_distance_to_uniform(ranks_weighted)
    else:
        raise ValueError("Invalid distance")

def stacking_ranks(ranks,
                   n_it=2000,
                   lr=0.01):
    """Find optimal mixture weights for ranks stacking."""
    assert ranks.ndim == 3

    # General settings
    device = ranks.device
    nb_models = ranks.shape[0]

    # Weights initialization
    weights = torch.randn(nb_models, device=device)
    weights.requires_grad = True

    # Optimizer
    optimizer = torch.optim.Adam([weights], lr=lr)

    # Optimize weights
    losses = torch.zeros(n_it)
    for i in range(n_it):
        def closure():
            optimizer.zero_grad()
            l = torch.zeros(1, device=device)
            for j in range(ranks.shape[-1]):
                l += distance_to_uniform(ranks[:, :, j], weights, dist_type="cvm")
            l = l / ranks.shape[-1]
            l.backward()
            losses[i] = l.item()
            return l
        optimizer.step(closure)
    opt_weights = torch.softmax(weights, dim=0).detach()
    stacked_ranks = torch.sum(opt_weights.reshape((-1,) + (1,) * (ranks.ndim - 1)) * ranks, dim=0)

    return opt_weights, stacked_ranks, losses

#
# Intervals stacking
#

def mixture_interval(intervals, weights):
    assert intervals.ndim == 4 or intervals.ndim == 3
    assert weights.ndim == 1

    weights = weights.reshape(-1, 2)

    if intervals.ndim == 3:
        weights = weights.unsqueeze(-2)
    if intervals.ndim == 4:
        weights = weights.unsqueeze(-2).unsqueeze(-2)

    return torch.sum(weights*intervals, dim=0)

def intervals_loss(intervals, thetas, weights, alpha, nparams=None, eps=0.1):
    assert intervals.ndim == 4
    assert thetas.ndim == 2
    assert weights.ndim == 1

    weights = weights.reshape(-1, 2)

    weights = weights.unsqueeze(-2).unsqueeze(-2)

    mintervals = torch.sum(weights*intervals, dim=0)

    l = mintervals[..., 1] - mintervals[..., 0]

    l += 2/alpha * (mintervals[..., 0] - thetas)*utils.smooth_indicator(mintervals[..., 0] - thetas, eps=eps)
    l += 2/alpha * (thetas - mintervals[..., 1])*utils.smooth_indicator(thetas - mintervals[..., 1], eps=eps)
    
    if nparams is not None:
        return l[..., :nparams].mean()
    else:
        return l.mean()

def stacking_invervals(intervals,
                       thetas,
                       alpha,
                       nparams=None,
                       n_it=200):
    """Find optimal weights for intervals stacking."""
    assert intervals.ndim == 4
    assert thetas.ndim == 2

    # General settings
    device = intervals.device
    nb_models = intervals.shape[0]

    # Weights initialization
    weights = torch.randn(2*nb_models, device=device)
    weights.requires_grad = True

    # Optimizer
    optimizer = torch.optim.LBFGS([weights], line_search_fn="strong_wolfe")

    # Optimize weights
    eps = (intervals[..., 1] - intervals[..., 0]).min().item() / 1000 # For smooth_indicator approximation (1/100 of the mean interval length)
    losses = torch.zeros(n_it)
    for i in range(n_it):
        def closure():
            optimizer.zero_grad()
            l = intervals_loss(intervals, thetas, weights, alpha, eps=eps, nparams=nparams)
            l.backward()
            losses[i] = l.item()
            return l
        optimizer.step(closure)

    opt_weights = weights.detach()
    return opt_weights, losses
