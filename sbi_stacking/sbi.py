from sbi import utils as Ut
from sbi import inference as Inference
import torch

def get_density_estimator(density_estimator='maf',
                          num_transforms=3,
                          num_bins=None,
                          tail_bound=None,
                          num_blocks=2,
                          hidden_features=256,
                          use_batch_norm=True,
                          dropout_probability=0.0):
    """Get a density estimator."""
    density_estimator = Ut.posterior_nn(density_estimator,
                                        num_transforms=num_transforms,
                                        num_bins=num_bins,
                                        tail_bound=tail_bound,
                                        num_blocks=num_blocks,
                                        hidden_features=hidden_features,
                                        use_batch_norm=use_batch_norm,
                                        dropout_probability=dropout_probability)
    return density_estimator

def npe(x,
        theta,
        prior,
        density_estimator,
        batch_size=50,
        lr=1e-3,
        clip_max_norm=100.0,
        device="cpu",
        log_fn=None,
        epochs=None,
        save_every=None,
        output_path=None,
        ret_npe=False):
    """Train a neural posterior estimator (NPE) using SBI."""
    npe = Inference.SNPE(prior=prior,
                         density_estimator=density_estimator,
                         device=device)
    npe.append_simulations(theta, x)
    
    if epochs is None:
        epochs = 2**31 - 1 # Default parameter of sbi package
        stop_after_epochs = 20 # Early stopping
    else:
        stop_after_epochs = 2**31 - 1 # Disable early stopping
    

    posterior_est = npe.train(training_batch_size=batch_size,
                          learning_rate=lr,
                          clip_max_norm=clip_max_norm,
                          log_fn=log_fn,
                          max_num_epochs=epochs,
                          stop_after_epochs=stop_after_epochs,
                          save_every=save_every,
                          output_path=output_path)
    
    posterior = npe.build_posterior(posterior_est)

    if ret_npe:
        return posterior, npe
    else:
        return posterior
