import torch
from torch import nn

class PosteriorSample(nn.Module):
    """Represents an approximate sparse GP posterior sample computed via Matheron pathwise conditioning.
    
    The sparse GP model needs to be a GPyTorch sparse GP with a Cholesky variational distribution strategy.
    Also, the covariance function *has* to be an RBF kernel decorated with a scale kernel. 

    If you want to use a non GPyTorch sparse GP, you just need to edit how the GP hypers l and sigma_f as 
    well as Z, m, and S (i.e. the sampling of u) are handled.
    
    This object represents a function that can be queried arbitrarily via its .forward() method, meaning it
    is amenable to e.g. gradient-based optimisation.
    
    This class implements https://arxiv.org/pdf/2002.09309 .
    
    """
    def __init__(self, sgp, fourier_features=1000, x_dim=1):
        super().__init__()
        w = torch.randn((fourier_features,)) # w_i ~ N(0, 1)
        tau = torch.rand((fourier_features,)) * 2 * torch.pi # tau_i ~ U(0, 2pi)
        l = sgp.covar_module.base_kernel.lengthscale.detach().squeeze()
        
        while len(l.shape) < 1:
            l = l.unsqueeze(0)
        assert len(l.shape) == 1
        assert l.shape[0] == 1 or l.shape[0] == x_dim # depends if we're using ARD or not.

        sigma_f = sgp.covar_module.outputscale.detach().squeeze().sqrt()
        w = sigma_f * w # ensure outputscale is maintained
        theta = torch.randn((fourier_features, x_dim)) / l # theta_i ~ kernel's spectral density (see https://docs.gpytorch.ai/en/latest/kernels.html#rffkernel)

        with torch.no_grad():
            Z = sgp.variational_strategy.inducing_points
            m = sgp.variational_strategy.variational_distribution.mean
            S = sgp.variational_strategy.variational_distribution.covariance_matrix
            
        Kmm = sgp.covar_module(Z).evaluate()
        Lmm = torch.linalg.cholesky_ex(Kmm + torch.eye(Z.shape[0])*0.0001)[0]
        Kmm_inv = torch.cholesky_inverse(Lmm)
            
        q_u = torch.distributions.MultivariateNormal(m, S) # gpytorch parameterises u in "whitened" space
        # u = sgp.mean_module(Z) + (Lmm @ q_u.sample()) # so we need to "unwhiten" it here.
        u = (Lmm @ q_u.sample()) # so we need to "unwhiten" it here.

        Phi_Z = torch.tensor(2 / fourier_features).sqrt() * torch.cos(Z @ theta.T + tau.unsqueeze(0)) # shape (M, fourier_features)
        v = Kmm_inv @ (u - Phi_Z @ w) # shape (M,)
        
        self.w = w.detach()
        self.theta = theta.detach()
        self.tau = tau
        self.v = v.detach()
        with torch.no_grad():
            self.k = sgp.covar_module
            self.mean = sgp.mean_module
        self.num_ff = fourier_features
        self.Z = Z
        self.x_dim = x_dim

    def forward(self, X):
        # this implements Eq (13) of https://arxiv.org/pdf/2002.09309
        Phi_X = torch.tensor(2 / self.num_ff).sqrt() * torch.cos(X @ self.theta.T + self.tau.unsqueeze(0)) # shape (N, fourier_features)
        prior_term = Phi_X @ self.w + self.mean(X)

        k = self.k(X, self.Z).evaluate() # shape (N, M)
        update_term = k @ self.v

        return prior_term + update_term