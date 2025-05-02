import gpytorch
import torch
from tqdm import tqdm

"""Some boilerplate simple GPyTorch code."""

class SVGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, init_lengthscale=1.0, init_outputscale=1.0):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(SVGP, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(has_lengthscale=True))
        self.covar_module.outputscale = init_outputscale
        self.covar_module.base_kernel.lengthscale = init_lengthscale

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    

def train_svgp(model, likelihood, train_x, train_y, steps=100, lr=1e-2):
    model.train()
    likelihood.train()
    
    # Use the Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Use the Variational ELBO as the loss function
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, train_y.numel())
    
    loss_evo = []
    for _ in tqdm(range(steps)):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y.squeeze())
        loss.backward()
        optimizer.step()
        loss_evo.append(loss.detach().item())
    
    return loss_evo