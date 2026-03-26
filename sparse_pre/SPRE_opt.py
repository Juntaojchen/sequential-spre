

import torch
import gpytorch
from .model_def import SPREModel

def SPRE_opt(A, X, Y, k_name, training_iter=100, lr=0.1):
    """
    Optimize kernel parameters for SPRE using GPyTorch with L-BFGS.
    L-BFGS is much closer to JAX's scipy.minimize(method='BFGS').
    """

    if not torch.is_tensor(A): A = torch.tensor(A)
    if not torch.is_tensor(X): X = torch.tensor(X)
    if not torch.is_tensor(Y): Y = torch.tensor(Y)
    
    A = A.double()
    X = X.double()
    Y = Y.double().flatten()
    
    ep = 1e-16
    nX = ep + (X.max(dim=0).values - X.min(dim=0).values)
    nY = ep + (Y.max() - Y.min())
    
    X_norm = X / nX
    Y_norm = Y / nY
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = SPREModel(X_norm, Y_norm, likelihood, A, k_name)
    
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter=50)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    final_loss = 0.0
    
    def closure():
        optimizer.zero_grad()
        output = model(X_norm)
        loss = -mll(output, Y_norm)
        loss.backward()
        return loss

    for i in range(100): # usually converges very fast
        loss = optimizer.step(closure)
        final_loss = loss.item()
        
    return {
        'model': model,
        'likelihood': likelihood,
        'cv': final_loss,
        'normalization': (nX, nY)
    }

