import numpy as np
import torch
import matplotlib.pyplot as plt

from .SPRE import SPRE

def extrapolation(X, Y, options=None):
    """
    Extrapolation to estimate f(0).
    """
    if options is None:
        options = {}
    
    name = options.get("name", "SPRE")
    k_name = options.get("k_name", "Gaussian") # Default to Gaussian usually better than white
    plot = options.get("plot", True)
    plot_filename = options.get("plot_filename", "")

    X = np.array(X)
    Y = np.array(Y)
    d = X.shape[1]

    if name == "SPRE":
        spre = SPRE(k_name=k_name, dimension=d)
        
    elif name == "GRE":
        gre_base = torch.zeros((1, d), dtype=torch.int64) 
        spre = SPRE(k_name=k_name, dimension=d, gre_base=gre_base)
        
    elif name == "MRE":
        raise NotImplementedError("MRE extrapolation is not implemented in this GPyTorch wrapper yet.")
    else:
        raise ValueError(f"Unknown extrapolation method: {name}")
    
    spre.set_normalised_data(X, Y)

    out = spre.stepwise_selection()

    if (plot or plot_filename) and name != "MRE":
        mu_cv_flat = out["mu_cv"].flatten()
        std_cv = np.sqrt(out["var_cv"]).flatten()
        
        n_train = X.shape[0]
        
        plt.close('all') 
        plt.figure(figsize=(8, 5))
        
        plt.errorbar(np.arange(1, n_train + 1), mu_cv_flat, yerr=std_cv, 
                     fmt='bo', capsize=5, label='Predicted (LOOCV)')
        
        plt.scatter(np.arange(1, n_train + 1), Y, 
                    color='r', marker='x', s=50, label='Actual Data', zorder=5)
        
        plt.xticks(np.arange(1, n_train + 1))
        plt.xlabel('Data Point Index')
        plt.ylabel('Value')
        plt.title(f"{name} Fit (Kernel: {k_name})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if plot_filename:
            plt.savefig(plot_filename) 
        if plot:
            plt.show()

    return out