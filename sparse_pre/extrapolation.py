# ##############################################################################
# # Sparse Probabilistic Richardson Extrapolation (SPRE)
# # PyTorch / GPyTorch Implementation
# ##############################################################################

# # Python modules
# import numpy as np
# import torch
# import matplotlib.pyplot as plt

# # Application modules
# from .SPRE import SPRE

# def extrapolation(X, Y, options=None):
#     """
#     Extrapolation to estimate f(0) from input-output training data (X, Y).

#     Parameters:
#         X : array-like of shape (n_train, d)
#             Training input vectors.
#         Y : array-like of shape (n_train,)
#             Training scalar outputs.
#         options : dict, optional
#             Extrapolation options with keys:
#                 - "name"   : str, one of {"MRE", "GRE", "SPRE"} (default: "SPRE")
#                 - "k_name" : str, one of {"Gaussian", "GaussianARD", "Matern1/2", "Matern3/2", "white"} (default: "white")
#                 - "plot"   : bool, whether to plot LOOCV results (default: True)
#                 - "plot_filename" : filename to plot LOOCV results (default: True)
       
#     Returns:
#         out : dict
#             A dictionary containing:
#                 - "mu"     : predictive mean for f(0)
#                 - "var"    : predictive variance (if available)
#                 - "mu_cv"  : LOOCV means (optional)
#                 - "var_cv" : LOOCV variances (optional)
#     """

#     # Default options
#     if options is None:
#         options = {}
#     name = options.get("name", "SPRE")
#     k_name = options.get("k_name", "white")
#     plot = options.get("plot", True)
#     plot_filename = options.get("plot_filename", "")

#     # Ensure inputs are standard numpy arrays
#     X = np.array(X)
#     Y = np.array(Y)
#     d = X.shape[1]

#     # Set up SPRE object
#     if name == "SPRE":
#         # PyTorch SPRE class handles dimension inference, but we can pass it explicitly
#         spre = SPRE(k_name=k_name, dimension=d)
#     elif name == "GRE":
#         # Pass gre_base as a torch tensor (intercept only)
#         gre_base = torch.zeros((1, d), dtype=torch.float64)
#         spre = SPRE(k_name=k_name, dimension=d, gre_base=gre_base)
#     elif name == "MRE":
#         raise NotImplementedError("MRE extrapolation is not implemented yet.")
#     else:
#         raise ValueError(f"Unknown extrapolation method: {name}")
    
#     # Set data (Handles normalization and tensor conversion internally)
#     spre.set_normalised_data(X, Y)

#     # Perform Stepwise Selection (Core Logic)
#     out = spre.stepwise_selection()

#     # Calculate errors for plotting
#     # Note: 'var_cv' from our PyTorch SPRE is a numpy array, so use np.sqrt
#     errors = np.sqrt(out["var_cv"]).flatten()
 
#     # Plot LOOCV fit if applicable
#     if (plot or plot_filename) and name != "MRE" and "mu_cv" in out and "var_cv" in out:
#         n_train = X.shape[0]
#         mu_cv_flat = np.array(out["mu_cv"]).flatten()
        
#         plt.close('all') 
#         plt.figure()
#         plt.errorbar(np.arange(1, n_train + 1), mu_cv_flat, yerr=errors, fmt='bo', label='predicted')
#         plt.scatter(np.arange(1, n_train + 1), Y, color='k', marker='x', label='actual')
#         plt.xticks(np.arange(1, n_train + 1))
#         plt.xlabel(r'$i$')
#         plt.ylabel(r'$f(\mathbf{x}_i)$')
#         plt.title(f"Leave-one-out cross validation ({name})")
#         plt.legend()
        
#         # Write file and/or show plot
#         if plot_filename:
#             plt.savefig(plot_filename) 
#         if plot:
#             plt.show()

#     return out
##############################################################################
# Sparse Probabilistic Richardson Extrapolation (SPRE)
# PyTorch / GPyTorch Implementation
##############################################################################

import numpy as np
import torch
import matplotlib.pyplot as plt

# Import the updated Wrapper class
from .SPRE import SPRE

def extrapolation(X, Y, options=None):
    """
    Extrapolation to estimate f(0).
    """
    if options is None:
        options = {}
    
    # 1. Parse Options
    name = options.get("name", "SPRE")
    k_name = options.get("k_name", "Gaussian") # Default to Gaussian usually better than white
    plot = options.get("plot", True)
    plot_filename = options.get("plot_filename", "")

    # 2. Data Prep
    # Ensure inputs are standard numpy arrays initially
    X = np.array(X)
    Y = np.array(Y)
    d = X.shape[1]

    # 3. Initialize SPRE Wrapper
    if name == "SPRE":
        spre = SPRE(k_name=k_name, dimension=d)
        
    elif name == "GRE":
        # Define GRE Basis (Intercept terms usually)
        # Using a dummy tensor basis for now, or adapt based on specific GRE logic
        gre_base = torch.zeros((1, d), dtype=torch.int64) 
        spre = SPRE(k_name=k_name, dimension=d, gre_base=gre_base)
        
    elif name == "MRE":
        # MRE usually doesn't need GP, but if implemented:
        raise NotImplementedError("MRE extrapolation is not implemented in this GPyTorch wrapper yet.")
    else:
        raise ValueError(f"Unknown extrapolation method: {name}")
    
    # 4. Pass Data
    spre.set_normalised_data(X, Y)

    # 5. Run Logic
    # (Training and Prediction happen inside here)
    out = spre.stepwise_selection()

    # 6. Visualization
    if (plot or plot_filename) and name != "MRE":
        # Ensure data is numpy for matplotlib
        # out['mu_cv'] and out['var_cv'] are already numpy from SPRE.stepwise_selection
        mu_cv_flat = out["mu_cv"].flatten()
        std_cv = np.sqrt(out["var_cv"]).flatten()
        
        n_train = X.shape[0]
        
        plt.close('all') 
        plt.figure(figsize=(8, 5))
        
        # Plot predictions with error bars
        plt.errorbar(np.arange(1, n_train + 1), mu_cv_flat, yerr=std_cv, 
                     fmt='bo', capsize=5, label='Predicted (LOOCV)')
        
        # Plot actual values
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