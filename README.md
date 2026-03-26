# Calibrating Black-Box Probabilistic Numerical Methods

**Authors:** Juntao Chen, Markus Michael Rau, Chris J. Oates  
**Institution:** Newcastle University, UK

## About

Calibrates Bayesian credible intervals for numerical codes by extrapolating multi-fidelity simulations to the infinite-precision limit. This approach couples related prediction tasks to ensure consistent uncertainty quantification without requiring complex, problem-specific statistical models.

## Overview

Two experiments demonstrating Sequential SPRE for Gaussian process hyperparameter sharing across tasks:

- Lorenz experiment (`lorenz_sequential_blpn/`): extrapolates Euler ODE solutions to step size h → 0
- QR experiment (`qr_sequential_blpn/`): extrapolates QR iteration diagonal entries to eigenvalues

Four methods are compared in each experiment: Independent REML, Geometric Means, Sequential SPRE, MOGP (ICM), and SLFM.

## Upstream SPRE Implementation

This repository contains modified and experiment-specific code based on the
official Sparse Probabilistic Richardson Extrapolation (SPRE) implementation.

Official SPRE repository:
https://github.com/NewcastleRSE/sparse-probabilistic-richardson-extrapolation

The code in this repository adapts and extends the SPRE methodology for
sequential GP hyperparameter sharing experiments on Lorenz and QR settings.

## Repository Structure (This Project)

```text
.
sequential-spre/
├── README.md
│
├── spre/                 
│   ├── __init__.py
│   ├── spre.py
│   ├── kernels.py
│   ├── kriging.py
│   ├── basis.py
│   ├── normalise.py
│   ├── optimise.py
│   ├── extrapolate.py
│   ├── selection.py
│   ├── mre.py
│   ├── utils.py
│   └── constants.py
│
├── sparse_pre/                 
│   ├── __init__.py
│   ├── SPRE.py
│   ├── SPRE_opt.py
│   ├── extrapolation.py
│   ├── helper_functions.py
│   └── model_def.py
│
├── qr_sequential_blpn/           
│   ├── __init__.py
│   ├── constants.py
│   ├── qr.py
│   ├── matrix.py
│   ├── normalise.py
│   ├── gp_utils.py
│   ├── fitting.py
│   ├── grw_fitting.py
│   ├── init_utils.py
│   ├── predict.py
│   ├── calibration.py
│   ├── metrics.py
│   ├── experiment.py
│   ├── lmc_baseline.py
│   └── slfm_baseline.py
│
├── lorenz_sequential_blpn/   
│   ├── __init__.py
│   ├── constants.py
│   ├── lorenz.py
│   ├── normalise.py
│   ├── gp_utils.py
│   ├── fitting.py
│   ├── grw_fitting.py 
│   ├── init_utils.py
│   ├── predict.py
│   ├── metrics.py
│   ├── plotting.py
│   ├── experiment.py
│   ├── lmc_baseline.py
│   └── slfm_baseline.py
```

## Requirements

- Python >= 3.9
- numpy
- scipy
- torch
- matplotlib
- jupyter
- tqdm
- scikit-learn

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run With Workspace Virtual Environment

In this repository, a local virtual environment is expected at `.venv`.

```bash
/workspaces/CBLPNM/.venv/bin/python -m pip install -r requirements.txt
```

## Attribution

Parts of this project are derived from or inspired by the official SPRE
codebase maintained by NewcastleRSE. Please refer to the upstream repository
for the canonical implementation, documentation, and project background:
https://github.com/NewcastleRSE/sparse-probabilistic-richardson-extrapolation