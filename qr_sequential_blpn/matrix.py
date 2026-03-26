"""
Test matrix: 2D Poisson 5-point stencil Laplacian.

Provides a symmetric positive-definite matrix with known closed-form
eigenvalues, used as the ground-truth benchmark for the QR extrapolation.

Build:
    A = blocktridiag(-I, B, -I),  B = tridiag(-1, 4, -1)

Closed-form eigenvalues:
    λ_{p,q} = 4 − 2 cos(p π/(l+1)) − 2 cos(q π/(m+1))
    for p = 1,…,l  and  q = 1,…,m

Choose l ≠ m to avoid the degeneracy λ_{p,q} = λ_{q,p} which prevents
QR iteration from separating the corresponding eigenvalues.
"""

import numpy as np


def build_poisson_2d(l: int, m: int) -> np.ndarray:
    """
    Build the 2D Poisson 5-point stencil Laplacian of size (m·l) × (m·l).

    Parameters
    ----------
    l : int  block size (interior grid points in one direction)
    m : int  number of blocks (grid points in other direction)

    Returns
    -------
    A : np.ndarray, shape (m*l, m*l)
    """
    n = m * l
    B = (np.diag(4.0 * np.ones(l))
         + np.diag(-1.0 * np.ones(l - 1),  1)
         + np.diag(-1.0 * np.ones(l - 1), -1))
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(m):
        A[i*l:(i+1)*l, i*l:(i+1)*l] = B
        if i > 0:
            A[i*l:(i+1)*l, (i-1)*l:i*l] = -np.eye(l)
        if i < m - 1:
            A[i*l:(i+1)*l, (i+1)*l:(i+2)*l] = -np.eye(l)
    return A


def poisson_eigenvalues_exact(l: int, m: int) -> np.ndarray:
    """
    Closed-form eigenvalues of the 2D Poisson Laplacian, sorted ascending.

    λ_{p,q} = 4 − 2 cos(p π/(l+1)) − 2 cos(q π/(m+1))
    for p = 1,…,l  and  q = 1,…,m
    """
    eigs = [
        4.0
        - 2.0 * np.cos(p * np.pi / (l + 1))
        - 2.0 * np.cos(q * np.pi / (m + 1))
        for p in range(1, l + 1)
        for q in range(1, m + 1)
    ]
    return np.sort(np.array(eigs, dtype=np.float64))


def verify_matrix(l: int, m: int) -> None:
    """
    Assert that closed-form eigenvalues match numpy's eigensolver.
    Raises AssertionError if max|exact − numpy| > 1e-10.
    """
    A       = build_poisson_2d(l, m)
    exact   = poisson_eigenvalues_exact(l, m)
    numeric = np.sort(np.linalg.eigvalsh(A))
    err     = float(np.max(np.abs(exact - numeric)))
    assert err < 1e-10, f"Eigenvalue verification failed: max error = {err:.2e}"
    print(f"  [OK] Poisson(l={l}, m={m}): n={l*m}, max|exact - numpy| = {err:.2e}")
