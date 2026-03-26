"""
Entry point for running the full Sequential SPRE Lorenz experiment.

Usage
-----
    python -m lorenz_trspr

    lorenz-trspr
"""

from pathlib import Path

from .experiment import run_lorenz_experiment
from .plotting   import plot_comparison, plot_hyperparams, plot_pit
from .constants  import COV_2SIGMA


def main():
    results = run_lorenz_experiment(verbose=True)

    out = Path('./lorenz_sequential_results')

    print("\n" + "=" * 65)
    print(f"{'Coord':<8} {'Method':<15} {'Cov±2σ':>8} "
          f"{'z̄':>7} {'s_z':>7} {'KS':>7}")
    print("-" * 65)
    for coord in range(3):
        cname = ['x', 'y', 'z'][coord]
        for key, label in [('indep', 'Independent'),
                           ('geomn', 'Geom. Means'),
                           ('sequential', 'Sequential')]:
            m = results[coord][key]['metrics']
            print(f"{cname:<8} {label:<15} {m['cov_2s']:>8.3f}"
                  f"  {m['z_mean']:>7.3f}  {m['z_std']:>7.3f}  {m['ks_stat']:>7.3f}")
    print("=" * 65)
    print(f"\nCov±2σ: fraction of |z|≤2.  "
          f"Theoretical = {COV_2SIGMA:.4f} for N(0,1).")

    plot_comparison(results, out_path=str(out / 'fig_comparison.pdf'))
    plot_hyperparams(results, out_path=str(out / 'fig_hyperparams.pdf'))
    plot_pit(results,         out_path=str(out / 'fig_pit.pdf'))


if __name__ == '__main__':
    main()
