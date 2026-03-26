"""
Publication-quality figures for the Sequential SPRE Lorenz experiment.

Three figure types:
    plot_comparison   — error bars + z-scores (3 rows × 2 cols)
    plot_hyperparams  — amplitude & lengthscale trajectories (3 rows × 2 cols)
    plot_pit          — PIT histograms (3 rows × 3 cols)
"""

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import binom as sp_binom

from .constants import COV_2SIGMA

_RC = {
    'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'axes.labelsize': 11, 'axes.linewidth': 0.9,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.size': 4,   'ytick.major.size': 4,
    'figure.dpi': 150,
}

_C = dict(
    truth   ='#B22222',
    ind_fill='#C5D8F5', ind_eb='#2859A0', ind_mk='#4472C4',
    gm_fill ='#FDE8CC', gm_eb ='#A85010', gm_mk ='#E07B2A',
    tr_fill ='#C8E8C8', tr_eb ='#1B6B1B', tr_mk ='#2E8B2E',
)

_COORD_LABELS = [r'$x(T)$', r'$y(T)$', r'$z(T)$']
_PANELS       = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']


def plot_comparison(
    results:  Dict,
    out_path: Optional[str] = None,
    dpi:      int = 600,
):
    """
    3-row × 2-col comparison figure.

    Column 0: error bars (Independent / Geom. Means / Sequential + ground truth).
    Column 1: z-scores over time with ±1σ and ±2σ reference lines.

    Parameters
    ----------
    results  : dict keyed by coord ∈ {0,1,2} — output of run_lorenz_experiment
    out_path : optional file path to save (PDF recommended)
    dpi      : resolution for rasterised output
    """
    plt.rcParams.update(_RC)
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex='col')
    fig.subplots_adjust(hspace=0.06, wspace=0.30)

    for row, coord in enumerate(range(3)):
        r = results[coord]
        T = r['T']

        ax = axes[row, 0]
        for key, fc, ec, mc, lbl in [
            ('indep', _C['ind_fill'], _C['ind_eb'], _C['ind_mk'], 'Independent'),
            ('geomn', _C['gm_fill'],  _C['gm_eb'],  _C['gm_mk'],  'Geom. Means'),
            ('sequential', _C['tr_fill'],  _C['tr_eb'],  _C['tr_mk'],  'Sequential'),
        ]:
            import numpy as np
            mu  = r[key]['mu']
            std = np.maximum(r[key]['std'], 1e-12)
            cov = r[key]['metrics']['cov_2s']
            ax.fill_between(T, mu - 2*std, mu + 2*std,
                            color=fc, alpha=0.35, linewidth=0, zorder=1)
            ax.errorbar(T, mu, yerr=2*std, fmt='none',
                        ecolor=ec, elinewidth=0.9, capsize=2.5,
                        alpha=0.75, zorder=2)
            ax.plot(T, mu, 'o', color=mc,
                    markerfacecolor='white', markeredgecolor=mc,
                    markeredgewidth=1.2, markersize=3.5, zorder=3,
                    label=rf'{lbl}  Cov$_{{\pm2\sigma}}$={cov:.3f}')

        ax.plot(T, r['truth'], color=_C['truth'], lw=1.8,
                label='Truth', zorder=5)
        ax.set_ylabel(_COORD_LABELS[coord], fontsize=11)
        ax.text(-0.12, 1.03, _PANELS[row * 2],
                transform=ax.transAxes, fontsize=11, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.25, linewidth=0.4)
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        if row == 0:
            ax.legend(fontsize=8, framealpha=0.9,
                      edgecolor='#BBBBBB', fancybox=False, loc='upper right')

        ax2 = axes[row, 1]
        for key, mc, lbl in [
            ('indep', _C['ind_mk'], 'Independent'),
            ('geomn', _C['gm_mk'],  'Geom. Means'),
            ('sequential', _C['tr_mk'],  'Sequential'),
        ]:
            z = r[key]['metrics']['z']
            ax2.plot(T, z, 'o-', color=mc, markersize=2.5,
                     markerfacecolor='white', markeredgewidth=0.8,
                     linewidth=0.8, label=lbl, zorder=3)

        ax2.axhline(0,  color='k',        lw=0.7, alpha=0.5)
        ax2.axhline( 2, color='#D62728',  lw=0.7, ls='--', alpha=0.5)
        ax2.axhline(-2, color='#D62728',  lw=0.7, ls='--', alpha=0.5)
        ax2.axhline( 1, color='#FF8C00',  lw=0.7, ls=':',  alpha=0.5)
        ax2.axhline(-1, color='#FF8C00',  lw=0.7, ls=':',  alpha=0.5)
        ax2.set_ylim(-5, 5)
        ax2.set_ylabel(rf'z-score ({_COORD_LABELS[coord]})', fontsize=10)
        ax2.text(-0.12, 1.03, _PANELS[row * 2 + 1],
                 transform=ax2.transAxes, fontsize=11, fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.25, linewidth=0.4)
        ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
        if row == 0:
            ax2.legend(fontsize=8, framealpha=0.9,
                       edgecolor='#BBBBBB', fancybox=False)

    axes[-1, 0].set_xlabel(r'Time $T$', fontsize=11)
    axes[-1, 1].set_xlabel(r'Time $T$', fontsize=11)

    fig.suptitle(
        rf'Lorenz System · Euler method · $\pm2\sigma$ intervals'
        rf' (theoretical coverage {COV_2SIGMA:.4f})'
        r' · Indep / Geom. Means / Sequential',
        fontsize=10.5, y=1.003,
    )
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.06, wspace=0.30)
    if out_path:
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f'Saved → {out_path}')
    plt.show()
    return fig, axes


def plot_hyperparams(
    results:  Dict,
    out_path: Optional[str] = None,
    dpi:      int = 600,
):
    """
    Hyperparameter trajectories: amplitude and lengthscale over time.
    3-row × 2-col; y-axis log-scaled.
    """
    plt.rcParams.update(_RC)
    fig, axes = plt.subplots(3, 2, figsize=(11, 7), sharex=True)
    fig.subplots_adjust(hspace=0.06, wspace=0.35)

    for row, coord in enumerate(range(3)):
        r = results[coord]
        T = r['T']
        for col, (key_hp, label_hp) in enumerate([
            ('amp', 'Amplitude'),
            ('ell', 'Lengthscale'),
        ]):
            ax = axes[row, col]
            ax.plot(T, r['indep'][key_hp], 'o-',
                    color=_C['ind_mk'], ms=3, lw=0.9, alpha=0.7,
                    label='Independent')
            gm_val = float(r['geomn'][key_hp][0])
            ax.axhline(gm_val, color=_C['gm_eb'], lw=1.4, ls='--', alpha=0.85,
                       label='Geom. Means')
            ax.plot(T, r['sequential'][key_hp], 's-',
                    color=_C['tr_mk'],  ms=3, lw=1.2,
                    label='Sequential')
            ax.set_yscale('log')
            ax.set_ylabel(f'{label_hp} ({_COORD_LABELS[coord]})', fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.25, linewidth=0.4)
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            if row == 0:
                ax.legend(fontsize=8, framealpha=0.9,
                          edgecolor='#BBBBBB', fancybox=False)
            if row == 2:
                ax.set_xlabel(r'Time $T$', fontsize=11)

    fig.suptitle('Hyperparameter Trajectories: Independent / Geom. Means / Sequential',
                 fontsize=12)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f'Saved → {out_path}')
    plt.show()
    return fig, axes


def plot_pit(
    results:  Dict,
    out_path: Optional[str] = None,
    dpi:      int = 600,
):
    """
    PIT histograms for all three methods, 3 rows × 3 columns.
    Shaded band = 95 % binomial confidence interval for uniform distribution.
    """
    plt.rcParams.update(_RC)
    n_bins = 10
    fig, axes = plt.subplots(3, 3, figsize=(13, 8))
    fig.subplots_adjust(hspace=0.40, wspace=0.32)

    for row, coord in enumerate(range(3)):
        r = results[coord]
        for col, (key, lbl, clr) in enumerate([
            ('indep', 'Independent', _C['ind_mk']),
            ('geomn', 'Geom. Means', _C['gm_mk']),
            ('sequential', 'Sequential', _C['tr_mk']),
        ]):
            ax  = axes[row, col]
            pit = r[key]['metrics']['pit']
            n   = len(pit)
            p   = 1.0 / n_bins
            lo  = sp_binom.ppf(0.025, n, p) / (n * p)
            hi  = sp_binom.ppf(0.975, n, p) / (n * p)

            ax.axhspan(lo, hi, color=clr, alpha=0.15)
            ax.axhline(1.0, color='#B22222', lw=1.2, ls='--')
            ax.hist(pit, bins=n_bins, range=(0, 1), density=True,
                    color=clr, alpha=0.7, edgecolor='white', linewidth=0.8)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, max(2.5, hi * 1.3))
            ax.set_xlabel('PIT', fontsize=9)
            ax.set_ylabel('Density', fontsize=9)
            ks = r[key]['metrics']['ks_stat']
            ax.set_title(f'{_COORD_LABELS[coord]} — {lbl}\nKS={ks:.3f}',
                         fontsize=9)

    plt.suptitle('PIT Histograms', fontsize=12)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f'Saved → {out_path}')
    plt.show()
    return fig, axes
