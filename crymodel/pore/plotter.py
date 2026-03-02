# crymodel/pore/plotter.py
"""Plotting utilities for pyHole pore profiles."""
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt


def _parse_pair_any(arg, *, flag_name='ylim') -> Optional[Tuple[float, float]]:
    """Accept either [a b], "a,b", or "a b" and return (a,b) floats; return None if arg is None."""
    if arg is None:
        return None
    if isinstance(arg, list):
        s = " ".join(arg)
    else:
        s = str(arg)
    s = s.strip().replace(',', ' ')
    toks = [t for t in s.split() if t]
    if len(toks) != 2:
        raise ValueError(f"--{flag_name} requires two values (e.g., --{flag_name} 0.5 8.0 or --{flag_name}=0.5,8.0)")
    try:
        return float(toks[0]), float(toks[1])
    except Exception as e:
        raise ValueError(f"--{flag_name} values must be numeric: {toks}") from e


def _parse_hlines(s: Optional[str]) -> List[Tuple[float, Optional[str]]]:
    """Parse horizontal/vertical line specifications."""
    out = []
    if not s:
        return out
    for tok in s.split(','):
        tok = tok.strip()
        if not tok:
            continue
        if ':' in tok:
            y, lbl = tok.split(':', 1)
            out.append((float(y), lbl.strip()))
        else:
            out.append((float(tok), None))
    return out


def _detect_files(arg: str) -> Tuple[Path, Path, str]:
    """Return (csv_path, summary_path, label) for a prefix or CSV/JSON path."""
    p = Path(arg)
    if p.suffix.lower() == '.csv':
        csv_path = p
        summary_path = p.with_name(p.stem + '_summary.json')
        label = p.stem
    elif p.suffix.lower() == '.json' and p.name.endswith('_summary.json'):
        summary_path = p
        csv_path = p.with_name(p.name.replace('_summary.json', '.csv'))
        label = p.stem.replace('_summary', '')
    else:
        csv_path = p.with_suffix('.csv')
        summary_path = p.with_name(p.name + '_summary.json')
        label = p.name
    return csv_path, summary_path, label


def _load_one(arg: str) -> Tuple[pd.DataFrame, dict, str]:
    """Load one pyHole dataset."""
    csv_path, summary_path, label = _detect_files(arg)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            summary = json.load(f)
    else:
        sys.stderr.write(f"[warn] Summary JSON not found: {summary_path}\n")
        summary = {}
    df = pd.read_csv(csv_path)
    if 's_A' not in df.columns or 'radius_A' not in df.columns:
        raise ValueError(f"{csv_path} missing required columns s_A and radius_A")
    return df, summary, label


def _shade_blocked(ax, summary: dict, species: Optional[str], *, swap_axes: bool = False) -> None:
    """Shade blocked spans on the plot."""
    if not summary:
        return
    passes = summary.get('passability', {})
    if not passes:
        return
    items = [(species, passes.get(species))] if (species and species in passes) else list(passes.items())
    for spec, info in items:
        if not info:
            continue
        spans = info.get('blocked_spans', [])
        for b in spans:
            x0 = b.get('start_s_A', None)
            x1 = b.get('end_s_A', None)
            if x0 is None or x1 is None:
                continue
            if swap_axes:
                ax.axhspan(x0, x1, color='0.85', alpha=0.5, lw=0, label=None)
            else:
                ax.axvspan(x0, x1, color='0.85', alpha=0.5, lw=0, label=None)


def _annotate_stats(ax, summary: dict) -> None:
    """Annotate statistics on the plot."""
    if not summary:
        return
    txts = []
    if 'min_radius_A' in summary:
        try:
            txts.append(f"min r = {float(summary['min_radius_A']):.2f} Å")
        except Exception:
            pass
    if 'G_nS' in summary:
        try:
            txts.append(f"G ≈ {float(summary['G_nS']):.2f} nS")
        except Exception:
            pass
    if not txts:
        return
    ax.text(0.98, 0.02, "  •  ".join(txts), transform=ax.transAxes,
            ha='right', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.7'))


def _apply_paper_style() -> None:
    """Apply publication-quality styling."""
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
    })


def _pick_secondary_column(df: pd.DataFrame, which: str) -> Tuple[Optional[str], str]:
    """Pick secondary column from dataframe."""
    cols_lower = {c.lower(): c for c in df.columns}
    
    def pick(*names):
        for n in names:
            if n in df.columns:
                return n
            if n in cols_lower:
                return cols_lower[n]
        return None
    
    if which == 'hydro':
        col = pick('hydro_index', 'hydropathy', 'hydro')
        label = "Hydropathy index"
    elif which == 'electro':
        col = pick('electro_index', 'electrostatics', 'electro')
        label = "Electrostatics index"
    else:
        col = pick('occ_value', 'occupancy', 'occ')
        label = "Occupancy value"
    return col, label


def plot_single(
    df: pd.DataFrame,
    summary: dict,
    *,
    ax=None,
    title: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None,
    species: Optional[str] = None,
    hlines: Optional[List[Tuple[float, Optional[str]]]] = None,
    color: str = 'black',
    label: Optional[str] = None,
    swap_axes: bool = False,
    secondary: Optional[str] = None,
    sec_ylim: Optional[Tuple[float, float]] = None,
    sec_label: Optional[str] = None,
    sec_color: Optional[str] = None,
    sec_order: str = 'asc',
) -> plt.Axes:
    """Plot a single pore profile."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))
    
    # Main line
    if swap_axes:
        ax.plot(df['radius_A'], df['s_A'], lw=1.8, color=color, label=label)
        ax.set_xlabel("Radius (Å)")
        ax.set_ylabel("Axial coordinate s (Å)")
        if ylim:
            ax.set_xlim(*ylim)  # ylim refers to radius range
        if hlines:
            for y, lbl in hlines:
                ax.axvline(y, color='0.75', lw=0.8, ls='--')
                if lbl:
                    ax.text(y, ax.get_ylim()[1], f" {lbl}", rotation=90,
                            ha='left', va='top', fontsize=8, color='0.35')
        _shade_blocked(ax, summary, species, swap_axes=True)
    else:
        ax.plot(df['s_A'], df['radius_A'], lw=1.8, color=color, label=label)
        ax.set_xlabel("Axial coordinate s (Å)")
        ax.set_ylabel("Radius (Å)")
        if ylim:
            ax.set_ylim(*ylim)
        if hlines:
            for y, lbl in hlines:
                ax.axhline(y, color='0.75', lw=0.8, ls='--')
                if lbl:
                    y0, y1 = ax.get_ylim()
                    frac = (y - y0) / (y1 - y0) if y1 != y0 else 0.0
                    ax.text(0.01, frac, f" {lbl}", transform=ax.transAxes,
                            ha='left', va='center', fontsize=8, color='0.35')
        _shade_blocked(ax, summary, species, swap_axes=False)
    
    # Secondary axis
    if secondary:
        col, auto_label = _pick_secondary_column(df, secondary)
        if col is not None:
            # Choose default color by type if none provided
            use_sec_color = sec_color
            if use_sec_color is None:
                if secondary == 'electro':
                    use_sec_color = 'red'
                elif secondary == 'hydro':
                    use_sec_color = 'blue'
                else:
                    use_sec_color = '0.4'  # gray for occupancy
            if swap_axes:
                ax2 = ax.twiny()
                ax2.plot(df[col], df['s_A'], lw=1.2, alpha=0.95, color=use_sec_color)
                ax2.set_xlabel(sec_label or auto_label)
                if sec_ylim:
                    ax2.set_xlim(*sec_ylim)
                if sec_order == 'desc':
                    ax2.invert_xaxis()
            else:
                ax2 = ax.twinx()
                ax2.plot(df['s_A'], df[col], lw=1.2, alpha=0.95, color=use_sec_color)
                ax2.set_ylabel(sec_label or auto_label)
                if sec_ylim:
                    ax2.set_ylim(*sec_ylim)
                if sec_order == 'desc':
                    ax2.invert_yaxis()
        else:
            sys.stderr.write(f"[warn] Secondary column not found for '{secondary}' in columns: {list(df.columns)}\n")
    
    _annotate_stats(ax, summary)
    if title:
        ax.set_title(title)
    if label:
        ax.legend(frameon=False, loc='upper right')
    ax.grid(True, alpha=0.25)
    return ax

