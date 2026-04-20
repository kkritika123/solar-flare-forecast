"""Plot yearly TSS/HSS trends from evaluation_results.

Default: one PNG per model (full-disk and region). Use --combined-grid to make
a single all-models figure. Run this after model.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ImportError as e:
    raise SystemExit("matplotlib is required. Install with: pip install matplotlib") from e


def _plot_style(poster: bool, grid: bool = False) -> dict:
    """Matplotlib rcParams for normal or poster mode, optionally scaled down for grid panels."""
    base = 16 if poster else 13
    lw = 1.65 if poster else 1.35
    ms = 8.0 if poster else 6.5
    if grid:
        scale = 0.62 if poster else 0.70
        base = max(7, int(round(base * scale)))
        lw *= 0.78
        ms *= 0.88
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans", "sans-serif"],
        "font.size": base,
        "font.weight": "bold",
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "axes.titlesize": max(10, base + 3),
        "axes.labelsize": max(9, base + 2),
        "xtick.labelsize": max(8, base),
        "ytick.labelsize": max(8, base),
        "legend.fontsize": max(8, base),
        "axes.edgecolor": "#1a1a1a",
        "axes.linewidth": 1.35,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "grid.linewidth": 0.9,
        "grid.alpha": 0.45,
        "lines.linewidth": lw,
        "lines.markersize": ms,
    }


def _bold_tick_labels(ax) -> None:
    for lb in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lb.set_fontweight("bold")


def _threshold_rank(t: str) -> Tuple[int, str]:
    """Sort C before M before X (unknowns last)."""
    s = str(t).strip().upper()
    letter = s[0] if s else "?"
    return ({"C": 0, "M": 1, "X": 2}.get(letter, 9), s)


def _threshold_color_map(models: List[Tuple[str, pd.DataFrame]]) -> Dict[str, int]:
    """Assign a stable color index to each C/M/X threshold so colors match across panels."""
    thrs: set[str] = set()
    for _, mdf in models:
        thrs.update(str(t) for t in mdf["threshold"].dropna().unique())
    ordered = sorted(thrs, key=_threshold_rank)
    return {t: i for i, t in enumerate(ordered)}


def _has_curves(mdf: pd.DataFrame) -> bool:
    if mdf.empty:
        return False
    for _, g in mdf.groupby("threshold", sort=True):
        g = g.drop_duplicates("year").sort_values("year")
        if g["TSS"].notna().any() or g["HSS"].notna().any():
            return True
    return False


def _scope_label(forecast_type: str) -> str:
    return "(full-disk)" if forecast_type == "full_disk" else "(region)"


def _draw_panel(
    mdf: pd.DataFrame,
    ax_tss,
    ax_hss,
    *,
    year_min: int,
    year_max: int,
    rc: dict,
    poster: bool,
    title: str,
    show_legend: bool,
    legend_ncol: int,
    hide_top_xticklabels: bool,
    color_map: Optional[Dict[str, int]] = None,
    multipanel: bool = False,
) -> bool:
    """Draw TSS on ax_tss and HSS on ax_hss for one model. Returns True if anything was drawn."""
    cmap = plt.get_cmap("tab10")
    lw = float(rc["lines.linewidth"])
    ms = float(rc["lines.markersize"])
    if multipanel:
        mew = 0.32 if poster else 0.28
        marker_edge = "#666666"
    else:
        mew = 0.55 if poster else 0.45
        marker_edge = "#444444"

    plotted = False
    for i, (thr, g) in enumerate(mdf.groupby("threshold", sort=True)):
        g = g.drop_duplicates("year").sort_values("year")
        if g["TSS"].notna().sum() == 0 and g["HSS"].notna().sum() == 0:
            continue
        ci = color_map[str(thr)] if color_map and str(thr) in color_map else i
        color = cmap(ci % 10)
        years = g["year"].astype(int).to_numpy()
        tss = g["TSS"].to_numpy(dtype=float)
        hss = g["HSS"].to_numpy(dtype=float)
        kw = dict(
            marker="o", color=color, label=f"\u2265{thr}", linewidth=lw,
            markersize=ms, markeredgewidth=mew, markeredgecolor=marker_edge,
        )
        ax_tss.plot(years, tss, **kw)
        ax_hss.plot(years, hss, **kw)
        plotted = True

    if not plotted:
        return False

    zlw = 0.5 if multipanel else 0.65
    ax_tss.set_ylabel("TSS", fontweight="bold")
    ax_tss.axhline(0.0, color="gray", linewidth=zlw, linestyle="--", alpha=0.55)
    ax_tss.set_title(title, fontweight="bold")
    ax_tss.grid(True)
    if hide_top_xticklabels:
        ax_tss.tick_params(axis="x", labelbottom=False)
    if show_legend:
        ax_tss.legend(
            loc="best", ncol=legend_ncol, frameon=True, edgecolor="#333333",
            framealpha=0.95, prop={"size": rc["legend.fontsize"], "weight": "bold"},
        )
    _bold_tick_labels(ax_tss)

    ax_hss.set_ylabel("HSS", fontweight="bold")
    ax_hss.axhline(0.0, color="gray", linewidth=zlw, linestyle="--", alpha=0.55)
    ax_hss.set_xlabel("Year", fontweight="bold")
    ax_hss.set_xticks(list(range(year_min, year_max + 1)))
    ax_hss.grid(True)
    _bold_tick_labels(ax_hss)
    return True


def _load_yearly_frame(eval_root: Path) -> pd.DataFrame:
    """Read every evaluation_results/<MODEL>/<MODEL>_yearly_scores.csv into one DataFrame."""
    paths = sorted(eval_root.glob("*/*_yearly_scores.csv"))
    if not paths:
        raise FileNotFoundError(f"No yearly scores found under {eval_root}/*/")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df = df[df["evaluation_mode"].astype(str) == "yearly"].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["TSS"] = pd.to_numeric(df["TSS"], errors="coerce")
    df["HSS"] = pd.to_numeric(df["HSS"], errors="coerce")
    return df


def plot_one_model(
    mdf: pd.DataFrame,
    year_min: int,
    year_max: int,
    out_path: Path,
    *,
    forecast_type: str,
    poster: bool,
    dpi: int,
) -> bool:
    """Write one per-model figure (TSS on top, HSS on bottom). Returns True if written."""
    mdf = mdf[
        (mdf["year"] >= year_min)
        & (mdf["year"] <= year_max)
        & (mdf["forecast_type"].astype(str) == forecast_type)
    ].copy()
    if mdf.empty:
        return False

    rc = _plot_style(poster, grid=False)
    w, h = (14, 9) if poster else (11, 7.5)
    model_name = str(mdf["model_name"].iloc[0])
    scope = _scope_label(forecast_type)
    year_span = f"{year_min}\u2013{year_max}"
    title = f"{model_name}: Yearly model trends {scope}, {year_span}"

    with plt.rc_context(rc):
        fig, (ax_t, ax_h) = plt.subplots(
            2, 1, figsize=(w, h), sharex=True, constrained_layout=True,
        )
        ok = _draw_panel(
            mdf, ax_t, ax_h,
            year_min=year_min, year_max=year_max,
            rc=rc, poster=poster, title=title,
            show_legend=True, legend_ncol=1 if poster else 2,
            hide_top_xticklabels=True,
        )
        if not ok:
            plt.close(fig)
            return False
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
    return True


def plot_all_models_grid(
    df: pd.DataFrame,
    year_min: int,
    year_max: int,
    out_path: Path,
    *,
    forecast_type: str,
    poster: bool,
    dpi: int,
    ncols: int = 0,
) -> bool:
    """One figure: grid of per-model TSS+HSS panels with a shared bottom legend."""
    df = df[
        (df["year"] >= year_min)
        & (df["year"] <= year_max)
        & (df["forecast_type"].astype(str) == forecast_type)
    ].copy()

    models: List[Tuple[str, pd.DataFrame]] = []
    for name, mdf in df.groupby("model_name", sort=True):
        if _has_curves(mdf):
            models.append((str(name), mdf))
    if not models:
        return False

    n = len(models)
    ncols_eff = ncols if ncols > 0 else max(2, int(np.ceil(np.sqrt(n))))
    nrows = int(np.ceil(n / ncols_eff))

    rc = _plot_style(poster, grid=True)
    fw = (4.2 if poster else 3.6) * ncols_eff
    fh = (3.5 if poster else 3.0) * nrows
    year_span = f"{year_min}\u2013{year_max}"
    color_map = _threshold_color_map(models)

    with plt.rc_context(rc):
        fig = plt.figure(figsize=(fw, fh))
        outer = fig.add_gridspec(
            nrows, ncols_eff,
            hspace=0.42, wspace=0.30,
            left=0.07, right=0.98, top=0.90, bottom=0.14,
        )
        for idx, (name, mdf) in enumerate(models):
            r, c = divmod(idx, ncols_eff)
            inner = outer[r, c].subgridspec(2, 1, hspace=0.07)
            ax_t = fig.add_subplot(inner[0, 0])
            ax_h = fig.add_subplot(inner[1, 0], sharex=ax_t)
            _draw_panel(
                mdf, ax_t, ax_h,
                year_min=year_min, year_max=year_max,
                rc=rc, poster=poster, title=name.replace("_", " "),
                show_legend=False, legend_ncol=1, hide_top_xticklabels=True,
                color_map=color_map, multipanel=True,
            )

        cmap = plt.get_cmap("tab10")
        handles: List[Line2D] = []
        labels: List[str] = []
        for thr in sorted(color_map.keys(), key=_threshold_rank):
            handles.append(
                Line2D(
                    [0], [0],
                    color=cmap(color_map[thr] % 10),
                    marker="o", linestyle="-",
                    linewidth=rc["lines.linewidth"],
                    markersize=rc["lines.markersize"],
                    markeredgewidth=0.32, markeredgecolor="#666666",
                )
            )
            labels.append(f"\u2265{thr}")
        fig.legend(
            handles, labels,
            loc="lower center", bbox_to_anchor=(0.5, 0.01),
            ncol=min(6, max(1, len(labels))),
            frameon=True, edgecolor="#333333", framealpha=0.95,
            prop={"size": max(8, rc["legend.fontsize"] + 1), "weight": "bold"},
        )
        fig.suptitle(
            f"Yearly TSS & HSS {_scope_label(forecast_type)}, {year_span}",
            fontweight="bold", fontsize=max(11, rc["axes.titlesize"] + 2), y=0.98,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Per-model yearly TSS/HSS line plots.")
    p.add_argument("--eval-root", default="evaluation_results")
    p.add_argument("--out-dir", default="evaluation_results/figures/per_model_yearly_trends")
    p.add_argument("--year-min", type=int, default=2020)
    p.add_argument("--year-max", type=int, default=2025)
    p.add_argument(
        "--forecast-type",
        choices=("full_disk", "region", "both"),
        default="both",
        help="Which forecast stream to plot (default: both = full_disk + region separately)",
    )
    p.add_argument("--poster", action="store_true", help="Larger fonts (for posters/slides)")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument(
        "--combined-grid", action="store_true",
        help="Write one PNG with all models in a grid",
    )
    p.add_argument(
        "--no-per-model", action="store_true",
        help="Skip per-model figures (use with --combined-grid)",
    )
    p.add_argument(
        "--grid-ncols", type=int, default=0,
        help="Columns in combined grid (0 = auto, about sqrt of model count)",
    )
    args = p.parse_args()

    if args.no_per_model and not args.combined_grid:
        p.error("--no-per-model requires --combined-grid")

    eval_root = Path(args.eval_root)
    out_dir = Path(args.out_dir)
    y0, y1 = args.year_min, args.year_max
    ft_list = ["full_disk", "region"] if args.forecast_type == "both" else [args.forecast_type]

    df = _load_yearly_frame(eval_root)

    if not args.no_per_model:
        written = 0
        for ft in ft_list:
            for name, mdf in df.groupby("model_name", sort=True):
                safe = str(name).replace("/", "_")
                out_path = out_dir / f"{safe}_yearly_tss_hss_{y0}_{y1}_{ft}.png"
                if plot_one_model(
                    mdf, y0, y1, out_path,
                    forecast_type=ft, poster=args.poster, dpi=args.dpi,
                ):
                    print("Wrote", out_path)
                    written += 1
                else:
                    print(f"Skip ({ft}): {name}")
        print(f"Done. {written} per-model figure(s) in {out_dir.resolve()}")

    if args.combined_grid:
        for ft in ft_list:
            grid_path = out_dir / f"all_models_yearly_grid_{y0}_{y1}_{ft}.png"
            if plot_all_models_grid(
                df, y0, y1, grid_path,
                forecast_type=ft, poster=args.poster,
                dpi=args.dpi, ncols=args.grid_ncols,
            ):
                print("Wrote combined grid:", grid_path.resolve())
            else:
                print(f"Combined grid ({ft}): no plottable models.")


if __name__ == "__main__":
    main()
