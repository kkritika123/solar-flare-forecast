"""
Plot yearly TSS/HSS trends from evaluation_results.

Default: full-disk and region PNGs per model.
Use --combined-grid to make one all-models figure.
Run this after model.py.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required. Install with: pip install matplotlib"
    ) from e


def _plot_style_rc(poster: bool) -> dict:
    """Bold sans-serif labels; data lines kept moderate (not heavy)."""
    common = {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans", "sans-serif"],
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "axes.edgecolor": "#1a1a1a",
        "axes.linewidth": 1.35,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.minor.width": 0.9,
        "ytick.minor.width": 0.9,
        "grid.linewidth": 0.9,
        "grid.alpha": 0.45,
    }
    if poster:
        return {
            **common,
            "font.size": 16,
            "font.weight": "bold",
            "axes.titlesize": 20,
            "axes.labelsize": 17,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
            "lines.linewidth": 1.65,
            "lines.markersize": 8,
        }
    return {
        **common,
        "font.size": 13,
        "font.weight": "bold",
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "lines.linewidth": 1.35,
        "lines.markersize": 6.5,
    }


def _bold_tick_labels(ax) -> None:
    for lb in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lb.set_fontweight("bold")


def _plot_style_rc_grid(poster: bool) -> dict:
    """Scaled-down type and markers for many panels on one page."""
    base = _plot_style_rc(poster)
    out = {**base}
    scale = 0.62 if poster else 0.70
    for k in (
        "font.size",
        "axes.titlesize",
        "axes.labelsize",
        "xtick.labelsize",
        "ytick.labelsize",
        "legend.fontsize",
    ):
        v = out.get(k)
        if isinstance(v, (int, float)):
            out[k] = max(7, int(round(float(v) * scale)))
    # Grid panels are small; lighter strokes keep them from looking crowded.
    out["lines.linewidth"] = float(out["lines.linewidth"]) * 0.78
    out["lines.markersize"] = float(out["lines.markersize"]) * 0.88
    return out


def _flare_threshold_rank(threshold: str) -> Tuple[int, str]:
    """Sort C-class before M before X (and unknowns last)."""
    s = str(threshold).strip().upper()
    letter = s[0] if s else "?"
    order = {"C": 0, "M": 1, "X": 2}
    return (order.get(letter, 9), s)


def _norm_threshold_key(key: Any) -> str:
    if isinstance(key, tuple) and len(key) == 1:
        return str(key[0])
    return str(key)


def _global_threshold_color_map(
    models: List[Tuple[str, pd.DataFrame]],
    forecast_type_filter: str,
) -> Optional[Dict[str, int]]:
    """Keep threshold colors stable (C/M/X) across all model panels."""
    if forecast_type_filter == "all":
        return None
    thrs: set[str] = set()
    for _, mdf in models:
        for key, _ in mdf.groupby(["threshold"], sort=False):
            thrs.add(_norm_threshold_key(key))
    if not thrs:
        return None
    ordered = sorted(thrs, key=lambda t: _flare_threshold_rank(t))
    return {t: i for i, t in enumerate(ordered)}


def _global_ftype_thr_color_map(
    models: List[Tuple[str, pd.DataFrame]],
    forecast_type_filter: str,
) -> Optional[Dict[Tuple[str, str], int]]:
    # Used only when we plot both forecast streams together.
    if forecast_type_filter != "all":
        return None
    seen: set[Tuple[str, str]] = set()
    for _, mdf in models:
        for key, _ in mdf.groupby(["forecast_type", "threshold"], sort=False):
            ftype, thr = key  # type: ignore[misc]
            seen.add((str(ftype), str(thr)))
    if not seen:
        return None
    ordered = sorted(seen, key=lambda p: (p[0], _flare_threshold_rank(p[1])))
    return {p: i for i, p in enumerate(ordered)}


def _forecast_scope_label(forecast_type_filter: str) -> str:
    if forecast_type_filter == "full_disk":
        return "(full-disk)"
    if forecast_type_filter == "region":
        return "(region)"
    return "(full-disk + region)"


def _plot_yearly_tss_hss_on_axes(
    mdf: pd.DataFrame,
    ax0,
    ax1,
    *,
    year_min: int,
    year_max: int,
    forecast_type_filter: str,
    rc: dict,
    poster: bool,
    title: str,
    show_legend: bool,
    legend_ncol: int,
    hide_top_xticklabels: bool,
    threshold_color_idx: Optional[Dict[str, int]] = None,
    ftype_thr_color_idx: Optional[Dict[Tuple[str, str], int]] = None,
    multipanel_grid: bool = False,
) -> bool:
    """Draw TSS (ax0) and HSS (ax1) yearly lines. Return True if anything was plotted."""
    if forecast_type_filter == "all":
        group_keys: List[str] = ["forecast_type", "threshold"]
    else:
        group_keys = ["threshold"]

    groups = mdf.groupby(group_keys, sort=True)
    cmap = plt.get_cmap("tab10")
    lw = float(rc["lines.linewidth"])
    ms = float(rc["lines.markersize"])
    # Use slightly softer marker edges in the all-models grid.
    if multipanel_grid:
        mew = 0.32 if poster else 0.28
        marker_edge = "#666666"
    else:
        mew = 0.55 if poster else 0.45
        marker_edge = "#444444"

    plotted = False
    for i, (key, g) in enumerate(groups):
        g2 = g.drop_duplicates(subset=["year"]).sort_values("year")
        if g2["TSS"].notna().sum() == 0 and g2["HSS"].notna().sum() == 0:
            continue
        # Pick color from the global map so legend and all panels stay consistent.
        if ftype_thr_color_idx is not None and forecast_type_filter == "all":
            ftype, thr = key  # type: ignore[misc]
            pair = (str(ftype), str(thr))
            ci = ftype_thr_color_idx.get(pair, i)
        elif threshold_color_idx is not None and forecast_type_filter != "all":
            thr = _norm_threshold_key(key)
            ci = threshold_color_idx.get(thr, i)
        else:
            ci = i
        color = cmap(ci % 10)
        if forecast_type_filter == "all":
            ftype, thr = key  # type: ignore[misc]
            label = f"{ftype} ≥{thr}"
        else:
            thr = key[0] if isinstance(key, tuple) else key
            label = f"≥{thr}"
        yv, tss = _series_for_plot(g2, "TSS")
        _, hss = _series_for_plot(g2, "HSS")
        ax0.plot(
            yv,
            tss,
            marker="o",
            color=color,
            label=label,
            linewidth=lw,
            markersize=ms,
            markeredgewidth=mew,
            markeredgecolor=marker_edge,
        )
        ax1.plot(
            yv,
            hss,
            marker="o",
            color=color,
            label=label,
            linewidth=lw,
            markersize=ms,
            markeredgewidth=mew,
            markeredgecolor=marker_edge,
        )
        plotted = True

    if not plotted:
        return False

    leg_fs = rc["legend.fontsize"]
    ax0.set_ylabel("TSS", fontweight="bold")
    zlw = 0.5 if multipanel_grid else 0.65
    ax0.axhline(0.0, color="gray", linewidth=zlw, linestyle="--", alpha=0.55)
    ax0.set_title(title, fontweight="bold")
    if show_legend:
        ax0.legend(
            loc="best",
            ncol=legend_ncol,
            frameon=True,
            fancybox=False,
            edgecolor="#333333",
            framealpha=0.95,
            prop={"size": leg_fs, "weight": "bold"},
        )
    ax0.grid(True)
    ax0.tick_params(axis="both", which="major", labelsize=rc["ytick.labelsize"])
    if hide_top_xticklabels:
        ax0.tick_params(axis="x", which="major", labelbottom=False)
    _bold_tick_labels(ax0)

    ax1.set_ylabel("HSS", fontweight="bold")
    ax1.axhline(0.0, color="gray", linewidth=zlw, linestyle="--", alpha=0.55)
    ax1.set_xlabel("Year", fontweight="bold")
    ax1.set_xticks(list(range(year_min, year_max + 1)))
    ax1.grid(True)
    ax1.tick_params(axis="both", which="major", labelsize=rc["xtick.labelsize"])
    _bold_tick_labels(ax1)
    return True


def _model_has_yearly_curves(mdf: pd.DataFrame, forecast_type_filter: str) -> bool:
    if mdf.empty:
        return False
    if forecast_type_filter == "all":
        gk: List[str] = ["forecast_type", "threshold"]
    else:
        gk = ["threshold"]
    for _, g in mdf.groupby(gk, sort=True):
        g2 = g.drop_duplicates(subset=["year"]).sort_values("year")
        if g2["TSS"].notna().sum() > 0 or g2["HSS"].notna().sum() > 0:
            return True
    return False


def _build_combined_legend_handles(
    models: List[Tuple[str, pd.DataFrame]],
    forecast_type_filter: str,
    rc: dict,
    threshold_color_idx: Optional[Dict[str, int]],
    ftype_thr_color_idx: Optional[Dict[Tuple[str, str], int]],
) -> Tuple[List[Line2D], List[str]]:
    """Shared legend: full threshold set with stable colors (full_disk / region)."""
    cmap = plt.get_cmap("tab10")
    lw = rc["lines.linewidth"]
    ms = rc["lines.markersize"]
    handles: List[Line2D] = []
    labels: List[str] = []

    if forecast_type_filter == "all":
        assert ftype_thr_color_idx is not None
        ordered = sorted(
            ftype_thr_color_idx.keys(),
            key=lambda p: (p[0], _flare_threshold_rank(p[1])),
        )
        for ftype, thr in ordered:
            ci = ftype_thr_color_idx[(ftype, thr)]
            lab = f"{ftype} ≥{thr}"
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=cmap(ci % 10),
                    marker="o",
                    linestyle="-",
                    linewidth=lw,
                    markersize=ms,
                    markeredgewidth=0.32,
                    markeredgecolor="#666666",
                )
            )
            labels.append(lab)
        return handles, labels

    if not threshold_color_idx:
        return [], []
    ordered_thr = sorted(threshold_color_idx.keys(), key=lambda t: _flare_threshold_rank(t))
    for thr in ordered_thr:
        ci = threshold_color_idx[thr]
        handles.append(
            Line2D(
                [0],
                [0],
                color=cmap(ci % 10),
                marker="o",
                linestyle="-",
                linewidth=lw,
                markersize=ms,
                markeredgewidth=0.32,
                markeredgecolor="#666666",
            )
        )
        labels.append(f"≥{thr}")
    return handles, labels


def plot_all_models_grid(
    df: pd.DataFrame,
    year_min: int,
    year_max: int,
    out_path: Path,
    *,
    forecast_type_filter: str,
    poster: bool,
    dpi: int,
    ncols: int = 0,
) -> bool:
    """
    One figure: rectangular grid of panels (TSS + HSS per model), single shared legend.
    """
    mdf_all = df[(df["year"] >= year_min) & (df["year"] <= year_max)].copy()
    if forecast_type_filter != "all":
        mdf_all = mdf_all[
            mdf_all["forecast_type"].astype(str) == forecast_type_filter
        ].copy()

    models: List[Tuple[str, pd.DataFrame]] = []
    for model_name, mdf in mdf_all.groupby("model_name", sort=True):
        if _model_has_yearly_curves(mdf, forecast_type_filter):
            models.append((str(model_name), mdf))

    if not models:
        return False

    n = len(models)
    # Auto layout: near-square grid unless user forces column count.
    ncols_eff = ncols if ncols > 0 else max(2, int(np.ceil(np.sqrt(n))))
    nrows = int(np.ceil(n / ncols_eff))

    rc = _plot_style_rc_grid(poster)
    if poster:
        fw, fh = 4.2 * ncols_eff, 3.5 * nrows
    else:
        fw, fh = 3.6 * ncols_eff, 3.0 * nrows

    scope = _forecast_scope_label(forecast_type_filter)
    year_span = f"{year_min}–{year_max}"
    thr_colors = _global_threshold_color_map(models, forecast_type_filter)
    pair_colors = _global_ftype_thr_color_map(models, forecast_type_filter)

    with plt.rc_context(rc):
        fig = plt.figure(figsize=(fw, fh))
        outer = fig.add_gridspec(
            nrows,
            ncols_eff,
            hspace=0.42,
            wspace=0.30,
            left=0.07,
            right=0.98,
            top=0.90,
            bottom=0.14,
        )

        for idx, (model_name, mdf) in enumerate(models):
            r, c = divmod(idx, ncols_eff)
            inner = outer[r, c].subgridspec(2, 1, hspace=0.07)
            ax0 = fig.add_subplot(inner[0, 0])
            ax1 = fig.add_subplot(inner[1, 0], sharex=ax0)
            title = model_name.replace("_", " ")
            _plot_yearly_tss_hss_on_axes(
                mdf,
                ax0,
                ax1,
                year_min=year_min,
                year_max=year_max,
                forecast_type_filter=forecast_type_filter,
                rc=rc,
                poster=poster,
                title=title,
                show_legend=False,
                legend_ncol=1,
                hide_top_xticklabels=True,
                threshold_color_idx=thr_colors,
                ftype_thr_color_idx=pair_colors,
                multipanel_grid=True,
            )

        # Hide leftover cells if grid has more slots than models.
        for idx in range(n, nrows * ncols_eff):
            r, c = divmod(idx, ncols_eff)
            inner = outer[r, c].subgridspec(1, 1)
            ax_empty = fig.add_subplot(inner[0, 0])
            ax_empty.set_visible(False)

        leg_handles, leg_labels = _build_combined_legend_handles(
            models, forecast_type_filter, rc, thr_colors, pair_colors
        )
        ncol_leg = min(6, max(1, len(leg_labels)))
        fig.legend(
            leg_handles,
            leg_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=ncol_leg,
            frameon=True,
            fancybox=False,
            edgecolor="#333333",
            framealpha=0.95,
            prop={
                "size": max(8, rc["legend.fontsize"] + 1),
                "weight": "bold",
            },
        )

        fig.suptitle(
            f"Yearly TSS & HSS {scope}, {year_span}",
            fontweight="bold",
            fontsize=max(11, rc["axes.titlesize"] + 2),
            y=0.98,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
    return True


def _load_yearly_frame(
    combined_csv: Optional[str],
    eval_root: Path,
) -> pd.DataFrame:
    if combined_csv and os.path.isfile(combined_csv):
        df = pd.read_csv(combined_csv)
    else:
        paths = sorted(eval_root.glob("*/*_yearly_scores.csv"))
        if not paths:
            raise FileNotFoundError(
                f"No yearly scores found. Expected {eval_root}/*/*_yearly_scores.csv "
                "or pass --combined-csv to all_models_yearly_scores.csv"
            )
        df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)

    df = df[df["evaluation_mode"].astype(str) == "yearly"].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["TSS"] = pd.to_numeric(df["TSS"], errors="coerce")
    df["HSS"] = pd.to_numeric(df["HSS"], errors="coerce")
    return df


def _series_for_plot(
    sub: pd.DataFrame,
    metric: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (years, values) with NaNs kept so matplotlib breaks lines appropriately."""
    g = sub.sort_values("year")
    y = g["year"].astype(int).to_numpy()
    v = g[metric].to_numpy(dtype=float)
    return y, v


def plot_one_model(
    model_df: pd.DataFrame,
    year_min: int,
    year_max: int,
    out_path: Path,
    *,
    forecast_type_filter: str,
    poster: bool = False,
    dpi: int = 200,
) -> bool:
    """Return True if a figure was written.

    forecast_type_filter: "full_disk" | "region" | "all"
    """
    mdf = model_df[
        (model_df["year"] >= year_min) & (model_df["year"] <= year_max)
    ].copy()
    if forecast_type_filter != "all":
        mdf = mdf[mdf["forecast_type"].astype(str) == forecast_type_filter].copy()
    if mdf.empty:
        return False

    rc = _plot_style_rc(poster)
    w, h = (14, 9) if poster else (11, 7.5)
    model_name = mdf["model_name"].iloc[0]

    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 1, figsize=(w, h), sharex=True, constrained_layout=True)
        scope = _forecast_scope_label(forecast_type_filter)
        year_span = f"{year_min}–{year_max}"
        leg_ncol = 1 if poster else 2
        title = f"{model_name}: Yearly model trends {scope}, {year_span}"
        ok = _plot_yearly_tss_hss_on_axes(
            mdf,
            axes[0],
            axes[1],
            year_min=year_min,
            year_max=year_max,
            forecast_type_filter=forecast_type_filter,
            rc=rc,
            poster=poster,
            title=title,
            show_legend=True,
            legend_ncol=leg_ncol,
            hide_top_xticklabels=True,
        )
        if not ok:
            plt.close(fig)
            return False

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
    return True


def write_summary_csv(
    df: pd.DataFrame,
    year_min: int,
    year_max: int,
    out_csv: Path,
    *,
    forecast_type_filter: str = "all",
) -> None:
    """First vs last calendar year with finite TSS/HSS (within year range) per series."""
    rows: List[dict] = []
    mdf = df[
        (df["evaluation_mode"] == "yearly")
        & (df["year"] >= year_min)
        & (df["year"] <= year_max)
    ]
    if forecast_type_filter != "all":
        mdf = mdf[mdf["forecast_type"].astype(str) == forecast_type_filter].copy()
    for (model, ftype, thr), g in mdf.groupby(
        ["model_name", "forecast_type", "threshold"], sort=False
    ):
        g = g.sort_values("year")
        row: dict = {
            "model_name": model,
            "forecast_type": ftype,
            "threshold": thr,
        }
        for metric in ("TSS", "HSS"):
            s = g[["year", metric]].dropna(subset=[metric])
            if len(s) < 2:
                continue
            y_a, v_a = int(s.iloc[0]["year"]), float(s.iloc[0][metric])
            y_b, v_b = int(s.iloc[-1]["year"]), float(s.iloc[-1][metric])
            row[f"{metric}_year_first"] = y_a
            row[f"{metric}_year_last"] = y_b
            row[f"{metric}_first"] = v_a
            row[f"{metric}_last"] = v_b
            row[f"delta_{metric}"] = v_b - v_a
        if len(row) > 3:
            rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(out_csv, index=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Per-model yearly TSS/HSS line plots.")
    p.add_argument(
        "--eval-root",
        type=str,
        default="evaluation_results",
        help="Folder containing <MODEL>/<MODEL>_yearly_scores.csv",
    )
    p.add_argument(
        "--combined-csv",
        type=str,
        default="",
        help="Optional: evaluation_results/all_models_yearly_scores.csv",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="evaluation_results/figures/per_model_yearly_trends",
        help="Output directory for PNG files",
    )
    p.add_argument("--year-min", type=int, default=2020)
    p.add_argument("--year-max", type=int, default=2025)
    p.add_argument(
        "--forecast-type",
        type=str,
        choices=("full_disk", "region", "all", "both"),
        default="both",
        help="Which forecast stream to plot (default: both = full_disk + region separately)",
    )
    p.add_argument(
        "--summary-csv",
        type=str,
        default="",
        help="If set, write first-vs-last-year deltas to this CSV path",
    )
    p.add_argument(
        "--poster",
        action="store_true",
        help="Larger fonts and figure (for posters/slides)",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure resolution (default: 200)",
    )
    p.add_argument(
        "--combined-grid",
        action="store_true",
        help="Write one PNG with all models in a rectangular grid (TSS+HSS per model)",
    )
    p.add_argument(
        "--no-per-model",
        action="store_true",
        help="Skip individual model figures (use with --combined-grid)",
    )
    p.add_argument(
        "--grid-ncols",
        type=int,
        default=0,
        help="Columns in combined grid (0 = auto, about sqrt of model count)",
    )
    p.add_argument(
        "--grid-output",
        type=str,
        default="",
        help="Output path for combined grid PNG (default: <out-dir>/all_models_yearly_grid_...)",
    )
    args = p.parse_args()

    if args.no_per_model and not args.combined_grid:
        p.error("--no-per-model requires --combined-grid")

    eval_root = Path(args.eval_root)
    out_dir = Path(args.out_dir)
    combined = args.combined_csv.strip() or None

    df = _load_yearly_frame(combined, eval_root)
    y0, y1 = args.year_min, args.year_max
    ft = args.forecast_type
    ft_list = ["full_disk", "region"] if ft == "both" else [ft]

    written = 0
    if not args.no_per_model:
        for ft_i in ft_list:
            ft_tag = ft_i if ft_i != "all" else "alltypes"
            for model_name, mdf in df.groupby("model_name", sort=True):
                safe = str(model_name).replace("/", "_")
                out_path = out_dir / f"{safe}_yearly_tss_hss_{y0}_{y1}_{ft_tag}.png"
                if plot_one_model(
                    mdf,
                    y0,
                    y1,
                    out_path,
                    forecast_type_filter=ft_i,
                    poster=args.poster,
                    dpi=args.dpi,
                ):
                    print("Wrote", out_path)
                    written += 1
                else:
                    print(f"Skip ({ft_i}, no plottable yearly metrics):", model_name)

        print(f"Done. {written} per-model figure(s) in {out_dir.resolve()}")

    if args.combined_grid:
        for ft_i in ft_list:
            ft_tag = ft_i if ft_i != "all" else "alltypes"
            if args.grid_output.strip():
                base = Path(args.grid_output.strip())
                if len(ft_list) == 1:
                    grid_path = base
                else:
                    grid_path = base.with_name(f"{base.stem}_{ft_tag}{base.suffix or '.png'}")
            else:
                grid_path = out_dir / f"all_models_yearly_grid_{y0}_{y1}_{ft_tag}.png"

            if plot_all_models_grid(
                df,
                y0,
                y1,
                grid_path,
                forecast_type_filter=ft_i,
                poster=args.poster,
                dpi=args.dpi,
                ncols=args.grid_ncols,
            ):
                print("Wrote combined grid:", grid_path.resolve())
            else:
                print(f"Combined grid ({ft_i}): no plottable models.")

    if args.summary_csv:
        base_sc = Path(args.summary_csv)
        for ft_i in ft_list:
            if len(ft_list) == 1:
                sc = base_sc
            else:
                sc = base_sc.with_name(f"{base_sc.stem}_{ft_i}{base_sc.suffix or '.csv'}")
            write_summary_csv(df, y0, y1, sc, forecast_type_filter=ft_i)
            print("Summary:", sc.resolve())


if __name__ == "__main__":
    main()
