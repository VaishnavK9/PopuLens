"""Charts for PopuLens.

Every function takes the tidy long table produced by :mod:`src.cleaning`,
returns a figure, and optionally saves it as a PNG in ``reports/figures``.
The same functions power the notebook and the Streamlit app.

Styling rules applied throughout:

* One shared look (``apply_style``): recessive grid, no top/right spines,
  left-aligned titles, a source footnote on every chart.
* Color follows the entity, not its rank: each Region always gets the same
  hue (``REGION_COLORS``), in every chart.
* Magnitude and ordered categories use a single blue ramp; the choropleth is
  sequential blue on a log scale because populations span 5 orders of
  magnitude.
* Populations are printed in readable units (12K, 250M, 1.4B).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from src.cleaning import INCOME_ORDER, UNCLASSIFIED_INCOME, latest_year, snapshot
from src.data_loader import FIGURES_DIR

# --------------------------------------------------------------------------- #
# Palette & style
# --------------------------------------------------------------------------- #
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8983"
GRID = "#e4e3df"
NEUTRAL = "#b7b5ad"

PRIMARY = "#2a78d6"  # single-series blue
# Blue ramp (light -> dark) for ordered categories and magnitude.
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
SEQUENTIAL_SCALE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# Fixed categorical order (validated for colour-vision deficiency, adjacent pairs).
REGION_ORDER: list[str] = [
    "East Asia & Pacific",
    "South Asia",
    "Sub-Saharan Africa",
    "Europe & Central Asia",
    "Latin America & Caribbean",
    "Middle East, North Africa, Afghanistan & Pakistan",
    "North America",
]
_CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
REGION_COLORS: dict[str, str] = dict(zip(REGION_ORDER, _CATEGORICAL))

REGION_SHORT: dict[str, str] = {
    "Middle East, North Africa, Afghanistan & Pakistan": "Middle East, N. Africa,\nAfghanistan & Pakistan",
}
REGION_SHORT_ONELINE: dict[str, str] = {
    "Middle East, North Africa, Afghanistan & Pakistan": "MENA, Afghanistan & Pakistan",
}

INCOME_COLORS: dict[str, str] = {
    "Low income": BLUE_RAMP[0],
    "Lower middle income": BLUE_RAMP[1],
    "Upper middle income": BLUE_RAMP[3],
    "High income": BLUE_RAMP[4],
    UNCLASSIFIED_INCOME: NEUTRAL,
}

SOURCE_NOTE = "Source: World Bank, World Development Indicators (SP.POP.TOTL)"


def apply_style() -> None:
    """Apply the shared PopuLens matplotlib/seaborn style."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.labelsize": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "font.size": 10,
            "figure.dpi": 110,
            "savefig.dpi": 150,
        }
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def human_format(value: float, decimals: int = 1) -> str:
    """Format a number with K/M/B suffixes, e.g. 1_412_000_000 -> '1.41B'.

    Billions get two decimals so that India (1.45B) and China (1.41B) stay
    distinguishable; trailing zeros are stripped (2.0B -> '2B').
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    sign = "-" if value < 0 else ""
    value = abs(float(value))
    for threshold, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if value >= threshold:
            places = max(decimals, 2) if suffix == "B" else decimals
            text = f"{value / threshold:.{places}f}".rstrip("0").rstrip(".")
            return f"{sign}{text}{suffix}"
    return f"{sign}{value:.0f}"


HUMAN_FORMATTER = FuncFormatter(lambda x, _pos: human_format(x))


def _region_label(region: str, oneline: bool = False) -> str:
    mapping = REGION_SHORT_ONELINE if oneline else REGION_SHORT
    return mapping.get(region, region)


def _add_source(fig: Figure, extra: str = "") -> None:
    note = SOURCE_NOTE + (f" · {extra}" if extra else "")
    fig.text(0.01, 0.005, note, ha="left", va="bottom", fontsize=8, color=TEXT_MUTED)


def save_figure(fig: Figure, name: str, directory: Path = FIGURES_DIR) -> Path:
    """Save a matplotlib figure as ``<directory>/<name>.png`` and return the path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    return path


def _maybe_save(fig: Figure, name: str, save: bool, directory: Path) -> None:
    if save:
        save_figure(fig, name, directory)


def filter_regions(df: pd.DataFrame, regions: list[str] | None) -> pd.DataFrame:
    """Keep only the given regions (all regions when ``regions`` is empty/None)."""
    if not regions:
        return df
    return df[df["region"].isin(regions)]


# --------------------------------------------------------------------------- #
# 1. Histogram: distribution of country populations (linear vs log)
# --------------------------------------------------------------------------- #
def plot_population_histogram(
    df: pd.DataFrame,
    year: int | None = None,
    regions: list[str] | None = None,
    save: bool = False,
    directory: Path = FIGURES_DIR,
) -> Figure:
    """Histogram of country populations for one year, linear and log scale side by side."""
    apply_style()
    year = latest_year(df) if year is None else year
    snap = filter_regions(snapshot(df, year), regions)
    pop = snap["population"].astype(float)
    median = float(pop.median())

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(13, 5))

    sns.histplot(x=pop, bins=40, ax=ax_lin, color=PRIMARY, edgecolor=SURFACE, linewidth=0.5, alpha=1)
    ax_lin.set_title("Linear scale: a few giants, a long tail")
    ax_lin.set_xlabel("Population")
    ax_lin.set_ylabel("Number of countries")
    ax_lin.xaxis.set_major_formatter(HUMAN_FORMATTER)
    ax_lin.grid(axis="x", visible=False)
    giants = snap.nlargest(2, "population")
    # Label the two largest together only when they share a bar (e.g. India & China).
    if len(giants) == 2 and giants["population"].iloc[1] < 0.9 * giants["population"].iloc[0]:
        giants = giants.head(1)
    giant_x = float(giants["population"].mean())
    y_top = ax_lin.get_ylim()[1]
    ax_lin.annotate(
        " & ".join(giants["country"]) + "\n(" + " / ".join(giants["population"].map(human_format)) + ")",
        xy=(giant_x, y_top * 0.01), xytext=(giant_x, y_top * 0.17), ha="center", fontsize=9,
        color=TEXT_SECONDARY, arrowprops={"arrowstyle": "-", "color": TEXT_MUTED, "lw": 0.8},
    )

    sns.histplot(x=pop, bins=30, log_scale=True, ax=ax_log, color=PRIMARY, edgecolor=SURFACE, linewidth=1.2, alpha=1)
    ax_log.set_title("Log scale: the shape of the distribution")
    ax_log.set_xlabel("Population (log scale)")
    ax_log.set_ylabel("Number of countries")
    ax_log.xaxis.set_major_formatter(HUMAN_FORMATTER)
    ax_log.grid(axis="x", visible=False)
    ax_log.axvline(median, color=TEXT_PRIMARY, lw=1.2, ls="--")
    ax_log.text(
        median / 1.2, ax_log.get_ylim()[1] * 0.95, f"Median country: {human_format(median)}",
        ha="right", va="top", fontsize=9.5, color=TEXT_PRIMARY,
    )

    fig.suptitle(
        f"How population is distributed across {len(snap)} countries ({year})",
        x=0.01, ha="left", fontsize=15, fontweight="bold", color=TEXT_PRIMARY,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    _add_source(fig)
    _maybe_save(fig, "01_population_histogram", save, directory)
    return fig


# --------------------------------------------------------------------------- #
# 2. Bar chart: top N most populous countries
# --------------------------------------------------------------------------- #
def plot_top_countries(
    df: pd.DataFrame,
    year: int | None = None,
    n: int = 15,
    regions: list[str] | None = None,
    save: bool = False,
    directory: Path = FIGURES_DIR,
) -> Figure:
    """Horizontal bar chart of the ``n`` most populous countries, coloured by region."""
    apply_style()
    year = latest_year(df) if year is None else year
    snap = filter_regions(snapshot(df, year), regions)
    top = snap.nlargest(n, "population").iloc[::-1]
    world_total = snapshot(df, year)["population"].sum()

    fig, ax = plt.subplots(figsize=(11, 0.42 * len(top) + 1.8))
    colors = [REGION_COLORS.get(r, NEUTRAL) for r in top["region"]]
    ax.barh(top["country"], top["population"], color=colors, height=0.72, edgecolor=SURFACE, linewidth=1.5)

    x_max = float(top["population"].max())
    for y, value in enumerate(top["population"]):
        share = value / world_total * 100
        ax.text(value + x_max * 0.01, y, f"{human_format(value)}  ({share:.1f}%)",
                va="center", fontsize=9, color=TEXT_SECONDARY)

    ax.set_xlim(0, x_max * 1.18)
    ax.xaxis.set_major_formatter(HUMAN_FORMATTER)
    ax.set_xlabel("Population")
    ax.set_ylabel("")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    scope = "selected regions" if regions else "the world"
    ax.set_title(f"Top {len(top)} most populous countries in {scope} ({year})\n", fontsize=14)
    ax.text(0, 1.02, "Bar labels show population and share of world population",
            transform=ax.transAxes, fontsize=9.5, color=TEXT_SECONDARY)

    present = [r for r in REGION_ORDER if r in set(top["region"])]
    handles = [plt.Rectangle((0, 0), 1, 1, color=REGION_COLORS[r]) for r in present]
    ax.legend(handles, [_region_label(r, oneline=True) for r in present],
              title="Region", loc="lower right", title_fontsize=9.5)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _add_source(fig)
    _maybe_save(fig, "02_top15_countries", save, directory)
    return fig


# --------------------------------------------------------------------------- #
# 3. Bar chart: total population by region
# --------------------------------------------------------------------------- #
def region_totals(df: pd.DataFrame, year: int | None = None) -> pd.DataFrame:
    """Total population, country count and world share per region for one year."""
    snap = snapshot(df, year)
    out = (
        snap.groupby("region")
        .agg(population=("population", "sum"), countries=("country", "count"))
        .sort_values("population", ascending=False)
    )
    out["share_pct"] = out["population"] / out["population"].sum() * 100
    return out.reset_index()


def plot_population_by_region(
    df: pd.DataFrame,
    year: int | None = None,
    regions: list[str] | None = None,
    save: bool = False,
    directory: Path = FIGURES_DIR,
) -> Figure:
    """Horizontal bar chart of total population per World Bank region."""
    apply_style()
    year = latest_year(df) if year is None else year
    totals = region_totals(df, year)
    if regions:
        totals = totals[totals["region"].isin(regions)]
    totals = totals.iloc[::-1]

    fig, ax = plt.subplots(figsize=(11, 0.6 * len(totals) + 1.8))
    labels = [_region_label(r) for r in totals["region"]]
    ax.barh(labels, totals["population"], color=[REGION_COLORS[r] for r in totals["region"]],
            height=0.68, edgecolor=SURFACE, linewidth=1.5)

    x_max = float(totals["population"].max())
    for y, (value, share, count) in enumerate(totals[["population", "share_pct", "countries"]].itertuples(index=False)):
        ax.text(value + x_max * 0.01, y, f"{human_format(value)}  ·  {share:.1f}%  ·  {count} countries",
                va="center", fontsize=9, color=TEXT_SECONDARY)

    ax.set_xlim(0, x_max * 1.35)
    ax.xaxis.set_major_formatter(HUMAN_FORMATTER)
    ax.set_xlabel("Total population")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(f"Total population by region ({year})\n", fontsize=14)
    ax.text(0, 1.02, "Labels: population · share of world · number of countries",
            transform=ax.transAxes, fontsize=9.5, color=TEXT_SECONDARY)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _add_source(fig)
    _maybe_save(fig, "03_population_by_region", save, directory)
    return fig


# --------------------------------------------------------------------------- #
# 4. Bar charts: country count and population by income group
# --------------------------------------------------------------------------- #
def income_totals(df: pd.DataFrame, year: int | None = None, regions: list[str] | None = None) -> pd.DataFrame:
    """Country count and total population per income group, in income order."""
    snap = filter_regions(snapshot(df, year), regions)
    out = snap.groupby("income_group").agg(
        countries=("country", "count"), population=("population", "sum")
    )
    order = [g for g in INCOME_ORDER if g in out.index]
    out = out.loc[order]
    out["country_pct"] = out["countries"] / out["countries"].sum() * 100
    out["population_pct"] = out["population"] / out["population"].sum() * 100
    return out.reset_index()


def plot_income_groups(
    df: pd.DataFrame,
    year: int | None = None,
    regions: list[str] | None = None,
    save: bool = False,
    directory: Path = FIGURES_DIR,
) -> Figure:
    """Two bar charts: how many countries vs how many people fall in each income group."""
    apply_style()
    year = latest_year(df) if year is None else year
    totals = income_totals(df, year, regions)
    labels = [g.replace(" income", "").replace("Lower middle", "Lower-\nmiddle")
              .replace("Upper middle", "Upper-\nmiddle").replace("Not classified", "Not\nclassified")
              for g in totals["income_group"]]
    colors = [INCOME_COLORS[g] for g in totals["income_group"]]

    fig, (ax_n, ax_p) = plt.subplots(1, 2, figsize=(13, 5.2))

    ax_n.bar(labels, totals["countries"], color=colors, width=0.7, edgecolor=SURFACE, linewidth=1.5)
    for x, (count, pct) in enumerate(totals[["countries", "country_pct"]].itertuples(index=False)):
        ax_n.text(x, count, f"{count}\n({pct:.0f}%)", ha="center", va="bottom", fontsize=9, color=TEXT_SECONDARY)
    ax_n.set_title("Number of countries")
    ax_n.set_ylabel("Countries")
    ax_n.set_ylim(0, totals["countries"].max() * 1.22)

    ax_p.bar(labels, totals["population"], color=colors, width=0.7, edgecolor=SURFACE, linewidth=1.5)
    for x, (pop, pct) in enumerate(totals[["population", "population_pct"]].itertuples(index=False)):
        ax_p.text(x, pop, f"{human_format(pop)}\n({pct:.0f}%)", ha="center", va="bottom", fontsize=9, color=TEXT_SECONDARY)
    ax_p.set_title("Total population")
    ax_p.set_ylabel("Population")
    ax_p.yaxis.set_major_formatter(HUMAN_FORMATTER)
    ax_p.set_ylim(0, totals["population"].max() * 1.22)

    for ax in (ax_n, ax_p):
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", length=0)
        ax.set_xlabel("World Bank income group (low → high)")

    fig.suptitle(f"Countries vs people by income group ({year})",
                 x=0.01, ha="left", fontsize=15, fontweight="bold", color=TEXT_PRIMARY)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    _add_source(fig)
    _maybe_save(fig, "04_income_groups", save, directory)
    return fig


# --------------------------------------------------------------------------- #
# 5. Line chart: population growth of the top 10 countries
# --------------------------------------------------------------------------- #
def top_country_series(
    df: pd.DataFrame, n: int = 10, year: int | None = None, regions: list[str] | None = None
) -> pd.DataFrame:
    """Full time series for the ``n`` most populous countries in ``year``."""
    top = filter_regions(snapshot(df, year), regions).nlargest(n, "population")["country"]
    return df[df["country"].isin(top)]


def plot_population_growth(
    df: pd.DataFrame,
    n: int = 10,
    year: int | None = None,
    regions: list[str] | None = None,
    save: bool = False,
    directory: Path = FIGURES_DIR,
) -> Figure:
    """Population over time for the top ``n`` countries: absolute and indexed (first year = 100).

    Lines are coloured by region (consistent with the other charts) and
    labelled directly at their end, so identity never relies on colour alone.
    """
    apply_style()
    year = latest_year(df) if year is None else year
    series = top_country_series(df, n, year, regions)
    series = series[series["year"] <= year]
    order = (series[series["year"] == year].sort_values("population", ascending=False)["country"].tolist())
    first_year = int(series["year"].min())

    fig, (ax_abs, ax_idx) = plt.subplots(1, 2, figsize=(15, 6.2))

    end_abs: list[tuple[float, str, str]] = []
    end_idx: list[tuple[float, str, str]] = []
    for country in order:
        s = series[series["country"] == country].sort_values("year")
        color = REGION_COLORS.get(s["region"].iloc[0], NEUTRAL)
        ax_abs.plot(s["year"], s["population"], color=color, lw=2)
        base = s["population"].iloc[0]
        indexed = s["population"] / base * 100
        ax_idx.plot(s["year"], indexed, color=color, lw=2)
        end_abs.append((float(s["population"].iloc[-1]), country, color))
        end_idx.append((float(indexed.iloc[-1]), country, color))

    _direct_labels(ax_abs, end_abs, year, fmt=human_format)
    _direct_labels(ax_idx, end_idx, year, fmt=lambda v: f"{v / 100:.1f}×")

    ax_abs.set_title("Population")
    ax_abs.yaxis.set_major_formatter(HUMAN_FORMATTER)
    ax_abs.set_ylabel("Population")
    ax_idx.set_title(f"Growth since {first_year} (index, {first_year} = 100)")
    ax_idx.set_ylabel(f"Index ({first_year} = 100)")
    ax_idx.axhline(100, color=TEXT_MUTED, lw=1, ls="--")
    for ax in (ax_abs, ax_idx):
        ax.set_xlabel("Year")
        ax.set_xlim(first_year, year + (year - first_year) * 0.28)
        ax.set_xticks([t for t in range(first_year, year + 1, 10)] if year - first_year >= 20
                      else list(range(first_year, year + 1)))
        ax.grid(axis="x", visible=False)

    present = [r for r in REGION_ORDER if r in set(series["region"])]
    handles = [plt.Line2D([0], [0], color=REGION_COLORS[r], lw=2.5) for r in present]
    fig.legend(handles, [_region_label(r, oneline=True) for r in present], title="Region",
               loc="lower center", ncol=len(present), bbox_to_anchor=(0.5, 0.035), title_fontsize=9.5)

    fig.suptitle(f"Population growth of the {len(order)} most populous countries, {first_year}–{year}",
                 x=0.01, ha="left", fontsize=15, fontweight="bold", color=TEXT_PRIMARY)
    fig.tight_layout(rect=(0, 0.11, 1, 0.97))
    _add_source(fig)
    _maybe_save(fig, "05_population_growth_top10", save, directory)
    return fig


def _direct_labels(ax: plt.Axes, ends: list[tuple[float, str, str]], x: float, fmt) -> None:
    """Place end-of-line labels, nudging them apart so they never overlap."""
    ends = sorted(ends, key=lambda t: t[0])
    ymin, ymax = ax.get_ylim()
    min_gap = (ymax - ymin) * 0.038
    placed = [value for value, _name, _color in ends]
    # Relax: push overlapping neighbours apart symmetrically until none overlap.
    for _ in range(500):
        moved = False
        for i in range(len(placed) - 1):
            overlap = min_gap - (placed[i + 1] - placed[i])
            if overlap > 1e-9:
                placed[i] -= overlap / 2
                placed[i + 1] += overlap / 2
                moved = True
        if placed:
            placed[0] = max(placed[0], ymin + min_gap / 2)
        if not moved:
            break
    for (value, name, color), y in zip(ends, placed):
        ax.plot([x, x + 1.2], [value, y], color=color, lw=0.8)
        ax.text(x + 1.6, y, f"{name}  {fmt(value)}", va="center", fontsize=8.8, color=TEXT_PRIMARY)


# --------------------------------------------------------------------------- #
# 6. Bonus: interactive choropleth world map (Plotly)
# --------------------------------------------------------------------------- #
def plot_choropleth(
    df: pd.DataFrame,
    year: int | None = None,
    regions: list[str] | None = None,
    save: bool = False,
    directory: Path = FIGURES_DIR,
) -> go.Figure:
    """Interactive world map of population (log colour scale, hover for exact values).

    Saves ``06_world_population_map.html`` and, when a static image engine
    (kaleido + Chrome) is available, a PNG with the same name.
    """
    year = latest_year(df) if year is None else year
    snap = filter_regions(snapshot(df, year), regions).copy()
    snap["log_pop"] = np.log10(snap["population"])
    snap["pop_label"] = snap["population"].map(human_format)
    snap["pop_exact"] = snap["population"].map("{:,}".format)

    ticks = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
    fig = px.choropleth(
        snap,
        locations="code",
        locationmode="ISO-3",
        color="log_pop",
        hover_name="country",
        custom_data=["pop_label", "pop_exact", "region", "income_group"],
        color_continuous_scale=SEQUENTIAL_SCALE,
        range_color=(4, 9.2),
    )
    fig.update_traces(
        marker_line_color="#ffffff",
        marker_line_width=0.4,
        hovertemplate=(
            "<b>%{hovertext}</b><br>Population: %{customdata[0]} (%{customdata[1]})"
            "<br>Region: %{customdata[2]}<br>Income: %{customdata[3]}<extra></extra>"
        ),
    )
    fig.update_layout(
        title={"text": f"<b>World population by country ({year})</b>"
               "<br><sup>Log colour scale · hover a country for exact figures</sup>", "x": 0.01},
        coloraxis_colorbar={
            "title": "Population",
            "tickvals": [np.log10(t) for t in ticks],
            "ticktext": [human_format(t) for t in ticks],
            "thickness": 14,
        },
        geo={"showframe": False, "showcoastlines": False, "projection_type": "natural earth",
             "bgcolor": SURFACE, "showcountries": True, "countrycolor": "#ffffff",
             "showland": True, "landcolor": "#efeeea"},
        paper_bgcolor=SURFACE,
        font={"family": "Helvetica, Arial, sans-serif", "color": TEXT_PRIMARY},
        margin={"l": 10, "r": 10, "t": 70, "b": 30},
        height=560,
        annotations=[{"text": SOURCE_NOTE, "x": 0.01, "y": -0.04, "xref": "paper", "yref": "paper",
                      "showarrow": False, "font": {"size": 10, "color": TEXT_MUTED}}],
    )

    if save:
        directory.mkdir(parents=True, exist_ok=True)
        fig.write_html(directory / "06_world_population_map.html", include_plotlyjs="cdn")
        try:
            fig.write_image(directory / "06_world_population_map.png", width=1400, height=760, scale=1.5)
        except Exception as exc:  # kaleido/Chrome are optional
            print(f"Skipped PNG export of the map ({type(exc).__name__}). The HTML version was saved.")
    return fig


# --------------------------------------------------------------------------- #
# Generate everything
# --------------------------------------------------------------------------- #
def generate_all_figures(df: pd.DataFrame, directory: Path = FIGURES_DIR) -> list[Path]:
    """Create and save every chart; return the list of files in ``directory``."""
    builders = [
        plot_population_histogram,
        plot_top_countries,
        plot_population_by_region,
        plot_income_groups,
        plot_population_growth,
    ]
    for build in builders:
        fig = build(df, save=True, directory=directory)
        plt.close(fig)
    plot_choropleth(df, save=True, directory=directory)
    return sorted(p for p in directory.iterdir() if p.suffix in {".png", ".html"})


if __name__ == "__main__":
    from src.cleaning import load_processed

    for file in generate_all_figures(load_processed()):
        print("saved", file.relative_to(FIGURES_DIR.parents[1]))
