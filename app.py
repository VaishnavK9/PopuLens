"""PopuLens – interactive Streamlit explorer for World Bank population data.

Run with:
    streamlit run app.py

Pick a year, one or more regions and a chart type in the sidebar; every chart
reuses the same functions as the analysis notebook (``src/visualize.py``).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.cleaning import load_processed, snapshot
from src.visualize import (
    REGION_ORDER,
    human_format,
    plot_choropleth,
    plot_income_groups,
    plot_population_by_region,
    plot_population_growth,
    plot_population_histogram,
    plot_top_countries,
)

st.set_page_config(page_title="PopuLens", page_icon="🌍", layout="wide")

CHART_TYPES: dict[str, str] = {
    "Histogram – population distribution": "histogram",
    "Bar – top countries": "top",
    "Bar – population by region": "region",
    "Bar – income groups": "income",
    "Line – growth of the largest countries": "growth",
    "Map – world choropleth": "map",
}

CHART_NOTES: dict[str, str] = {
    "histogram": "On a linear axis a few giants dominate. The log axis shows the real shape of the distribution.",
    "top": "Bars are coloured by region. Labels give population and share of world population.",
    "region": "Total population per World Bank region, with world share and number of countries.",
    "income": "The same income groups counted two ways: by number of countries and by number of people.",
    "growth": "Left: absolute population. Right: growth indexed to the first year (= 100), to compare rates.",
    "map": "Log colour scale. Hover over a country for exact figures; scroll to zoom.",
}


@st.cache_data(show_spinner="Loading World Bank data…")
def get_data() -> pd.DataFrame:
    """Load (and cache) the cleaned long-format population table."""
    return load_processed()


def show_matplotlib(fig: plt.Figure) -> None:
    """Render a matplotlib figure in Streamlit and free its memory."""
    st.pyplot(fig, width="stretch")
    plt.close(fig)


df = get_data()
min_year, max_year = int(df["year"].min()), int(df["year"].max())

# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🌍 PopuLens")
    st.caption("World Bank population, 1960–2024")

    year = st.slider("Year", min_value=min_year, max_value=max_year, value=max_year)
    regions = st.multiselect(
        "Regions", options=REGION_ORDER, default=[],
        placeholder="All regions",
        help="Leave empty to include every region.",
    )
    chart_label = st.radio("Chart type", list(CHART_TYPES))
    chart = CHART_TYPES[chart_label]

    if chart in {"top", "growth"}:
        n = st.slider("Number of countries", 5, 20, 15 if chart == "top" else 10)
    else:
        n = 10

    st.divider()
    st.caption(
        "Prodigy InfoTech Data Science Internship – Task 01  \n"
        "Built by [Vaishnav Kumbhar](https://github.com/VaishnavK9)"
    )

# --------------------------------------------------------------------------- #
# Headline numbers for the current selection
# --------------------------------------------------------------------------- #
snap = snapshot(df, year)
selected = snap[snap["region"].isin(regions)] if regions else snap

st.title("PopuLens: how the world's population is distributed")
scope = ", ".join(regions) if regions else "All regions"
st.caption(f"{scope} · {year}")

if selected.empty:
    st.warning("No countries match this selection. Try another year or region.")
    st.stop()

largest = selected.loc[selected["population"].idxmax()]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total population", human_format(selected["population"].sum()))
c2.metric("Countries", f"{len(selected)}")
c3.metric("Median country", human_format(selected["population"].median()))
c4.metric(f"Largest: {largest['country']}", human_format(largest["population"]))

if regions:
    share = selected["population"].sum() / snap["population"].sum()
    st.caption(f"The selected regions hold **{share:.1%}** of the world's population in {year}.")

# --------------------------------------------------------------------------- #
# Chart
# --------------------------------------------------------------------------- #
st.subheader(chart_label.split("–", 1)[1].strip().capitalize())
st.caption(CHART_NOTES[chart])

if chart == "histogram":
    show_matplotlib(plot_population_histogram(df, year, regions=regions))
elif chart == "top":
    show_matplotlib(plot_top_countries(df, year, n=n, regions=regions))
elif chart == "region":
    show_matplotlib(plot_population_by_region(df, year, regions=regions))
elif chart == "income":
    show_matplotlib(plot_income_groups(df, year, regions=regions))
elif chart == "growth":
    if year - min_year < 5:
        st.info(f"Pick a year after {min_year + 4} to see growth over time.")
    else:
        show_matplotlib(plot_population_growth(df, n=n, year=year, regions=regions))
else:
    st.plotly_chart(plot_choropleth(df, year, regions=regions), width="stretch")

# --------------------------------------------------------------------------- #
# Data table + download
# --------------------------------------------------------------------------- #
with st.expander("View the data behind this chart"):
    table = (
        selected.sort_values("population", ascending=False)
        .loc[:, ["country", "region", "income_group", "population"]]
        .reset_index(drop=True)
    )
    table.index += 1
    st.dataframe(
        table,
        width="stretch",
        column_config={"population": st.column_config.NumberColumn("population", format="localized")},
    )
    st.download_button(
        "Download CSV",
        table.to_csv(index_label="rank").encode("utf-8"),
        file_name=f"populens_{year}.csv",
        mime="text/csv",
    )

st.caption("Source: World Bank, World Development Indicators (SP.POP.TOTL). Aggregates such as 'World' are excluded.")
