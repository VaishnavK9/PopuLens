"""Clean and reshape the World Bank population data.

Cleaning decisions (each is applied by a function below):

1. **Metadata rows / empty columns.** The 4 header lines are skipped in the
   loader and the empty ``Unnamed: N`` columns are dropped. ``Indicator Name``
   and ``Indicator Code`` hold one constant value, so they are dropped too.
2. **Aggregates removed.** The file mixes 217 real economies with 48
   aggregates ("World", "Euro area", "High income", "IDA total", ...). In the
   World Bank metadata only real economies have a ``Region``, so we keep the
   rows whose code has a non-null Region. ``INX`` ("Not classified") is not
   in the metadata at all and has no data, so the same rule removes it.
3. **Wide to long.** Year columns become rows: one row per (country, year).
4. **Missing population values are not imputed.** Only West Bank and Gaza has
   gaps (1960–1989, before the series begins). Inventing population counts
   would be misleading, so those country-years are dropped from the long
   table. Every country has a value for the latest year.
5. **Missing IncomeGroup.** Ethiopia and Venezuela are temporarily
   unclassified by the World Bank. They are kept and labelled
   ``"Not classified"`` so their population still counts in regional totals.
6. **Types.** ``year`` and ``population`` are stored as integers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.data_loader import PROCESSED_DIR, RawData, load_raw_data

UNCLASSIFIED_INCOME = "Not classified"

# Natural order for the income categories (used for sorting and plotting).
INCOME_ORDER: list[str] = [
    "Low income",
    "Lower middle income",
    "Upper middle income",
    "High income",
    UNCLASSIFIED_INCOME,
]

PROCESSED_FILENAME = "population_long.csv"

LONG_COLUMNS = ["country", "code", "region", "income_group", "year", "population"]


@dataclass
class CleaningReport:
    """Records what the cleaning pipeline removed or changed."""

    raw_rows: int = 0
    aggregates_removed: list[str] = field(default_factory=list)
    codes_missing_metadata: list[str] = field(default_factory=list)
    countries_kept: int = 0
    income_filled: list[str] = field(default_factory=list)
    missing_values_dropped: dict[str, int] = field(default_factory=dict)
    long_rows: int = 0
    year_range: tuple[int, int] = (0, 0)

    def summary(self) -> str:
        """Return a human-readable summary of the cleaning steps."""
        gaps = ", ".join(f"{k} ({v} yrs)" for k, v in self.missing_values_dropped.items())
        return "\n".join(
            [
                f"Raw rows in population file : {self.raw_rows}",
                f"Aggregate rows removed      : {len(self.aggregates_removed)}",
                f"Codes without metadata      : {', '.join(self.codes_missing_metadata) or 'none'}",
                f"Real countries kept         : {self.countries_kept}",
                f"IncomeGroup set to '{UNCLASSIFIED_INCOME}': {', '.join(self.income_filled) or 'none'}",
                f"Missing country-years dropped: {gaps or 'none'}",
                f"Long-format rows            : {self.long_rows:,}",
                f"Year range                  : {self.year_range[0]}–{self.year_range[1]}",
            ]
        )


def get_year_columns(df: pd.DataFrame) -> list[str]:
    """Return the columns whose names are 4-digit years."""
    return [c for c in df.columns if str(c).isdigit() and len(str(c)) == 4]


def filter_real_countries(
    population: pd.DataFrame, country_meta: pd.DataFrame, report: CleaningReport
) -> pd.DataFrame:
    """Keep only real economies and attach their Region and IncomeGroup.

    A row is a real economy when its code appears in the country metadata
    with a non-null ``Region``. Aggregates have an empty Region.
    """
    meta = country_meta[["Country Code", "Region", "IncomeGroup"]]
    real_codes = meta.loc[meta["Region"].notna(), "Country Code"]

    known_codes = set(meta["Country Code"])
    report.codes_missing_metadata = sorted(set(population["Country Code"]) - known_codes)
    is_aggregate = population["Country Code"].isin(known_codes) & ~population[
        "Country Code"
    ].isin(real_codes)
    report.aggregates_removed = population.loc[is_aggregate, "Country Name"].tolist()

    merged = population.merge(meta, on="Country Code", how="inner")
    merged = merged[merged["Region"].notna()].copy()
    report.countries_kept = len(merged)
    return merged


def fill_income_group(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Label countries without an IncomeGroup as ``"Not classified"``."""
    missing = df["IncomeGroup"].isna()
    report.income_filled = df.loc[missing, "Country Name"].tolist()
    df = df.copy()
    df["IncomeGroup"] = df["IncomeGroup"].fillna(UNCLASSIFIED_INCOME)
    return df


def to_long_format(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Melt year columns into rows and drop country-years with no value."""
    years = get_year_columns(df)
    long = df.melt(
        id_vars=["Country Name", "Country Code", "Region", "IncomeGroup"],
        value_vars=years,
        var_name="year",
        value_name="population",
    ).rename(
        columns={
            "Country Name": "country",
            "Country Code": "code",
            "Region": "region",
            "IncomeGroup": "income_group",
        }
    )

    missing = long["population"].isna()
    report.missing_values_dropped = long.loc[missing, "country"].value_counts().to_dict()
    long = long.loc[~missing].copy()

    long["year"] = long["year"].astype("int64")
    long["population"] = long["population"].round().astype("int64")
    long = long.sort_values(["country", "year"]).reset_index(drop=True)
    return long[LONG_COLUMNS]


def clean_population_data(raw: RawData | None = None) -> tuple[pd.DataFrame, CleaningReport]:
    """Run the full cleaning pipeline.

    Args:
        raw: Pre-loaded raw tables. Loaded from ``data/raw`` when omitted.

    Returns:
        The tidy long table (country, code, region, income_group, year,
        population) and a :class:`CleaningReport`.
    """
    raw = raw or load_raw_data()
    report = CleaningReport(raw_rows=len(raw.population))

    df = raw.population.drop(columns=["Indicator Name", "Indicator Code"], errors="ignore")
    df = filter_real_countries(df, raw.country_meta, report)
    df = fill_income_group(df, report)
    long = to_long_format(df, report)

    report.long_rows = len(long)
    report.year_range = (int(long["year"].min()), int(long["year"].max()))
    return long, report


def latest_year(df: pd.DataFrame) -> int:
    """Return the most recent year in a long table."""
    return int(df["year"].max())


def snapshot(df: pd.DataFrame, year: int | None = None) -> pd.DataFrame:
    """Return one row per country for a single year (latest year by default)."""
    year = latest_year(df) if year is None else year
    return df.loc[df["year"] == year].reset_index(drop=True)


def save_processed(df: pd.DataFrame, directory: Path = PROCESSED_DIR) -> Path:
    """Write the cleaned long table to ``data/processed`` and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / PROCESSED_FILENAME
    df.to_csv(path, index=False)
    return path


def load_processed(directory: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Load the cleaned table, rebuilding it from raw data if it does not exist."""
    path = directory / PROCESSED_FILENAME
    if not path.exists():
        long, _ = clean_population_data()
        save_processed(long, directory)
        return long
    return pd.read_csv(path, dtype={"year": "int64", "population": "int64"})


if __name__ == "__main__":
    tidy, cleaning_report = clean_population_data()
    out = save_processed(tidy)
    print(cleaning_report.summary())
    print(f"\nSaved {out.relative_to(PROCESSED_DIR.parents[1])}")
