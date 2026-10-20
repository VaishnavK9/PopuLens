"""Load the raw World Bank population files.

The World Bank "API_SP.POP.TOTL" bundle ships three CSVs:

* ``API_SP.POP.TOTL_*.csv`` – population, one row per country/aggregate and
  one column per year. It starts with 4 metadata lines that must be skipped.
* ``Metadata_Country_*.csv`` – Region and IncomeGroup for each code.
* ``Metadata_Indicator_*.csv`` – a description of the indicator itself.

Every path is resolved relative to the project root, so the code works no
matter where the repository is cloned.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR: Path = PROJECT_ROOT / "reports" / "figures"

POPULATION_PATTERN = "API_SP.POP.TOTL_*.csv"
COUNTRY_META_PATTERN = "Metadata_Country_API_SP.POP.TOTL_*.csv"
INDICATOR_META_PATTERN = "Metadata_Indicator_API_SP.POP.TOTL_*.csv"

# Lines before the real header in the population CSV
# ("Data Source", blank, "Last Updated Date", blank).
METADATA_ROWS_TO_SKIP = 4


@dataclass
class RawData:
    """Container for the three raw World Bank tables."""

    population: pd.DataFrame
    country_meta: pd.DataFrame
    indicator_meta: pd.DataFrame


def find_file(pattern: str, directory: Path = RAW_DIR) -> Path:
    """Return the single file in ``directory`` matching a glob ``pattern``.

    Globbing keeps the loader working when the World Bank changes the
    version suffix of the file names (e.g. ``_v2_38144``).

    Raises:
        FileNotFoundError: if no file matches.
    """
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No file matching '{pattern}' in {directory}. "
            "Download the dataset from "
            "https://github.com/Prodigy-InfoTech/data-science-datasets/tree/main/Task%201 "
            "and place the CSVs in data/raw/."
        )
    return matches[-1]


def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the empty ``Unnamed: N`` columns created by trailing commas."""
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]


def load_population_wide(directory: Path = RAW_DIR) -> pd.DataFrame:
    """Load the population table in its original wide format (years as columns)."""
    path = find_file(POPULATION_PATTERN, directory)
    df = pd.read_csv(path, skiprows=METADATA_ROWS_TO_SKIP, encoding="utf-8-sig")
    return _drop_unnamed(df)


def load_country_metadata(directory: Path = RAW_DIR) -> pd.DataFrame:
    """Load country metadata (Region, IncomeGroup, SpecialNotes, TableName)."""
    path = find_file(COUNTRY_META_PATTERN, directory)
    return _drop_unnamed(pd.read_csv(path, encoding="utf-8-sig"))


def load_indicator_metadata(directory: Path = RAW_DIR) -> pd.DataFrame:
    """Load the one-row indicator description table."""
    path = find_file(INDICATOR_META_PATTERN, directory)
    return _drop_unnamed(pd.read_csv(path, encoding="utf-8-sig"))


def load_raw_data(directory: Path = RAW_DIR) -> RawData:
    """Load all three raw tables at once."""
    return RawData(
        population=load_population_wide(directory),
        country_meta=load_country_metadata(directory),
        indicator_meta=load_indicator_metadata(directory),
    )


def inspect_dataframe(df: pd.DataFrame, name: str, max_columns: int = 10) -> pd.DataFrame:
    """Print shape, columns, dtypes and null counts for a quick first look.

    Args:
        df: The table to inspect.
        name: Label printed in the header.
        max_columns: How many columns to list in full before truncating.

    Returns:
        A per-column summary table (dtype, non-null count, null count, null %).
    """
    print(f"{'=' * 70}\n{name}: {df.shape[0]:,} rows x {df.shape[1]:,} columns\n{'=' * 70}")
    cols = list(df.columns)
    shown = cols if len(cols) <= max_columns else cols[:6] + ["..."] + cols[-3:]
    print("Columns:", shown)
    print("Dtypes :", df.dtypes.astype(str).value_counts().to_dict())
    nulls = df.isna().sum()
    print(f"Nulls  : {int(nulls.sum()):,} total across {int((nulls > 0).sum())} column(s)\n")
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non_null": df.notna().sum(),
            "nulls": nulls,
            "null_pct": (nulls / len(df) * 100).round(1),
        }
    )


if __name__ == "__main__":
    raw = load_raw_data()
    inspect_dataframe(raw.population, "Population (wide)")
    inspect_dataframe(raw.country_meta, "Country metadata")
    inspect_dataframe(raw.indicator_meta, "Indicator metadata")
