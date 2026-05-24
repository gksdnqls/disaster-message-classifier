from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import LABEL_COLUMN, TEXT_COLUMN


def load_split(data_dir: str | Path, split: str) -> pd.DataFrame:
    path = Path(data_dir) / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")

    df = pd.read_csv(path)
    required = {TEXT_COLUMN, LABEL_COLUMN}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df = df[[TEXT_COLUMN, LABEL_COLUMN]].copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    return df


def load_dataset(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        load_split(data_dir, "train"),
        load_split(data_dir, "valid"),
        load_split(data_dir, "test"),
    )
