"""Table helpers: a single reader/writer for TSV, CSV and Parquet."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

__all__ = ["read_table", "write_table", "concat_tables"]


def _suffix(path: str | Path) -> str:
    return Path(path).suffix.lower()


def read_table(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read a DataFrame, dispatching on file extension."""
    suffix = _suffix(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path, **kwargs)
    if suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    if suffix in {".tsv", ".txt", ".tab", ""}:
        return pd.read_csv(path, sep="\t", **kwargs)
    raise ValueError(f"unsupported table extension: {suffix!r}")


def write_table(df: pd.DataFrame, path: str | Path, index: bool = False, **kwargs) -> Path:
    """Write a DataFrame, creating parent directories and dispatching on suffix."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = _suffix(out)
    if suffix in {".parquet", ".pq"}:
        df.to_parquet(out, index=index, **kwargs)
    elif suffix == ".csv":
        df.to_csv(out, index=index, **kwargs)
    elif suffix in {".tsv", ".txt", ".tab", ""}:
        df.to_csv(out, sep="\t", index=index, **kwargs)
    else:
        raise ValueError(f"unsupported table extension: {suffix!r}")
    return out


def concat_tables(paths: Iterable[str | Path], **kwargs) -> pd.DataFrame:
    """Concatenate several tables of the same format."""
    frames = [read_table(path, **kwargs) for path in paths]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
