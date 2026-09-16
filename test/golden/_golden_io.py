"""Array <-> DataFrame helpers shared by generate_baseline.py and test_golden.py.

Fixtures store their own input data so the golden tests never depend on how the
synthetic series were generated. Indexes are stored as int64 nanoseconds (UTC)
together with the original resolution unit and timezone, because RTide's sample
rate inference depends on the index unit (ns vs us).
"""
import numpy as np
import pandas as pd


def frame_to_arrays(df: pd.DataFrame, prefix: str) -> dict:
    index = pd.DatetimeIndex(df.index)
    return {
        f"{prefix}_values": df.to_numpy(dtype=np.float64),
        f"{prefix}_columns": np.asarray([str(c) for c in df.columns], dtype="U"),
        f"{prefix}_dtypes": np.asarray([str(t) for t in df.dtypes], dtype="U"),
        f"{prefix}_index_ns": np.asarray(index.as_unit("ns").asi8, dtype=np.int64),
        f"{prefix}_index_unit": np.asarray(index.unit, dtype="U"),
        f"{prefix}_index_tz": np.asarray("" if index.tz is None else str(index.tz), dtype="U"),
    }


def index_from_arrays(z, prefix: str) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(np.asarray(z[f"{prefix}_index_ns"], dtype=np.int64).astype("datetime64[ns]"))
    tz = str(z[f"{prefix}_index_tz"])
    if tz:
        index = index.tz_localize("UTC").tz_convert(tz)
    return index.as_unit(str(z[f"{prefix}_index_unit"]))


def frame_from_arrays(z, prefix: str) -> pd.DataFrame:
    columns = [str(c) for c in z[f"{prefix}_columns"]]
    return pd.DataFrame(
        np.asarray(z[f"{prefix}_values"], dtype=np.float64),
        index=index_from_arrays(z, prefix),
        columns=columns,
    )
