import datetime as dt
from typing import Callable, Iterable, List, Optional

import polars as pl


class DataAdapter:
    def __init__(
        self,
        get_data: Optional[Callable] = None,
        get_data_chunk_by_code: Optional[Callable] = None,
        get_data_chunk_by_date: Optional[Callable] = None,
    ) -> None:
        self.get_data = get_data
        self.get_data_chunk_by_code = get_data_chunk_by_code
        self.get_data_chunk_by_date = get_data_chunk_by_date

    @staticmethod
    def _to_polars(df) -> pl.DataFrame:
        if isinstance(df, pl.DataFrame):
            return df
        try:
            import pandas as pd  # type: ignore

            if isinstance(df, pd.DataFrame):
                return pl.from_pandas(df)
        except Exception:
            pass
        raise TypeError("DataAdapter expects pandas or polars DataFrame from get_data functions")

    def fetch(
        self,
        universe: List[str],
        start: dt.date,
        end: dt.date,
        fields: List[str],
        freq: str,
    ) -> pl.DataFrame:
        if self.get_data is None:
            raise RuntimeError("get_data is not provided for DataAdapter.fetch")
        df = self.get_data(symbols=universe, start=start, end=end, freq=freq, fields=fields)
        df_pl = self._to_polars(df)
        expected = ["date", "symbol", *fields]
        missing = [c for c in expected if c not in df_pl.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return df_pl.select(expected)

    def iter_by_code(
        self,
        universe: List[str],
        start: dt.date,
        end: dt.date,
        fields: List[str],
        freq: str,
        batch_size: int,
    ) -> Iterable[pl.DataFrame]:
        if self.get_data_chunk_by_code is None:
            raise RuntimeError("get_data_chunk_by_code is not provided for DataAdapter.iter_by_code")
        for i in range(0, len(universe), batch_size):
            batch = universe[i : i + batch_size]
            df = self.get_data_chunk_by_code(symbols=batch, start=start, end=end, freq=freq, fields=fields)
            yield self._to_polars(df).select(["date", "symbol", *fields])

    def iter_by_date(
        self,
        universe: List[str],
        start: dt.date,
        end: dt.date,
        fields: List[str],
        freq: str,
        chunk_days: int,
    ) -> Iterable[pl.DataFrame]:
        if self.get_data_chunk_by_date is None:
            raise RuntimeError("get_data_chunk_by_date is not provided for DataAdapter.iter_by_date")
        cur = start
        while cur <= end:
            chunk_end = min(end, cur + dt.timedelta(days=chunk_days - 1))
            df = self.get_data_chunk_by_date(symbols=universe, start=cur, end=chunk_end, freq=freq, fields=fields)
            yield self._to_polars(df).select(["date", "symbol", *fields])
            cur = chunk_end + dt.timedelta(days=1)