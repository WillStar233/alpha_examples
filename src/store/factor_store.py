import datetime as dt
import polars as pl
from typing import Optional


def _empty_frame() -> pl.DataFrame:
    return pl.DataFrame(schema={"date": pl.Date, "symbol": pl.String, "value": pl.Float64})


class FactorStore:
    def __init__(self):
        self._frames: dict[str, pl.DataFrame] = {}

    def write(self, factor_name: str, df: pl.DataFrame) -> None:
        if factor_name not in self._frames:
            self._frames[factor_name] = _empty_frame()
        merged = pl.concat([self._frames[factor_name], df], how="vertical_relaxed")
        self._frames[factor_name] = (
            merged.sort(["date", "symbol"]).unique(subset=["date", "symbol"], keep="last")
        )

    def overwrite(self, factor_name: str, df: pl.DataFrame) -> None:
        if df.is_empty():
            self._frames[factor_name] = _empty_frame()
        else:
            self._frames[factor_name] = df.sort(["date", "symbol"]).unique(
                subset=["date", "symbol"], keep="last"
            )

    def read(self, factor_name: str, start: Optional[dt.date] = None, end: Optional[dt.date] = None) -> pl.DataFrame:
        df = self._frames.get(factor_name, _empty_frame())
        if start is not None:
            df = df.filter(pl.col("date") >= start)
        if end is not None:
            df = df.filter(pl.col("date") <= end)
        return df

    # 预留：磁盘 Parquet 写入/读取接口（未在当前测试中使用）
    def write_parquet_partitioned(self, base_dir: str, factor_name: str) -> None:
        df = self._frames.get(factor_name)
        if df is None:
            return
        # TODO: 实现分区写入 factor_name/date=YYYYMMDD/*.parquet
        raise NotImplementedError