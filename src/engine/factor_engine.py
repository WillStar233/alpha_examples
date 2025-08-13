import io
import datetime as dt
from dataclasses import dataclass
from typing import Callable, Iterable, List, Tuple

import polars as pl
from expr_codegen import codegen_exec

from data.adapter import DataAdapter
from store.factor_store import FactorStore


@dataclass
class FactorSpec:
    name: str
    freq: str
    inputs: List[str]
    blocks: List[Callable]
    output_var: str
    lookback: int = 20
    lag: int = 1


class FactorEngine:
    def __init__(self, store: FactorStore, data_adapter: DataAdapter):
        self.store = store
        self.data = data_adapter

    def run_expr_codegen(self, df: pl.DataFrame, blocks: List[Callable], output_var: str, lag: int) -> Tuple[pl.DataFrame, str]:
        buf = io.StringIO()
        # 兼容 expr_codegen 期望的列名：asset/date
        needs_rename = "asset" not in df.columns and "symbol" in df.columns
        df_in = df.rename({"symbol": "asset"}) if needs_rename else df
        out = codegen_exec(
            df_in, *blocks, output_file=buf, over_null="partition_by", suppress_prefix=True, style="polars"
        )
        gen_code = buf.getvalue()
        if needs_rename and "asset" in out.columns and "symbol" not in out.columns:
            out = out.rename({"asset": "symbol"})
        out = out.with_columns(pl.col(output_var).shift(lag).over("symbol").alias("value"))
        out = out.select(["date", "symbol", "value"]).drop_nulls("value")
        return out, gen_code

    def compute_full(self, spec: FactorSpec, universe: List[str], start: dt.date, end: dt.date) -> pl.DataFrame:
        df = self.data.fetch(universe, start, end, spec.inputs, spec.freq)
        out, _ = self.run_expr_codegen(df, spec.blocks, spec.output_var, spec.lag)
        self.store.overwrite(spec.name, out)
        return out

    def compute_incremental(self, spec: FactorSpec, universe: List[str], new_dates: List[dt.date]) -> pl.DataFrame:
        d0, d1 = min(new_dates), max(new_dates)
        lb_start = d0 - dt.timedelta(days=spec.lookback + 2)
        df = self.data.fetch(universe, lb_start, d1, spec.inputs, spec.freq)
        out, _ = self.run_expr_codegen(df, spec.blocks, spec.output_var, spec.lag)
        out = out.filter((pl.col("date") >= d0) & (pl.col("date") <= d1))
        self.store.write(spec.name, out)
        return out

    def compute_full_by_date(self, spec: FactorSpec, universe: List[str], start: dt.date, end: dt.date, chunk_days: int) -> pl.DataFrame:
        outs: List[pl.DataFrame] = []
        for chunk in self.data.iter_by_date(universe, start, end, spec.inputs, spec.freq, chunk_days):
            chunk_start = chunk["date"].min()
            chunk_end = chunk["date"].max()
            lb_start = chunk_start - dt.timedelta(days=spec.lookback + 2)
            base = self.data.fetch(universe, lb_start, chunk_end, spec.inputs, spec.freq)
            out, _ = self.run_expr_codegen(base, spec.blocks, spec.output_var, spec.lag)
            out = out.filter((pl.col("date") >= chunk_start) & (pl.col("date") <= chunk_end))
            outs.append(out)
        merged = pl.concat(outs).sort(["date", "symbol"]) if outs else pl.DataFrame()
        self.store.overwrite(spec.name, merged)
        return merged

    def compute_full_by_code(self, spec: FactorSpec, universe: List[str], start: dt.date, end: dt.date, batch_size: int) -> pl.DataFrame:
        outs: List[pl.DataFrame] = []
        for panel in self.data.iter_by_code(universe, start, end, spec.inputs, spec.freq, batch_size):
            out, _ = self.run_expr_codegen(panel, spec.blocks, spec.output_var, spec.lag)
            outs.append(out)
        merged = pl.concat(outs).sort(["date", "symbol"]) if outs else pl.DataFrame()
        self.store.overwrite(spec.name, merged)
        return merged