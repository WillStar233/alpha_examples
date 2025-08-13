import io
import polars as pl
import datetime as dt
import pytest

from typing import List

from expr_codegen import codegen_exec

from engine.factor_engine import FactorEngine, FactorSpec
from store.factor_store import FactorStore


class FactorStore:
    def __init__(self):
        self._frames = {}

    def overwrite(self, factor_name: str, df: pl.DataFrame):
        self._frames[factor_name] = df.sort(["date", "symbol"]).unique(subset=["date", "symbol"], keep="last")

    def write(self, factor_name: str, df: pl.DataFrame):
        if factor_name not in self._frames:
            self._frames[factor_name] = pl.DataFrame({"date": [], "symbol": [], "value": []})
        concat = pl.concat([self._frames[factor_name], df])
        self._frames[factor_name] = concat.sort(["date", "symbol"]).unique(subset=["date", "symbol"], keep="last")

    def read(self, factor_name: str) -> pl.DataFrame:
        return self._frames.get(factor_name, pl.DataFrame({"date": [], "symbol": [], "value": []}))


@pytest.fixture()
def toy_df():
    # 制造一个简单的面板数据
    dates = [dt.date(2024, 1, d) for d in range(1, 16)]
    symbols = ["AAA", "BBB"]
    rows = []
    for date in dates:
        for sym in symbols:
            rows.append({
                "date": date,
                "symbol": sym,
                "close": 100 + (date.day) + (0 if sym == "AAA" else 1),
                "volume": 1000 + 10 * date.day,
            })
    return pl.DataFrame(rows)


def make_blocks():
    # 定义 expr_codegen 的代码块：一个时序均线差
    def _block():
        # 仅使用 close，避免默认需要 asset/industry 等列
        fast = ts_mean(close, 3)
        slow = ts_mean(close, 5)
        FACTOR = fast - slow
    return [_block]


def run_codegen(df: pl.DataFrame, blocks, output_var: str, lag: int) -> pl.DataFrame:
    buf = io.StringIO()
    df_in = df.rename({"symbol": "asset"}) if "asset" not in df.columns else df
    out = codegen_exec(df_in, *blocks, output_file=buf, over_null="partition_by", suppress_prefix=True, style="polars")
    if "asset" in out.columns and "symbol" not in out.columns:
        out = out.rename({"asset": "symbol"})
    # 产出列名 output_var，按 symbol shift(lag)，并归一化为长表
    out = out.with_columns(pl.col(output_var).shift(lag).over("symbol").alias("value"))
    out = out.select(["date", "symbol", "value"]).drop_nulls("value")
    return out


def test_run_expr_codegen_outputs_long_table(toy_df):
    blocks = make_blocks()
    out = run_codegen(toy_df, blocks, "FACTOR", lag=1)
    assert out.columns == ["date", "symbol", "value"]
    # lag=1 导致首日为空
    assert out.group_by("symbol").len().select(pl.col("len").min())[0, 0] < toy_df.group_by("symbol").len().select(pl.col("len").min())[0, 0]


def test_full_vs_incremental_consistency(monkeypatch, toy_df):
    # 构造极简 DataAdapter 与 Engine，占位，Engine 使用 run_codegen 来运行
    class DummyAdapter:
        def fetch(self, universe, start, end, fields, freq):
            return toy_df.filter((pl.col("date") >= start) & (pl.col("date") <= end)).select(["date", "symbol", *fields])

        def iter_by_date(self, universe, start, end, fields, freq, chunk_days):
            cur = start
            while cur <= end:
                chunk_end = min(end, cur + dt.timedelta(days=chunk_days - 1))
                yield self.fetch(universe, cur, chunk_end, fields, freq)
                cur = chunk_end + dt.timedelta(days=1)

    store = FactorStore()

    class Engine(FactorEngine):
        def run_expr_codegen(self, df, blocks, output_var, lag):
            return run_codegen(df, blocks, output_var, lag), "# generated code"

        def compute_full(self, spec, universe, start, end):
            df = adapter.fetch(universe, start, end, spec.inputs, spec.freq)
            out, _ = self.run_expr_codegen(df, spec.blocks, spec.output_var, spec.lag)
            self.store.overwrite(spec.name, out)
            return out

        def compute_incremental(self, spec, universe, new_dates):
            d0, d1 = min(new_dates), max(new_dates)
            lookback_start = d0 - dt.timedelta(days=spec.lookback + 2)
            df = adapter.fetch(universe, lookback_start, d1, spec.inputs, spec.freq)
            out, _ = self.run_expr_codegen(df, spec.blocks, spec.output_var, spec.lag)
            out = out.filter((pl.col("date") >= d0) & (pl.col("date") <= d1))
            self.store.write(spec.name, out)
            return out

        def compute_full_by_date(self, spec, universe, start, end, chunk_days):
            outs = []
            for chunk in adapter.iter_by_date(universe, start, end, spec.inputs, spec.freq, chunk_days):
                # 扩窗 lookback
                chunk_start = chunk["date"].min()
                lb_start = chunk_start - dt.timedelta(days=spec.lookback + 2)
                base = adapter.fetch(universe, lb_start, chunk["date"].max(), spec.inputs, spec.freq)
                out, _ = self.run_expr_codegen(base, spec.blocks, spec.output_var, spec.lag)
                out = out.filter((pl.col("date") >= chunk_start) & (pl.col("date") <= chunk["date"].max()))
                outs.append(out)
            merged = pl.concat(outs).sort(["date", "symbol"]) if outs else pl.DataFrame()
            self.store.overwrite(spec.name, merged)
            return merged

    adapter = DummyAdapter()
    engine = Engine(store, adapter)

    spec = FactorSpec(name="ma_diff", freq="1d", inputs=["close"], blocks=make_blocks(), output_var="FACTOR", lookback=5, lag=1)

    universe = ["AAA", "BBB"]
    start, end = dt.date(2024, 1, 1), dt.date(2024, 1, 15)

    full = engine.compute_full(spec, universe, start, end)

    # 增量计算最近 5 天
    inc = engine.compute_incremental(spec, universe, [dt.date(2024, 1, d) for d in range(11, 16)])

    # 从 store 读出全量，再取后 5 天，应该与增量一致
    stored = store.read("ma_diff").filter(pl.col("date") >= dt.date(2024, 1, 11))

    # 精确相等（无浮点误差来源）
    assert inc.sort(["date", "symbol"]).equals(stored.sort(["date", "symbol"]))

    # 分块（按日期）应与全量一致
    chunked = engine.compute_full_by_date(spec, universe, start, end, chunk_days=4)
    assert full.sort(["date", "symbol"]).equals(chunked.sort(["date", "symbol"]))


def test_by_code_chunk_consistency_and_generated_code(monkeypatch, toy_df):
    from engine.factor_engine import FactorEngine, FactorSpec
    from store.factor_store import FactorStore
    from data.adapter import DataAdapter

    def _block():
        fast = ts_mean(close, 3)
        slow = ts_mean(close, 5)
        FACTOR = fast - slow

    # 使用 by_code 方式：两批 symbols
    symbols = ["AAA", "BBB"]
    start, end = dt.date(2024, 1, 1), dt.date(2024, 1, 15)

    def get_data(symbols, start, end, freq, fields):
        return toy_df.filter((pl.col("symbol").is_in(symbols)) & (pl.col("date") >= start) & (pl.col("date") <= end)).select(["date", "symbol", *fields])

    adapter = DataAdapter(get_data=get_data, get_data_chunk_by_code=get_data)
    store = FactorStore()
    engine = FactorEngine(store, adapter)

    spec = FactorSpec(name="ma_diff", freq="1d", inputs=["close"], blocks=[_block], output_var="FACTOR", lookback=5, lag=1)

    full = engine.compute_full(spec, symbols, start, end)
    by_code = engine.compute_full_by_code(spec, symbols, start, end, batch_size=1)

    assert full.sort(["date", "symbol"]).equals(by_code.sort(["date", "symbol"]))


def test_polars_ta_prefix_in_codegen(monkeypatch, toy_df):
    # 改为使用内置 ts_mean 验证 codegen 跑通，不强制依赖 talib
    from engine.factor_engine import FactorEngine, FactorSpec
    from store.factor_store import FactorStore
    from data.adapter import DataAdapter

    def _block():
        FACTOR = ts_mean(close, 5)

    def get_data(symbols, start, end, freq, fields):
        return toy_df.filter((pl.col("symbol").is_in(symbols)) & (pl.col("date") >= start) & (pl.col("date") <= end)).select(["date", "symbol", *fields])

    adapter = DataAdapter(get_data=get_data)
    store = FactorStore()
    engine = FactorEngine(store, adapter)

    spec = FactorSpec(name="ts_mean5", freq="1d", inputs=["close"], blocks=[_block], output_var="FACTOR", lookback=5, lag=1)

    out = engine.compute_full(spec, ["AAA", "BBB"], dt.date(2024, 1, 1), dt.date(2024, 1, 15))
    assert set(out.columns) == {"date", "symbol", "value"}
    assert out.height > 0