import datetime as dt
import polars as pl

from data.adapter import DataAdapter
from store.factor_store import FactorStore
from engine.factor_engine import FactorEngine, FactorSpec
from orchestrator.factor_orchestrator import FactorOrchestrator


class DummyLogger:
    def __init__(self):
        self.params = []
        self.metrics = []
        self.artifacts = []
        self.started = False

    def start(self, run_name: str):
        self.started = True

    def log_params(self, params: dict):
        self.params.append(params)

    def log_metrics(self, metrics: dict):
        self.metrics.append(metrics)

    def log_artifact_text(self, name: str, content: str):
        self.artifacts.append((name, content[:20]))

    def log_artifact_file(self, path: str, artifact_name: str | None = None):
        self.artifacts.append((artifact_name or path, path))

    def end(self):
        self.started = False


class DummyAI:
    def run(self, factor_df: pl.DataFrame, label_df: pl.DataFrame, groups=None, masks=None) -> dict:
        return {"IC": 0.1}


class DummyBT:
    def run(self, factor_df: pl.DataFrame, rules: dict) -> dict:
        return {"ret_annual": 0.15}


def _make_toy_df():
    dates = [dt.date(2024, 1, d) for d in range(1, 11)]
    symbols = ["AAA", "BBB"]
    rows = []
    for date in dates:
        for sym in symbols:
            rows.append({"date": date, "symbol": sym, "close": 100 + date.day + (0 if sym == "AAA" else 1)})
    return pl.DataFrame(rows)


def test_orchestrator_full_pipeline(monkeypatch, tmp_path):
    toy = _make_toy_df()

    def get_data(symbols, start, end, freq, fields):
        return toy.filter((pl.col("symbol").is_in(symbols)) & (pl.col("date") >= start) & (pl.col("date") <= end)).select(["date", "symbol", *fields])

    adapter = DataAdapter(get_data=get_data)
    store = FactorStore()
    engine = FactorEngine(store, adapter)

    def _block():
        FACTOR = ts_mean(close, 3) - ts_mean(close, 5)

    spec = FactorSpec(name="demo", freq="1d", inputs=["close"], blocks=[_block], output_var="FACTOR", lookback=5, lag=1)

    logger = DummyLogger()
    ai = DummyAI()
    bt = DummyBT()

    orch = FactorOrchestrator(engine, logger=logger, ai_runner=ai, bt_runner=bt)

    out = orch.run_full(spec, ["AAA", "BBB"], dt.date(2024, 1, 1), dt.date(2024, 1, 10), label_horizon=2)

    assert out["factor_rows"] > 0
    assert logger.params and logger.metrics


def test_ts_block_guard_in_date_chunks(monkeypatch):
    toy = _make_toy_df()

    def get_data(symbols, start, end, freq, fields):
        return toy.filter((pl.col("symbol").is_in(symbols)) & (pl.col("date") >= start) & (pl.col("date") <= end)).select(["date", "symbol", *fields])

    adapter = DataAdapter(get_data=get_data)
    store = FactorStore()
    engine = FactorEngine(store, adapter)

    def _block():
        a = ts_mean(close, 3)
        FACTOR = a

    spec = FactorSpec(name="ts_guard", freq="1d", inputs=["close"], blocks=[_block], output_var="FACTOR", lookback=3, lag=0)

    import pytest

    with pytest.raises(ValueError):
        engine.compute_full_by_date(spec, ["AAA", "BBB"], dt.date(2024, 1, 1), dt.date(2024, 1, 10), chunk_days=3)