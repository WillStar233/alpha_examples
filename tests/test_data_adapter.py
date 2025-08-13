import polars as pl
import datetime as dt
import pytest

from typing import Iterable, List

from data.adapter import DataAdapter


@pytest.fixture()
def sample_data():
    dates = [dt.date(2024, 1, d) for d in range(1, 11)]
    symbols = ["AAA", "BBB", "CCC"]
    rows = []
    for date in dates:
        for sym in symbols:
            rows.append({
                "date": date,
                "symbol": sym,
                "close": float(hash((date, sym)) % 1000) / 10.0,
                "volume": (hash((sym, date)) % 10000) + 1,
            })
    return pl.DataFrame(rows)


def mock_get_data(symbols, start, end, freq, fields):
    raise NotImplementedError


def mock_get_data_chunk_by_code(symbols, start, end, freq, fields):
    raise NotImplementedError


def mock_get_data_chunk_by_date(symbols, start, end, freq, fields):
    raise NotImplementedError


def test_fetch_returns_polars_df_with_required_columns(sample_data, monkeypatch):
    # stub get_data 返回 polars
    def _get_data(symbols, start, end, freq, fields):
        df = sample_data.filter(pl.col("symbol").is_in(symbols)).filter(
            (pl.col("date") >= start) & (pl.col("date") <= end)
        ).select(["date", "symbol", *fields])
        return df

    adapter = DataAdapter(_get_data)
    start, end = dt.date(2024, 1, 3), dt.date(2024, 1, 7)
    out = adapter.fetch(["AAA", "BBB"], start, end, ["close", "volume"], "1d")
    assert isinstance(out, pl.DataFrame)
    assert set(["date", "symbol", "close", "volume"]).issubset(out.columns)
    assert out.select(["date"]).min()[0, 0] == start
    assert out.select(["date"]).max()[0, 0] == end
    assert set(out.select(["symbol"]).unique().to_series().to_list()) == {"AAA", "BBB"}


def test_iter_by_code_batches(sample_data, monkeypatch):
    def _get_data_chunk_by_code(symbols, start, end, freq, fields):
        df = sample_data.filter(pl.col("symbol").is_in(symbols)).filter(
            (pl.col("date") >= start) & (pl.col("date") <= end)
        ).select(["date", "symbol", *fields])
        return df

    adapter = DataAdapter(get_data=None, get_data_chunk_by_code=_get_data_chunk_by_code)
    start, end = dt.date(2024, 1, 1), dt.date(2024, 1, 10)

    batches = list(adapter.iter_by_code(["AAA", "BBB", "CCC"], start, end, ["close"], "1d", batch_size=2))
    assert len(batches) == 2
    assert all(isinstance(b, pl.DataFrame) for b in batches)
    assert all(set(["date", "symbol", "close"]).issubset(b.columns) for b in batches)
    # 每个 batch 的 symbol 覆盖数量合理
    assert set(batches[0]["symbol"].unique().to_list()).issubset({"AAA", "BBB", "CCC"})


def test_iter_by_date_chunks(sample_data, monkeypatch):
    def _get_data_chunk_by_date(symbols, start, end, freq, fields):
        df = sample_data.filter(pl.col("symbol").is_in(symbols)).filter(
            (pl.col("date") >= start) & (pl.col("date") <= end)
        ).select(["date", "symbol", *fields])
        return df

    adapter = DataAdapter(get_data=None, get_data_chunk_by_date=_get_data_chunk_by_date)
    start, end = dt.date(2024, 1, 1), dt.date(2024, 1, 10)

    chunks = list(adapter.iter_by_date(["AAA", "BBB", "CCC"], start, end, ["close"], "1d", chunk_days=3))
    assert len(chunks) == 4  # 10 天按 3 天切块
    assert all(isinstance(c, pl.DataFrame) for c in chunks)
    assert all(set(["date", "symbol", "close"]).issubset(c.columns) for c in chunks)
    # 每块的日期范围有效
    mins = [c["date"].min() for c in chunks]
    maxs = [c["date"].max() for c in chunks]
    assert mins[0] == start and maxs[-1] == end