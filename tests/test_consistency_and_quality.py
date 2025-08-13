import polars as pl
import datetime as dt


def calc_factor(df: pl.DataFrame, lag: int) -> pl.DataFrame:
    # 简单的 ts_mean(close, 3) - ts_mean(close, 5) 替代实现，用于质量测试
    df = df.sort(["symbol", "date"]).with_columns([
        pl.col("close").rolling_mean(window_size=3, min_samples=3).over("symbol").alias("ma3"),
        pl.col("close").rolling_mean(window_size=5, min_samples=5).over("symbol").alias("ma5"),
    ])
    out = (
        df.with_columns((pl.col("ma3") - pl.col("ma5")).alias("raw"))
          .with_columns(pl.col("raw").shift(lag).over("symbol").alias("value"))
          .select(["date", "symbol", "value"]).drop_nulls("value")
    )
    return out


def test_full_vs_incremental_and_chunk_consistency():
    dates = [dt.date(2024, 2, d) for d in range(1, 21)]
    symbols = ["AAA", "BBB", "CCC"]
    rows = []
    for date in dates:
        for sym in symbols:
            rows.append({"date": date, "symbol": sym, "close": 100 + date.day + (0 if sym == "AAA" else 1)})
    df = pl.DataFrame(rows)

    lag, lookback = 1, 5

    # 全量
    full = calc_factor(df, lag)

    # 增量：仅用最后 5 天，底层数据扩窗 lookback
    inc_window = pl.concat([
        df.filter(pl.col("date") >= (dates[-5 - lookback])),
    ])
    inc = calc_factor(inc_window, lag).filter(pl.col("date") >= dates[-5])

    assert full.filter(pl.col("date").is_in(dates[-5:])).sort(["date", "symbol"]).equals(
        inc.sort(["date", "symbol"]) 
    )

    # 按日期块计算（4 天一块，扩窗 lookback）
    chunked_parts = []
    for i in range(0, len(dates), 4):
        chunk_start = dates[i]
        chunk_end = dates[min(i+3, len(dates)-1)]
        base = df.filter(pl.col("date") >= (chunk_start - dt.timedelta(days=lookback))).filter(pl.col("date") <= chunk_end)
        part = calc_factor(base, lag).filter((pl.col("date") >= chunk_start) & (pl.col("date") <= chunk_end))
        chunked_parts.append(part)
    chunked = pl.concat(chunked_parts).sort(["date", "symbol"]) 

    assert full.sort(["date", "symbol"]).equals(chunked)


def test_missing_values_and_alignment_rules():
    # 插入缺失，检查 drop_nulls 后一致
    dates = [dt.date(2024, 3, d) for d in range(1, 16)]
    rows = []
    for date in dates:
        rows.append({"date": date, "symbol": "AAA", "close": (None if date.day in (3, 4) else 100 + date.day)})
    df = pl.DataFrame(rows)
    out = calc_factor(df, lag=1)
    # 应无 NaN，且日期 >= 第5天才开始有值（五日均线）+ lag
    assert out.select(pl.col("value").is_null().any()).item() is False
    min_date = out.get_column("date").min()
    assert min_date is not None
    assert min_date >= dates[6]  # 5日均线的第5日是 d=5，再 shift 1 → d>=6