import polars as pl
import datetime as dt


from label.label_engine import make_forward_return


def test_forward_return_long_table():
    dates = [dt.date(2024, 1, d) for d in range(1, 8)]
    rows = []
    for d in dates:
        rows.append({"date": d, "symbol": "AAA", "close": 100 + d.day})
        rows.append({"date": d, "symbol": "BBB", "close": 200 + 2 * d.day})
    df = pl.DataFrame(rows)

    out = make_forward_return(df, horizon=2)
    assert out.columns == ["date", "symbol", "value"]
    # 有效样本应减少 horizon 天
    assert out.filter(pl.col("symbol") == "AAA").height == len(dates) - 2

    # 校验某一个具体值：AAA 在 d=1 的 2日收益
    v = (
        out.filter((pl.col("symbol") == "AAA") & (pl.col("date") == dt.date(2024, 1, 1)))
        ["value"].item()
    )
    # (price(t+2)/price(t) - 1)
    assert abs(v - ((100 + 3) / (100 + 1) - 1.0)) < 1e-9