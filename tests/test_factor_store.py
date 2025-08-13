import polars as pl
import datetime as dt

from store.factor_store import FactorStore


def test_store_idempotent_write_and_overwrite(tmp_path):
    store = FactorStore()
    df1 = pl.DataFrame({
        "date": [dt.date(2024, 1, 1), dt.date(2024, 1, 1), dt.date(2024, 1, 2)],
        "symbol": ["AAA", "AAA", "AAA"],
        "value": [1.0, 1.0, 2.0],
    })
    store.write("f", df1)
    out1 = store.read("f")
    assert out1.shape == (2, 3)

    # 覆盖写入
    df2 = pl.DataFrame({
        "date": [dt.date(2024, 1, 1), dt.date(2024, 1, 2)],
        "symbol": ["AAA", "AAA"],
        "value": [10.0, 20.0],
    })
    store.overwrite("f", df2)
    out2 = store.read("f")
    assert out2.sort(["date"]).to_dict(as_series=False)["value"] == [10.0, 20.0]

    # 追加写入，后一条覆盖前一条
    df3 = pl.DataFrame({
        "date": [dt.date(2024, 1, 2)],
        "symbol": ["AAA"],
        "value": [30.0],
    })
    store.write("f", df3)
    out3 = store.read("f")
    assert out3.filter(pl.col("date") == dt.date(2024, 1, 2))["value"].item() == 30.0