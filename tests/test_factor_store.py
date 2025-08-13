import polars as pl
import datetime as dt

class FactorStore:
    def __init__(self):
        self._frames = {}

    def write(self, factor_name: str, df: pl.DataFrame):
        if factor_name not in self._frames:
            self._frames[factor_name] = pl.DataFrame({"date": [], "symbol": [], "value": []})
        self._frames[factor_name] = (
            pl.concat([self._frames[factor_name], df])
            .sort(["date", "symbol"]).unique(subset=["date", "symbol"], keep="last")
        )

    def overwrite(self, factor_name: str, df: pl.DataFrame):
        self._frames[factor_name] = df.sort(["date", "symbol"]).unique(subset=["date", "symbol"], keep="last")

    def read(self, factor_name: str, start: dt.date | None = None, end: dt.date | None = None):
        df = self._frames.get(factor_name, pl.DataFrame({"date": [], "symbol": [], "value": []}))
        if start is not None:
            df = df.filter(pl.col("date") >= start)
        if end is not None:
            df = df.filter(pl.col("date") <= end)
        return df


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