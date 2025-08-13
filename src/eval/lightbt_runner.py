from typing import Dict, List
import polars as pl
import pandas as pd
import numpy as np
from lightbt import LightBT
from lightbt.enums import SizeType
from numba import njit


@njit
def _commission_zero(is_buy: bool, is_open: bool, fill_price: float, qty: float, amount: float) -> float:
    return 0.0


class LightBTRunner:
    def __init__(self, init_cash: float = 1_000_000.0) -> None:
        self.init_cash = init_cash

    def _to_weights(self, factor_df: pl.DataFrame, quantiles: int = 5) -> pl.DataFrame:
        df = (
            factor_df
            .with_columns([
                pl.len().over("date").alias("n"),
                pl.col("value").rank("ordinal").over("date").alias("rank"),
            ])
            .with_columns(((pl.col("rank") * quantiles / pl.col("n")).ceil().clip(1, quantiles).cast(pl.Int32)).alias("q"))
        )
        df = df.with_columns(
            pl.when(pl.col("q") == 1)
            .then(-1.0)
            .when(pl.col("q") == quantiles)
            .then(1.0)
            .otherwise(0.0)
            .alias("w")
        )
        return df.select(["date", "symbol", "w"]).rename({"symbol": "asset"})

    def _bars_struct(self, df: pl.DataFrame, str2int: dict) -> List[np.ndarray]:
        # 将列转换为 numpy 结构化数组，按日期分组
        recs = []
        for date, sub in df.group_by("date"):
            a = sub.select(
                [
                    pl.col("date").cast(pl.Datetime("ns")).cast(pl.UInt64).alias("date"),
                    pl.lit(int(SizeType.TargetAmount)).cast(pl.UInt32).alias("size_type"),
                    pl.col("asset").cast(pl.Utf8).alias("asset"),
                    pl.col("size").cast(pl.Float32).alias("size"),
                    pl.col("fill_price").cast(pl.Float32).alias("fill_price"),
                    pl.col("last_price").cast(pl.Float32).alias("last_price"),
                    pl.lit(0.0).cast(pl.Float32).alias("commission"),
                    pl.col("date_diff").cast(pl.Boolean).alias("date_diff"),
                ]
            ).to_pandas()
            # asset 字符串映射为内部整型ID
            a["asset"] = a["asset"].map(lambda s: int(str2int(s))).astype("uint32")
            arr = np.zeros(len(a), dtype=[
                ("date", np.uint64), ("size_type", np.uint32), ("asset", np.uint32),
                ("size", np.float32), ("fill_price", np.float32), ("last_price", np.float32),
                ("commission", np.float32), ("date_diff", np.bool_),
            ])
            arr["date"] = a["date"].values
            arr["size_type"] = a["size_type"].values
            arr["asset"] = a["asset"].values
            arr["size"] = a["size"].values
            arr["fill_price"] = a["fill_price"].values
            arr["last_price"] = a["last_price"].values
            arr["commission"] = a["commission"].values
            arr["date_diff"] = a["date_diff"].values
            recs.append(arr)
        return recs

    def _bars_from_weights(self, weights: pl.DataFrame, prices: pl.DataFrame, str2int: dict) -> List[np.ndarray]:
        df = weights.join(prices.rename({"symbol": "asset"}), on=["date", "asset"], how="inner")
        df = df.with_columns(pl.col("w").abs().sum().over("date").alias("tw"))
        df = df.with_columns(
            pl.when(pl.col("tw") > 1e-12).then(pl.col("w") / pl.col("tw")).otherwise(0.0).alias("target")
        )
        df = df.select(
            [
                pl.col("date"), pl.col("asset"),
                (pl.col("target") * 100).round(0).cast(pl.Float32).alias("size"),
                pl.col("close").alias("fill_price"),
                pl.col("close").alias("last_price"),
                pl.lit(True).alias("date_diff"),
            ]
        )
        return self._bars_struct(df, str2int)

    def run(self, factor_df: pl.DataFrame, prices_df: pl.DataFrame) -> Dict:
        weights = self._to_weights(factor_df)
        assets = weights.select("asset").unique().to_series().to_list()
        conf = pd.DataFrame({
            "asset": assets,
            "mult": 1.0,
            "margin_ratio": 1.0,
            "commission_ratio": 0.0,
            "commission_fn": [_commission_zero] * len(assets),
        })
        bt = LightBT(init_cash=self.init_cash)
        bt.setup(conf)
        bars = self._bars_from_weights(weights, prices_df, bt.asset_str2int)
        bt.run_bars(bars)
        perf = bt.trades_stats()
        return {"ret_annual": float(perf.get("ret_annual", 0.0)) if isinstance(perf, dict) else 0.0}