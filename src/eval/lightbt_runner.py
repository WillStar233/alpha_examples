from typing import Dict, List
import polars as pl
from lightbt import LightBT
from lightbt.enums import SizeType


class LightBTRunner:
    def __init__(self, init_cash: float = 1_000_000.0) -> None:
        self.init_cash = init_cash

    def _to_weights(self, factor_df: pl.DataFrame, quantiles: int = 5) -> pl.DataFrame:
        # 将因子横截面转为分位权重：顶/底分位 +1/-1，其余 0
        df = (
            factor_df.group_by("date")
            .agg([
                pl.col("symbol"),
                pl.col("value"),
                pl.col("value").rank("ordinal").alias("rank"),
                pl.len().alias("n"),
            ])
            .explode(["symbol", "value", "rank", "n"])
            .with_columns((pl.col("rank") * quantiles / pl.col("n")).ceil().clip(1, quantiles).cast(pl.Int32).alias("q"))
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

    def _bars_from_weights(self, weights: pl.DataFrame, prices: pl.DataFrame) -> pl.DataFrame:
        # 简化：每日根据权重生成持仓变动，以收盘价成交
        # 需将 date 转为 int64（ns）以适配 LightBT
        df = weights.join(prices.rename({"symbol": "asset"}), on=["date", "asset"], how="inner")
        # 目标头寸比例 = w / sum(|w|)，避免过度杠杆
        df = df.group_by("date").agg([
            pl.col("asset"),
            pl.col("w"),
            pl.col("close"),
            pl.col("w").abs().sum().alias("tw"),
        ]).explode(["asset", "w", "close", "tw"]).with_columns(
            (pl.when(pl.col("tw") > 0).then(pl.col("w") / pl.col("tw").clip_min(1e-12)).otherwise(0.0)).alias("target")
        )
        # 生成 bars: 使用 size 表示目标名义头寸（用现金的比例 * init_cash / close）
        # 为简便，size = target * 100 股（示例），可替换更精细的撮合与交易成本
        import numpy as np
        bars = df.select(
            [
                pl.col("date").cast(pl.Datetime("ns")).cast(pl.Int64).alias("date"),
                pl.lit(int(SizeType.ABSOLUTE.value)).alias("size_type"),
                pl.col("asset"),
                (pl.col("target") * 100).round(0).cast(pl.Int64).alias("size"),
                pl.col("close").alias("fill_price"),
                pl.col("close").alias("last_price"),
                pl.lit(1).alias("date_diff"),
            ]
        )
        return bars

    def run(self, factor_df: pl.DataFrame, prices_df: pl.DataFrame) -> Dict:
        # 准备权重与 bars
        weights = self._to_weights(factor_df)
        bars = self._bars_from_weights(weights, prices_df)
        # 配置资产信息
        assets = [{"asset": a} for a in bars.select("asset").unique().to_series().to_list()]
        bt = LightBT(init_cash=self.init_cash)
        bt.setup(assets)
        # 运行
        bt.run_bars(bars.to_pandas())
        perf = bt.trades_stats()
        # 简要返回
        return {"ret_annual": float(perf.get("ret_annual", 0.0)) if isinstance(perf, dict) else 0.0}