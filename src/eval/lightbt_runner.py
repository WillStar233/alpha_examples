from typing import Dict
import polars as pl


class LightBTRunner:
    def __init__(self) -> None:
        try:
            import lightbt  # type: ignore

            self._bt = lightbt
        except Exception:
            self._bt = None

    def run(self, factor_df: pl.DataFrame, rules: dict) -> Dict:
        # Fallback 简单回测：按日 zscore 后多空组合 +1/-1 权重，与 label value 相乘求均值
        # 需要规则里提供 label_df
        label_df = rules.get("label_df")
        if label_df is None or factor_df.is_empty():
            return {"ret_annual": 0.0}
        joined = factor_df.join(label_df, on=["date", "symbol"], how="inner", suffix="_label")
        if joined.is_empty():
            return {"ret_annual": 0.0}
        # 横截面标准化
        xx = (
            joined.group_by("date")
            .agg([
                ((pl.col("value") - pl.col("value").mean()) / (pl.col("value").std() + 1e-12)).alias("score"),
                pl.col("value_label"),
            ])
            .explode(["score", "value_label"])  # 保持对齐
        )
        # 简单权重：sign(score)
        xx = xx.with_columns((pl.col("score").clip_min(-1.0).clip_max(1.0).sign()).alias("w"))
        daily = xx.group_by("date").agg((pl.col("w") * pl.col("value_label")).mean().alias("ret"))
        ret_ann = float(daily["ret"].mean() * 252.0) if daily.height > 0 else 0.0
        return {"ret_annual": ret_ann}