from typing import Dict, Optional
import polars as pl


class AlphaInspectRunner:
    def __init__(self) -> None:
        try:
            import alphainspect  # type: ignore

            self._ai = alphainspect
        except Exception:
            self._ai = None

    def run(self, factor_df: pl.DataFrame, label_df: pl.DataFrame, groups: Optional[pl.DataFrame] = None, masks: Optional[pl.DataFrame] = None) -> Dict:
        # 因子与标签需按 date/symbol/value 对齐
        if self._ai is None:
            # Fallback: 简单计算相关系数作为 IC 近似
            joined = factor_df.join(label_df, on=["date", "symbol"], how="inner", suffix="_label")
            if joined.is_empty():
                return {"IC": 0.0, "IR": 0.0, "coverage": 0.0}
            grp = joined.group_by("date").agg(
                (pl.corr(pl.col("value"), pl.col("value_label"))).alias("ic")
            )
            ic = grp["ic"].mean()
            ir = (grp["ic"].mean() / (grp["ic"].std() + 1e-12)) if grp.height > 1 else 0.0
            cov = joined.height / max(1, factor_df.height)
            return {"IC": float(ic) if ic is not None else 0.0, "IR": float(ir), "coverage": float(cov)}
        # TODO: 如果安装了 AlphaInspect，这里调用实际 API
        return {"IC": 0.0, "IR": 0.0, "coverage": 0.0}