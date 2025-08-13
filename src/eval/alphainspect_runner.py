from typing import Dict, Optional, Sequence
import polars as pl
from alphainspect import ic as ai_ic
from alphainspect import portfolio as ai_pf
from alphainspect import reports as ai_reports


class AlphaInspectRunner:
    def __init__(self) -> None:
        pass

    def _prepare_df(self, factor_df: pl.DataFrame, label_df: pl.DataFrame, factor_name: str, label_col: str) -> pl.DataFrame:
        df = factor_df.join(label_df.rename({"value": label_col}), on=["date", "symbol"], how="inner")
        df = df.rename({"symbol": "asset", "value": factor_name})
        df = (
            df.with_columns([
                pl.len().over("date").alias("n"),
                pl.col(factor_name).rank("ordinal").over("date").alias("rank"),
            ])
            .with_columns(((pl.col("rank") * 5 / pl.col("n")).ceil().clip(1, 5).cast(pl.Int32)).alias("factor_quantile"))
            .drop(["rank", "n"])
        )
        return df

    def run(self, factor_df: pl.DataFrame, label_df: pl.DataFrame, *, factor_name: str = "factor", label_name: str = "RET_FWD", output_html: Optional[str] = None) -> Dict:
        df = self._prepare_df(factor_df, label_df, factor_name, label_name)
        if df.is_empty():
            return {"IC": 0.0, "IR": 0.0}
        ic_mat = ai_ic.calc_ic(df, factors=[factor_name], forward_returns=[label_name])
        pair_col = f"{factor_name}__{label_name}"
        mean_ic = ic_mat.select(pl.col(pair_col).mean()).item()
        ic_val = float(mean_ic) if mean_ic is not None else 0.0
        ir_df = ai_ic.calc_ir(ic_mat)
        ir_val_raw = ir_df.select(pl.col(pair_col)).item()
        ir_val = float(ir_val_raw) if ir_val_raw is not None else 0.0
        _ = ai_pf.calc_cum_return_by_quantile(df, factor_quantile="factor_quantile", fwd_ret_1=label_name)
        if output_html:
            ai_reports.report_html(name=factor_name, factors=[factor_name], df=df, output=output_html, fwd_ret_1=label_name, quantiles=5)
        return {"IC": ic_val, "IR": ir_val}