import os
import hashlib
import polars as pl
import datetime as dt
from typing import List, Optional

from engine.factor_engine import FactorEngine, FactorSpec
from label.label_engine import make_forward_return
from tracking.mlflow_logger import MLflowLogger
from eval.alphainspect_runner import AlphaInspectRunner
from eval.lightbt_runner import LightBTRunner


def _hash_text(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:8]


def _hash_df(df: pl.DataFrame, max_rows: int = 10000) -> str:
    if df.is_empty():
        return "empty"
    import io
    buf = io.BytesIO()
    df.head(max_rows).write_parquet(buf)
    return hashlib.md5(buf.getvalue()).hexdigest()[:8]


class FactorOrchestrator:
    def __init__(
        self,
        engine: FactorEngine,
        logger: Optional[MLflowLogger] = None,
        ai_runner: Optional[AlphaInspectRunner] = None,
        bt_runner: Optional[LightBTRunner] = None,
        experiment_name: str = "AlphaFactors",
        tracking_uri: Optional[str] = None,
    ) -> None:
        self.engine = engine
        self.logger = logger or MLflowLogger(tracking_uri=tracking_uri, experiment_name=experiment_name)
        self.ai = ai_runner or AlphaInspectRunner()
        self.bt = bt_runner or LightBTRunner()

    def run_full(
        self,
        spec: FactorSpec,
        universe: List[str],
        start: dt.date,
        end: dt.date,
        label_horizon: int = 5,
    ) -> dict:
        # 1) compute
        df_factor = self.engine.compute_full(spec, universe, start, end)
        # 2) label
        # 需要 close 数据
        raw_px = self.engine.data.fetch(universe, start, end, ["close"], spec.freq)
        df_label = make_forward_return(raw_px, label_horizon)
        # 3) eval/backtest
        metrics_ai = self.ai.run(df_factor, df_label)
        metrics_bt = self.bt.run(df_factor, {"label_df": df_label})
        # 4) log
        run_name = f"{spec.name}_{spec.freq}_full"
        self.logger.start(run_name)
        spec_hash = _hash_text("".join([fn.__name__ for fn in spec.blocks]) + spec.output_var)
        data_hash = _hash_df(raw_px)
        self.logger.log_params({
            "factor": spec.name,
            "freq": spec.freq,
            "inputs": ",".join(spec.inputs),
            "lookback": spec.lookback,
            "lag": spec.lag,
            "spec_hash": spec_hash,
            "data_hash": data_hash,
        })
        all_metrics = {**metrics_ai, **metrics_bt}
        self.logger.log_metrics(all_metrics)
        # 保存一份样本
        sample = df_factor.tail(1000)
        sample_path = os.path.join(os.getcwd(), f"{spec.name}_sample.parquet")
        sample.write_parquet(sample_path)
        self.logger.log_artifact_file(sample_path)
        # 生成代码（如果有）
        gen_code = getattr(self.engine, "last_generated_code", "")
        if gen_code:
            self.logger.log_artifact_text("generated_code.py", gen_code)
        self.logger.end()
        return {"metrics": all_metrics, "factor_rows": df_factor.height}

    def run_incremental(
        self,
        spec: FactorSpec,
        universe: List[str],
        new_dates: List[dt.date],
        label_horizon: int = 5,
    ) -> dict:
        d0, d1 = min(new_dates), max(new_dates)
        df_factor = self.engine.compute_incremental(spec, universe, new_dates)
        raw_px = self.engine.data.fetch(universe, d0, d1, ["close"], spec.freq)
        df_label = make_forward_return(raw_px, label_horizon)
        metrics_ai = self.ai.run(df_factor, df_label)
        metrics_bt = self.bt.run(df_factor, {"label_df": df_label})
        run_name = f"{spec.name}_{spec.freq}_inc"
        self.logger.start(run_name)
        self.logger.log_params({"dates": f"{d0}:{d1}", "factor": spec.name})
        self.logger.log_metrics({**metrics_ai, **metrics_bt})
        self.logger.end()
        return {"metrics": {**metrics_ai, **metrics_bt}, "factor_rows": df_factor.height}