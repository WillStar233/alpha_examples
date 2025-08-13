import io
import os
import tempfile
import polars as pl
import datetime as dt
import pytest

# 未来实现的接口占位
class MLflowLogger:
    def __init__(self):
        pass

    def start(self, run_name: str):
        raise NotImplementedError

    def log_params(self, params: dict):
        raise NotImplementedError

    def log_metrics(self, metrics: dict):
        raise NotImplementedError

    def log_artifact_text(self, name: str, content: str):
        raise NotImplementedError

    def end(self):
        raise NotImplementedError


class AlphaInspectRunner:
    def run(self, factor_df: pl.DataFrame, label_df: pl.DataFrame, groups=None, masks=None) -> dict:
        raise NotImplementedError


class LightBTRunner:
    def run(self, factor_df: pl.DataFrame, rules: dict) -> tuple[dict, str]:
        raise NotImplementedError


def test_mlflow_logging_and_evaluators(monkeypatch, tmp_path):
    calls = {"params": None, "metrics": None, "artifacts": []}

    class DummyLogger(MLflowLogger):
        def __init__(self):
            self._active = False
        def start(self, run_name: str):
            self._active = True
        def log_params(self, params: dict):
            calls["params"] = params
        def log_metrics(self, metrics: dict):
            calls["metrics"] = metrics
        def log_artifact_text(self, name: str, content: str):
            calls["artifacts"].append((name, content[:20]))
        def end(self):
            self._active = False

    class DummyAI(AlphaInspectRunner):
        def run(self, factor_df: pl.DataFrame, label_df: pl.DataFrame, groups=None, masks=None) -> dict:
            # 简单返回 IC=1.0 作为示意
            return {"IC": 1.0, "IR": 2.0}

    class DummyBT(LightBTRunner):
        def run(self, factor_df: pl.DataFrame, rules: dict) -> tuple[dict, str]:
            # 返回收益指标与报告文本
            return {"ret_annual": 0.2}, "report body"

    logger = DummyLogger()
    ai = DummyAI()
    bt = DummyBT()

    # 构造最小数据
    df_factor = pl.DataFrame({
        "date": [dt.date(2024, 1, 2), dt.date(2024, 1, 3)],
        "symbol": ["AAA", "AAA"],
        "value": [0.1, 0.2],
    })
    df_label = pl.DataFrame({
        "date": [dt.date(2024, 1, 2), dt.date(2024, 1, 3)],
        "symbol": ["AAA", "AAA"],
        "value": [0.01, 0.02],
    })

    # 模拟一次完整记录
    logger.start("run")
    logger.log_params({"factor": "demo", "version": "abc123"})

    metrics_ai = ai.run(df_factor, df_label)
    logger.log_metrics(metrics_ai)

    metrics_bt, report = bt.run(df_factor, {"rule": "ls_5x5"}), "report body"
    # emulating dual return form
    if isinstance(metrics_bt, tuple):
        metrics_bt, report = metrics_bt
    logger.log_metrics(metrics_bt)
    logger.log_artifact_text("generated_code.py", "print('hi')")

    # 生成代码与哈希留痕（模拟）
    spec_hash = "spec_abc"
    data_hash = "data_xyz"
    logger.log_params({"spec_hash": spec_hash, "data_hash": data_hash})
    logger.log_artifact_text("generated_code.py", "# code...\nFACTOR = ts_mean(close, 3)")

    logger.end()

    assert calls["params"] is not None and calls["metrics"] is not None
    assert any(name == "generated_code.py" for name, _ in calls["artifacts"])