import os
import json
import tempfile
from typing import Optional, Dict


class MLflowLogger:
    def __init__(self, tracking_uri: Optional[str] = None, experiment_name: str = "AlphaFactors") -> None:
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "")
        self.experiment_name = experiment_name
        self._mlflow = None
        self._run = None
        self._fallback_dir = None
        try:
            import mlflow  # type: ignore

            self._mlflow = mlflow
        except Exception:
            self._mlflow = None

    def start(self, run_name: str):
        if self._mlflow is not None:
            if self.tracking_uri:
                self._mlflow.set_tracking_uri(self.tracking_uri)
            self._mlflow.set_experiment(self.experiment_name)
            self._run = self._mlflow.start_run(run_name=run_name)
        else:
            self._fallback_dir = tempfile.mkdtemp(prefix="mlflow_fallback_")
            os.makedirs(self._fallback_dir, exist_ok=True)
            with open(os.path.join(self._fallback_dir, "run.json"), "w") as f:
                json.dump({"run_name": run_name}, f)

    def log_params(self, params: Dict):
        if self._mlflow is not None and self._run is not None:
            self._mlflow.log_params(params)
        elif self._fallback_dir:
            path = os.path.join(self._fallback_dir, "params.json")
            _merge_json(path, params)

    def log_metrics(self, metrics: Dict):
        if self._mlflow is not None and self._run is not None:
            self._mlflow.log_metrics(metrics)
        elif self._fallback_dir:
            path = os.path.join(self._fallback_dir, "metrics.json")
            _merge_json(path, metrics)

    def log_artifact_text(self, name: str, content: str):
        if self._mlflow is not None and self._run is not None:
            import io

            tmp = io.StringIO(content)
            # mlflow 无直接文本接口，写入临时文件
            import tempfile

            with tempfile.NamedTemporaryFile("w", delete=False, suffix=os.path.splitext(name)[1] or ".txt") as f:
                f.write(content)
                tmp_path = f.name
            self._mlflow.log_artifact(tmp_path, artifact_path="artifacts")
            os.unlink(tmp_path)
        elif self._fallback_dir:
            path = os.path.join(self._fallback_dir, name)
            with open(path, "w") as f:
                f.write(content)

    def log_artifact_file(self, path: str, artifact_name: Optional[str] = None):
        if self._mlflow is not None and self._run is not None:
            self._mlflow.log_artifact(path, artifact_path="artifacts")
        elif self._fallback_dir:
            import shutil

            dst = os.path.join(self._fallback_dir, artifact_name or os.path.basename(path))
            shutil.copy2(path, dst)

    def end(self):
        if self._mlflow is not None and self._run is not None:
            self._mlflow.end_run()
            self._run = None
        # fallback 模式无需额外处理


def _merge_json(path: str, content: Dict):
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception:
        data = {}
    data.update(content)
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)