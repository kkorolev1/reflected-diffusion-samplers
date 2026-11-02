"""JSON and Comet experiment logging behind the existing trainer interface.

Each persistent logger creates a unique run directory with run.json, flushed
logs.jsonl records, PNG figures, and NPZ arrays. Comet retains these local files.
"""

from collections.abc import Mapping
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from tempfile import mkdtemp
from typing import Any, Dict, List, Optional

from matplotlib.figure import Figure
import numpy as np


def _json_value(value):
    if isinstance(value, Mapping):
        return {
            str(key): "[redacted]" if re.search(
                r"(^|[._/])(api_key|password|access_token)$", str(key), re.IGNORECASE
            ) else _json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic) or hasattr(value, "__array__"):
        return _json_value(np.asarray(value).tolist())
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")


def _scalar_metrics(metrics):
    result = {}
    for name, value in metrics.items():
        array = np.asarray(value)
        if array.size != 1 or (value is not None and array.dtype.kind not in "biuf"):
            raise ValueError(f"Metric {name!r} must be numeric and scalar; use log_array for arrays")
        result[name] = _json_value(array.item())
    return result


def _flatten_parameters(parameters, prefix=""):
    result = {}
    for key, value in parameters.items():
        name = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten_parameters(value, name))
        else:
            result[name] = value
    return result


class Logger:
    """No-op logger; shared API for persistent and in-memory backends."""

    def init(self, config: Dict[str, Any], **kwargs) -> None:
        pass

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        pass

    def log_metrics(self, metrics, step=None):
        self.log(metrics, step=step)

    def log_figure(self, figure, figure_name, step=None):
        return self.log_image(figure_name, figure, step=step)

    def log_image(self, key: str, image: Any, step: Optional[int] = None) -> None:
        pass

    def log_images(self, key: str, images: List[Any], step: Optional[int] = None) -> None:
        for index, image in enumerate(images):
            self.log_image(f"{key}_{index}", image, step=step)

    def log_array(self, name, values, step=None):
        pass

    def log_code(self, path: str = ".") -> None:
        pass

    def set_summary(self, key: str, value: Any) -> None:
        pass

    def finish(self, exit_code: int = 0) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc is not None:
                self.set_summary("error", f"{exc_type.__name__}: {exc}")
        finally:
            self.finish(exit_code=0 if exc is None else 1)


class JSONLogger(Logger):
    """Persist the project's existing mixed scalar/figure logging payloads."""

    backend = "json"

    def __init__(self, log_dir=None, run_name=""):
        self._closed = True
        self.log_dir = None
        if log_dir is not None:
            self.init({}, log_dir=log_dir, run_name=run_name)

    def init(self, config, **kwargs):
        if not self._closed:
            raise RuntimeError("Logger is already initialized")
        root = Path(kwargs.get("log_dir", "logs")).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        self.log_dir = Path(mkdtemp(prefix=f"run_{timestamp}_", dir=root))
        self._step = 0
        self._artifact_index = 0
        self._metadata = {
            "format_version": 1,
            "backend": self.backend,
            "name": kwargs.get("run_name") or config.get("name") or "",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "parameters": _json_value(kwargs.get("log_config", config)),
            "summaries": {},
        }
        self._write_metadata()
        self._stream = (self.log_dir / "logs.jsonl").open("x", encoding="utf-8")
        self._closed = False
        print(f"Experiment logs: {self.log_dir}", flush=True)

    def _write_metadata(self):
        path = self.log_dir / "run.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self._metadata, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)

    def _resolve_step(self, step):
        if step is not None:
            self._step = int(step)
        return self._step

    def _record(self, event_type, **values):
        if self._closed:
            raise RuntimeError("Logger is not initialized or has finished")
        event = {"type": event_type, "time": datetime.now(timezone.utc).isoformat(), **values}
        self._stream.write(json.dumps(event, allow_nan=False) + "\n")
        self._stream.flush()

    def _artifact_path(self, folder, name, step, suffix):
        if self._closed:
            raise RuntimeError("Logger is not initialized or has finished")
        # Include a sequence number: repeated names/steps and sanitized-name
        # collisions must not overwrite earlier artifacts referenced in the log.
        self._artifact_index += 1
        filename = re.sub(r"[^a-zA-Z0-9_.-]", "_", name)[:120]
        path = self.log_dir / folder / f"{filename}_{step:08d}_{self._artifact_index:06d}.{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def log(self, metrics, step=None):
        if step is None:
            step = metrics.get("stats/step")
        step = self._resolve_step(step)
        scalars = {}
        for name, value in metrics.items():
            if isinstance(value, Figure):
                self.log_figure(value, name, step)
            elif isinstance(value, (list, tuple)) and value and all(isinstance(v, Figure) for v in value):
                if len(value) == 1:
                    self.log_figure(value[0], name, step)
                else:
                    self.log_images(name, value, step)
            elif name.startswith("data/"):
                self.log_array(name, value, step)
            elif np.asarray(value).size != 1:
                self.log_array(name, value, step)
            else:
                scalars[name] = value
        if scalars:
            self.log_metrics(scalars, step)

    def log_metrics(self, metrics, step=None):
        self._record("metrics", step=self._resolve_step(step), metrics=_scalar_metrics(metrics))

    def log_figure(self, figure, figure_name, step=None):
        step = self._resolve_step(step)
        path = self._artifact_path("figures", figure_name, step, "png")
        figure.savefig(path, dpi=150, bbox_inches="tight")
        self._record("figure", name=figure_name, step=step, path=str(path.relative_to(self.log_dir)))
        return path

    def log_image(self, key, image, step=None):
        if isinstance(image, Figure):
            return self.log_figure(image, key, step)
        from matplotlib import image as mpl_image

        step = self._resolve_step(step)
        path = self._artifact_path("figures", key, step, "png")
        if isinstance(image, (str, Path)):
            mpl_image.imsave(path, mpl_image.imread(image))
        elif hasattr(image, "save"):
            image.save(path)
        else:
            mpl_image.imsave(path, np.asarray(image))
        self._record("image", name=key, step=step, path=str(path.relative_to(self.log_dir)))
        return path

    def log_array(self, name, values, step=None):
        step = self._resolve_step(step)
        values = np.asarray(values)
        if values.dtype.hasobject:
            raise TypeError(f"Array {name!r} has object dtype and cannot be saved without pickle")
        path = self._artifact_path("data", name, step, "npz")
        np.savez_compressed(path, data=values)
        self._record("array", name=name, step=step, path=str(path.relative_to(self.log_dir)),
                     shape=list(values.shape), dtype=str(values.dtype))
        return path

    def log_code(self, path="."):
        self._record("code", path=str(Path(path).expanduser().resolve()))

    def set_summary(self, key, value):
        self._metadata["summaries"].update(_json_value({key: value}))
        self._write_metadata()

    def finish(self, exit_code=0):
        if self._closed:
            return
        self._metadata.update(
            status="finished" if exit_code == 0 else "failed",
            exit_code=int(exit_code),
            error=self._metadata["summaries"].get("error"),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        try:
            self._write_metadata()
        finally:
            self._stream.close()
            self._closed = True


class CometLogger(JSONLogger):
    """Mirror events to Comet while always retaining the local JSON run."""

    backend = "comet"

    def init(self, config, **kwargs):
        super().init(config, **kwargs)
        self.experiment = None
        try:
            from comet_ml import Experiment

            options = dict(kwargs.get("comet_config") or {})
            run_name = options.pop("experiment_name", None) or self._metadata["name"]
            tags = options.pop("tags", None)
            # Names and tags are methods, not Experiment constructor arguments.
            options.setdefault("log_code", False)
            options.setdefault("auto_metric_logging", False)
            options.setdefault("auto_param_logging", False)
            self.experiment = Experiment(**{key: value for key, value in options.items() if value is not None})
            if run_name:
                self.experiment.set_name(run_name)
            if tags:
                self.experiment.add_tags([tags] if isinstance(tags, str) else list(tags))
            self.experiment.log_parameters(_flatten_parameters(self._metadata["parameters"]))
        except BaseException as error:
            JSONLogger.set_summary(self, "error", f"{type(error).__name__}: {error}")
            JSONLogger.finish(self, exit_code=1)
            if self.experiment is not None:
                self.experiment.end()
            raise

    def log_metrics(self, metrics, step=None):
        step = self._resolve_step(step)
        super().log_metrics(metrics, step)
        values = {key: value for key, value in _scalar_metrics(metrics).items() if value is not None}
        if values:
            self.experiment.log_metrics(values, step=step)

    def log_figure(self, figure, figure_name, step=None):
        path = super().log_figure(figure, figure_name, step)
        self.experiment.log_figure(figure=figure, figure_name=figure_name, step=self._step)
        return path

    def log_image(self, key, image, step=None):
        if isinstance(image, Figure):
            return self.log_figure(image, key, step)
        path = super().log_image(key, image, step)
        self.experiment.log_image(str(path), name=key, step=self._step)
        return path

    def log_array(self, name, values, step=None):
        path = super().log_array(name, values, step)
        self.experiment.log_asset(str(path), file_name=path.name, step=self._step)
        return path

    def log_code(self, path="."):
        super().log_code(path)
        self.experiment.log_code(folder=path)

    def set_summary(self, key, value):
        super().set_summary(key, value)
        if value is not None:
            self.experiment.log_other(key, self._metadata["summaries"][key])

    def finish(self, exit_code=0):
        if self._closed:
            return
        # Finalize local files even if ending the remote experiment fails.
        super().finish(exit_code)
        try:
            self.experiment.log_other("status", self._metadata["status"])
        finally:
            self.experiment.end()


class LocalLogger(Logger):
    def __init__(self):
        self._metrics_log = []  # List of (step, metrics dict)
        self._metrics_by_name = {}  # Dict of {metric_name: list of (step, value)}
        self._images_log = []  # List of (step, key, image object)
        self._figures_log = []  # List of (step, key, figure object)
        self._summaries = {}  # Dict of {key: value}
        self._config_log = None  # Stores config if logged
        self._code_paths = []  # List of folder paths "logged"
        self._other_log = {}  # Dict of misc logs (key: value)
        self._finished = False
        self._finish_exit_code = None

    def init(self, config: dict, **kwargs) -> None:
        self._config_log = config
        # Log config as parameters if provided
        if "log_config" in kwargs:
            self._config_log = kwargs["log_config"]

    def log(self, metrics: dict, step: int = None) -> None:
        # Store metrics in a log for later retrieval
        self._metrics_log.append((step, dict(metrics)))
        # Update metrics by name for easy access
        for k, v in metrics.items():
            if k not in self._metrics_by_name:
                self._metrics_by_name[k] = []
            self._metrics_by_name[k].append((step, v))

    def log_image(self, key: str, image: any, step: int = None) -> None:
        # Store image logs (actual objects, not serialized)
        self._images_log.append((step, key, image))

    def log_images(self, key: str, images: list, step: int = None) -> None:
        # Store each image in the provided images list
        for i, img in enumerate(images):
            name = f"{key}_{i}"
            self._images_log.append((step, name, img))

    def log_code(self, path: str = ".") -> None:
        self._code_paths.append(path)

    def set_summary(self, key: str, value: any) -> None:
        self._summaries[key] = value

    def finish(self, exit_code: int = 0) -> None:
        self._finished = True
        self._finish_exit_code = exit_code
        if exit_code != 0:
            self._other_log["error"] = True

    def get_data(self):
        """Get a dictionary of all internally stored logging data."""
        return {
            "metrics_log": list(self._metrics_log),
            "metrics_by_name": {k: list(v) for k, v in self._metrics_by_name.items()},
            "images_log": list(self._images_log),
            "figures_log": list(self._figures_log),
            "summaries": dict(self._summaries),
            "config": self._config_log,
            "code_paths": list(self._code_paths),
            "other_log": dict(self._other_log),
            "finished": self._finished,
            "finish_exit_code": self._finish_exit_code,
        }

    def get_metrics(self):
        """Return the raw list of (step, metrics dict) pairs."""
        return list(self._metrics_log)

    def get_metrics_by_name(self, name: str):
        """Return all (step, value) pairs for a given metric name, or an empty list."""
        return list(self._metrics_by_name.get(name, []))

    def get_metrics_by_step(self, step: int):
        """
        Return a list of all metrics dicts for a given step.
        Each metrics dict corresponds to a single log call with that step.
        Returns an empty list if no logs for that step.
        """
        return [m for s, m in self._metrics_log if s == step]

    def get_last_metrics(self):
        """Return last logged (step, metrics dict), or None if empty."""
        return self._metrics_log[-1] if self._metrics_log else None

    def get_images(self):
        """Return the raw list of (step, key, image) tuples."""
        return list(self._images_log)

    def get_summaries(self):
        """Return the current summary dictionary."""
        return dict(self._summaries)

    def reset(self):
        """Clear all stored logs."""
        self._metrics_log.clear()
        self._metrics_by_name.clear()
        self._images_log.clear()
        self._figures_log.clear()
        self._summaries.clear()
        self._config_log = None
        self._code_paths.clear()
        self._other_log.clear()
        self._finished = False
        self._finish_exit_code = None


class MultiLogger(Logger):
    """Logger that supports multiple backends simultaneously."""

    def __init__(self, loggers: List[Logger]):
        self.loggers = loggers

    @property
    def log_dir(self):
        """Expose the persistent run directory when combined with local logging."""
        for logger in self.loggers:
            directory = getattr(logger, "log_dir", None)
            if directory is not None:
                return directory
        return None

    def init(self, config: Dict[str, Any], **kwargs) -> None:
        for logger in self.loggers:
            logger.init(config, **kwargs)

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        for logger in self.loggers:
            logger.log(metrics, step=step)

    def log_image(self, key: str, image: Any, step: Optional[int] = None) -> None:
        for logger in self.loggers:
            logger.log_image(key, image, step=step)

    def log_images(
        self, key: str, images: List[Any], step: Optional[int] = None
    ) -> None:
        for logger in self.loggers:
            logger.log_images(key, images, step=step)

    def log_code(self, path: str = ".") -> None:
        for logger in self.loggers:
            logger.log_code(path)

    def log_array(self, name, values, step=None):
        for logger in self.loggers:
            logger.log_array(name, values, step=step)

    def set_summary(self, key: str, value: Any) -> None:
        for logger in self.loggers:
            logger.set_summary(key, value)

    def finish(self, exit_code: int = 0) -> None:
        error = None
        for logger in self.loggers:
            try:
                logger.finish(exit_code=exit_code)
            except Exception as exc:
                error = exc
        if error is not None:
            raise error


def create_logger(cfg) -> Logger:
    """Create a logger based on configuration.

    Args:
        cfg: Hydra configuration object

    Returns:
        Logger instance (single logger or multi-logger)
    """
    loggers = []

    if not cfg.get("use_logger", True):
        return Logger()

    logger_cfg = cfg.get("logger", {})
    # Retain the existing boolean configuration and in-memory sweep logger.
    # Comet itself includes JSON, so do not create a duplicate persistent run.
    if logger_cfg.get("use_comet", False):
        loggers.append(CometLogger())
    elif logger_cfg.get("use_json", False):
        loggers.append(JSONLogger())

    if logger_cfg.get("use_local", False):
        loggers.append(LocalLogger())

    # Return appropriate logger
    if len(loggers) == 0:
        return Logger()  # No-op logger
    elif len(loggers) == 1:
        return loggers[0]
    else:
        return MultiLogger(loggers)


def init_logger(cfg, config_dict: Optional[Dict[str, Any]] = None) -> Logger:
    """Initialize and return a logger based on configuration.

    Args:
        cfg: Hydra configuration object
        config_dict: Optional config dictionary to log as parameters

    Returns:
        Initialized logger instance
    """
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf

    logger = create_logger(cfg)
    if config_dict is None:
        config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    logger_cfg = cfg.get("logger", {})
    comet_config = dict(logger_cfg.get("comet") or {})
    # Hydra can change cwd; its absolute output directory avoids nesting a
    # relative log_dir twice and separates multirun jobs correctly.
    log_dir = HydraConfig.get().runtime.output_dir if HydraConfig.initialized() else cfg.get("log_dir", "logs")
    logger.init(
        {},
        log_config=config_dict,
        comet_config=comet_config,
        log_dir=log_dir,
        run_name=comet_config.get("experiment_name") or "",
    )
    return logger


# Global logger instance for convenience
_logger: Optional[Logger] = None


def get_logger() -> Logger:
    """Get the global logger instance."""
    global _logger
    if _logger is None:
        _logger = Logger()  # No-op logger
    return _logger


def set_logger(logger: Logger) -> None:
    """Set the global logger instance."""
    global _logger
    _logger = logger


# Convenience functions that use the global logger
def log(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log metrics using the global logger."""
    get_logger().log(metrics, step=step)


def log_image(key: str, image: Any, step: Optional[int] = None) -> None:
    """Log an image using the global logger."""
    get_logger().log_image(key, image, step=step)


def log_images(key: str, images: List[Any], step: Optional[int] = None) -> None:
    """Log multiple images using the global logger."""
    get_logger().log_images(key, images, step=step)


def log_code(path: str = ".") -> None:
    """Log code using the global logger."""
    get_logger().log_code(path)


def log_array(name, values, step=None):
    """Save an array beside the run's metrics and figures."""
    return get_logger().log_array(name, values, step=step)


def set_summary(key: str, value: Any) -> None:
    """Set a summary value using the global logger."""
    get_logger().set_summary(key, value)


def finish(exit_code: int = 0) -> None:
    """Finish the global logger session."""
    get_logger().finish(exit_code=exit_code)
