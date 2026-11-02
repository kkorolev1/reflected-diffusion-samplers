import os
from contextlib import suppress
from datetime import datetime
import warnings

import hydra
import jax
import matplotlib
from omegaconf import DictConfig, OmegaConf

from utils.helper import reset_device_memory
from utils.logger import Logger, init_logger, log, set_summary, finish, set_logger
from utils.train_selector import get_train_fn

OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(version_base=None, config_path="configs", config_name="base_conf")
def main(cfg: DictConfig) -> None:
    os.environ["HYDRA_FULL_ERROR"] = "1"
    # Determine run name
    experiment_name = cfg.logger.comet.get("experiment_name")
    if not experiment_name:
        experiment_name = f"{cfg.algorithm.name}_{cfg.target.name}_{cfg.target.dim}_{datetime.now()}_seed{cfg.seed}"
        cfg.logger.comet.experiment_name = experiment_name

    if not cfg.visualize_samples:
        matplotlib.use("agg")

    # Save the resolved configuration before instantiation replaces target.fn
    # with a Python object. Also clear any logger left by a previous notebook run.
    set_logger(Logger())
    try:
        if cfg.use_logger:
            set_logger(init_logger(cfg))
        cfg = hydra.utils.instantiate(cfg)
        target = cfg.target.fn
        log({"Z_gt": target.Z, "logZ_gt": target.log_Z}, step=0)
        train_fn = get_train_fn(cfg.algorithm.name)

        if cfg.use_jit:
            train_fn(cfg, target)
        else:
            with jax.disable_jit():
                train_fn(cfg, target)
    except BaseException as error:
        try:
            set_summary("error", f"{type(error).__name__}: {error}")
        except Exception as logging_error:
            warnings.warn(f"Could not log run error: {logging_error}")
        try:
            finish(exit_code=1)
        except Exception as logging_error:
            warnings.warn(f"Could not finish logger: {logging_error}")
        with suppress(Exception):
            reset_device_memory()
        raise
    else:
        finish()
    finally:
        set_logger(Logger())


if __name__ == "__main__":
    main()
