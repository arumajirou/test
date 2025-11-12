import logging
import logging.config
import yaml
from pathlib import Path

def setup_logging(config_path: str | Path | None = None):
    if config_path is None:
        return
    p = Path(config_path)
    if p.exists():
        with p.open("r") as f:
            cfg = yaml.safe_load(f)
        logging.config.dictConfig(cfg)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
