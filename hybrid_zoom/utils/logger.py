"""Logging, metric accumulation, and TensorBoard-safe fallbacks."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

PathLike = Union[str, Path]


def setup_logger(
    name: str = "hybrid_zoom",
    log_file: Optional[PathLike] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create an idempotent console logger and optional UTF-8 file logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    if not any(getattr(handler, "_hybrid_zoom_console", False) for handler in logger.handlers):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console._hybrid_zoom_console = True  # type: ignore[attr-defined]
        logger.addHandler(console)
    if log_file is not None:
        destination = Path(log_file).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        existing = {
            Path(handler.baseFilename).resolve()
            for handler in logger.handlers
            if isinstance(handler, logging.FileHandler)
        }
        if destination not in existing:
            file_handler = logging.FileHandler(destination, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    return logger


class NullSummaryWriter:
    """No-op subset of SummaryWriter used when TensorBoard is unavailable."""

    def add_scalar(self, *args: Any, **kwargs: Any) -> None:
        pass

    def add_image(self, *args: Any, **kwargs: Any) -> None:
        pass

    def add_images(self, *args: Any, **kwargs: Any) -> None:
        pass

    def add_text(self, *args: Any, **kwargs: Any) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> "NullSummaryWriter":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def create_summary_writer(
    log_dir: PathLike,
    enabled: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Any:
    """Return TensorBoard SummaryWriter, or a no-op writer on import failure."""

    if not enabled:
        return NullSummaryWriter()
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=str(Path(log_dir).expanduser()))
    except (ImportError, RuntimeError) as exc:
        active_logger = logger or logging.getLogger("hybrid_zoom")
        active_logger.warning("TensorBoard unavailable; logging disabled: %s", exc)
        return NullSummaryWriter()


class AverageMeter:
    """Track a sample-weighted running average."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.value = 0.0
        self.total = 0.0
        self.count = 0

    @property
    def average(self) -> float:
        return self.total / self.count if self.count else 0.0

    @property
    def avg(self) -> float:
        return self.average

    def update(self, value: Union[float, int], count: int = 1) -> None:
        if count < 0:
            raise ValueError("count cannot be negative")
        self.value = float(value)
        self.total += self.value * count
        self.count += int(count)


get_logger = setup_logger
get_summary_writer = create_summary_writer

__all__ = [
    "AverageMeter",
    "NullSummaryWriter",
    "create_summary_writer",
    "get_logger",
    "get_summary_writer",
    "setup_logger",
]
