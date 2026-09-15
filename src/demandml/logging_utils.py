"""Consistent, quiet-by-default logging with rich formatting."""

from __future__ import annotations

import logging
import os

from rich.logging import RichHandler

_CONFIGURED = False


def setup_logging(level: str | int | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = level or os.environ.get("DEMANDML_LOGLEVEL", "INFO")
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False, markup=False)],
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
