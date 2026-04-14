"""Central logging configuration.

Pipes Python warnings/errors from the whole app into Telegram through the
TelegramLogHandler while leaving stdout readable.
"""

from __future__ import annotations

import logging
import sys
import warnings

from .telegram_bot import TelegramLogHandler, TelegramNotifier


def configure(notifier: TelegramNotifier, level: int = logging.INFO) -> None:
    root = logging.getLogger()
    # Clear pre-existing handlers
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(level)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    ))
    stream.setLevel(level)
    root.addHandler(stream)

    # Telegram mirror for WARNING+
    root.addHandler(TelegramLogHandler(notifier, level=logging.WARNING))

    # Capture Python warnings into logging
    logging.captureWarnings(True)
    warnings.simplefilter("default")

    # Route uncaught exceptions to logging so Telegram sees tracebacks
    def _excepthook(exc_type, exc, tb):
        logging.getLogger("al3x.uncaught").critical(
            "Uncaught exception", exc_info=(exc_type, exc, tb),
        )
    sys.excepthook = _excepthook
