from __future__ import annotations

import logging
from collections import deque


class MemoryLogHandler(logging.Handler):
    def __init__(self, capacity: int = 500) -> None:
        super().__init__()
        self.records: deque[str] = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(self.format(record))


memory_handler = MemoryLogHandler()
memory_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))


def configure_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    if memory_handler not in root.handlers:
        root.addHandler(memory_handler)


def recent_logs(limit: int = 100) -> list[str]:
    return list(memory_handler.records)[-limit:]
