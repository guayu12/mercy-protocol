import logging
import os
from pathlib import Path


_logger = logging.getLogger(__name__)


def discard_file(path: str | os.PathLike[str]) -> bool:
    """Best-effort delete; on Windows a file held open by another process
    (antivirus, indexer, a reader) cannot be removed and must not fail the caller."""
    try:
        Path(path).unlink(missing_ok=True)
        return True
    except OSError as error:
        _logger.warning("Could not delete %s: %s", Path(path).name, error)
        return False
