import logging
from logging.handlers import RotatingFileHandler
from app.logger import get_logger


def test_single_rotating_file_handler():
    # Call get_logger with multiple names
    log1 = get_logger("test1")
    log2 = get_logger("test2")
    log3 = get_logger("bel.test3")

    assert log1 is not None
    assert log2 is not None
    assert log3 is not None

    # Get parent logger "bel"
    parent = logging.getLogger("bel")

    # Assert there's only one RotatingFileHandler on the parent logger
    file_handlers = [h for h in parent.handlers if isinstance(h, RotatingFileHandler)]
    assert len(file_handlers) == 1, f"Expected exactly 1 RotatingFileHandler, found {len(file_handlers)}"

    # Child loggers should have no handlers of their own (they rely on propagation)
    assert len(log1.handlers) == 0
    assert len(log2.handlers) == 0
    assert len(log3.handlers) == 0
