from datetime import datetime
import contextvars
import logging
import os
import uuid
from logging.handlers import RotatingFileHandler

from utils.path_tool import get_abs_path

LOG_ROOT = get_abs_path("logs")

os.makedirs(LOG_ROOT, exist_ok=True)

class ConditionalLineFormatter(logging.Formatter):
    def __init__(self, normal_fmt: str, error_fmt: str):
        super().__init__()
        self.normal_formatter = logging.Formatter(normal_fmt)
        self.error_formatter = logging.Formatter(error_fmt)

    def format(self, record: logging.LogRecord) -> str:
        formatter = self.error_formatter if record.levelno >= logging.ERROR else self.normal_formatter
        return formatter.format(record)


DEFAULT_LOG_FORMAT = ConditionalLineFormatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
)

_TRACE_ID: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")


class TraceIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = get_trace_id()
        return True


def set_trace_id(trace_id: str) -> str:
    normalized = str(trace_id).strip()
    if not normalized:
        normalized = "-"
    _TRACE_ID.set(normalized)
    return normalized


def get_trace_id() -> str:
    return _TRACE_ID.get()


def clear_trace_id() -> None:
    _TRACE_ID.set("-")


def ensure_trace_id() -> str:
    current = get_trace_id()
    if current and current != "-":
        return current
    return set_trace_id(uuid.uuid4().hex)


def _resolve_log_level(env_name: str, default_level: int) -> int:
    raw_value = os.getenv(env_name, "").strip().upper()
    if not raw_value:
        return default_level

    if raw_value.isdigit():
        return int(raw_value)

    return getattr(logging, raw_value, default_level)

def get_logger(
    name: str = "agent", 
    log_file: str = None, 
    console_level: int = logging.WARNING, 
    file_level: int = logging.DEBUG
    ) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set to lowest level to capture all logs
    logger.propagate = False  # Prevent log messages from being propagated to the root logger
    
    if logger.handlers:
        return logger  # Logger already configured

    console_level = _resolve_log_level("RAG_AGENT_CONSOLE_LOG_LEVEL", console_level)
    file_level = _resolve_log_level("RAG_AGENT_FILE_LOG_LEVEL", file_level)
    
    #console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)
    
    logger.addHandler(console_handler)
    
    #file handler
    if not log_file:
        log_file = os.path.join(LOG_ROOT, f"{name}_{datetime.now().strftime('%Y-%m-%d')}.log")
        
    

    file_handler = RotatingFileHandler(
    log_file, 
    maxBytes=10*1024*1024, # 10MB
    backupCount=5,          # 保留 5 个旧文件
    encoding="utf-8"
)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)
    
    logger.addHandler(file_handler)
    
    return logger

#快捷获取日志器
logger = get_logger()

