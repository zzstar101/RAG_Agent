from datetime import datetime
import logging
import os
from logging.handlers import RotatingFileHandler

from utils.path_tool import get_abs_path

LOG_ROOT = get_abs_path("logs")

os.makedirs(LOG_ROOT, exist_ok=True)

DEFAULT_LOG_FORMAT = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)

def get_logger(
    name: str = "agent", 
    log_file: str = None, 
    console_level: int = logging.INFO, 
    file_level: int = logging.DEBUG
    ) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set to lowest level to capture all logs
    logger.propagate = False  # Prevent log messages from being propagated to the root logger
    
    if logger.handlers:
        return logger  # Logger already configured
    
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

