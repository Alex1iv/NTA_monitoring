import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FORMAT =  "%(asctime)s | %(levelname)-s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger( 
    log_dir: str, 
    file_name: str,
    enabled: bool = True,
    level=logging.INFO)-> logging.Logger:
    """Логгирование
    
    Example:
        get_logger(
            log_dir="logs",
            file_name="frontend.log"
        )
        
    Args:
        path (str): path to logging directory
        file (str): log file name. Defaults to "data.logs".


    Returns:
        _logging.Logger_: logger object
    """    
    if os.path.exists(log_dir):
        with open(Path(log_dir, file_name), "a") as f:
            f.write("\n")
        
    logger_name = file_name.replace(".log", "")
    
    logger = logging.getLogger(logger_name)
    
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False
    
    # logging disabled
    if not enabled:
        logger.addHandler(logging.NullHandler())
        return logger
    
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    handler = RotatingFileHandler(
        Path(log_dir) / file_name,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=3,
        encoding="utf-8"
    )

    handler.setFormatter(formatter)

    logger.addHandler(handler)

    # silence noisy libraries
    for lib in ("pika", "paramiko", "werkzeug"):
        logging.getLogger(lib).setLevel(logging.WARNING)
        logging.getLogger(lib).propagate = False
    

    return logger