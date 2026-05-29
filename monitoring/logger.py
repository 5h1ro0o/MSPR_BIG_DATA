"""
Logging structuré via Loguru.
Rotation quotidienne, rétention 30 jours, compression gzip.
"""
import sys
from pathlib import Path
from loguru import logger

def setup_logger(log_level: str = "INFO", log_dir: Path = Path("logs")) -> None:
    """Configure les sinks loguru : console + fichier rotatif."""
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()

    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    logger.add(
        log_dir / "etl_{time:YYYY-MM-DD}.log",
        level=log_level,
        format="{time:YYYY-MM-DDTHH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}",
        rotation="00:00",
        retention="30 days",
        compression="gz",
        encoding="utf-8",
    )

def get_logger(name: str):
    """Retourne un logger contextuel pour un module donné."""
    return logger.bind(module=name)
