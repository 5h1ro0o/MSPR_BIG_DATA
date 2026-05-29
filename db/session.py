"""
Gestion de la session SQLAlchemy.
Fournit un context manager et un engine singleton.
"""

from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from config.settings import settings
from monitoring.logger import get_logger

log = get_logger(__name__)

_engine = None


def get_engine():
    """Retourne l'engine SQLAlchemy (singleton)."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            settings.database_url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            echo=False,
        )
        log.debug("Engine SQLAlchemy créé")
    return _engine


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=None)


@contextmanager
def get_session() -> Session:
    """Context manager pour une session SQLAlchemy."""
    engine = get_engine()
    SessionLocal.configure(bind=engine)
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
