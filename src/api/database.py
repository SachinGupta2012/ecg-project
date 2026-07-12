"""
Database Configuration for ECG Arrhythmia API
================================================
SQLAlchemy setup with SQLite (v1) or PostgreSQL support.
"""

import logging
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, declarative_base, sessionmaker

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Default to SQLite for v1
SQLITE_PATH = PROJECT_ROOT / "ecg.db"

# Database URL (can be overridden by environment variable)
DATABASE_URL = f"sqlite:///{SQLITE_PATH}"

# SQLAlchemy setup
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Needed for SQLite
    echo=False,
)

# Enable WAL mode for SQLite (better concurrency)
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Set SQLite pragmas for better performance."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db() -> Session:
    """
    Get a database session.

    Yields
    ------
    Session
        SQLAlchemy session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize the database (create tables)."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized at %s", SQLITE_PATH)


def get_db_session() -> Session:
    """
    Get a database session (context manager version).

    Returns
    -------
    Session
        SQLAlchemy session.
    """
    return SessionLocal()


@contextmanager
def db_session():
    """Context manager for database sessions."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_database_connection() -> bool:
    """
    Check if the database is accessible.

    Returns
    -------
    bool
        True if database is connected.
    """
    try:
        with engine.connect() as conn:
            conn.execute(engine.text("SELECT 1"))
        return True
    except Exception:
        return False
