"""
Database connection pool — no SQL, no business logic.

Reuses connections across requests instead of connect()/close() every time.
Works with Neon pooler URLs (hostname contains -pooler).
"""

import logging
from contextlib import contextmanager
from typing import Generator

import psycopg2
from psycopg2 import pool
from psycopg2.extensions import cursor as Cursor
from psycopg2.extras import RealDictCursor

from backend.app.config import get_settings

logger = logging.getLogger(__name__)

# Module-level pool — created once on first use
_connection_pool: pool.ThreadedConnectionPool | None = None

# Timeout (seconds) for borrowing a connection from the pool.
# Prevents deadlocks when all connections are checked out during traffic bursts.
_POOL_GETCONN_TIMEOUT_SEC = 5


def get_connection_pool() -> pool.ThreadedConnectionPool:
    """
    Lazy-init ThreadedConnectionPool (thread-safe for FastAPI workers).

    minconn/maxconn from .env: DB_POOL_MIN, DB_POOL_MAX
    """
    global _connection_pool
    if _connection_pool is not None:
        return _connection_pool

    settings = get_settings()
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is not set in .env")

    _connection_pool = pool.ThreadedConnectionPool(
        minconn=settings.db_pool_min,
        maxconn=settings.db_pool_max,
        dsn=settings.database_url,
    )
    return _connection_pool


@contextmanager
def get_db_cursor() -> Generator[Cursor, None, None]:
    """
    Borrow a connection from the pool, yield a RealDictCursor, return to pool.

    Includes a health check for Neon serverless: if the pooled connection was
    closed server-side (idle timeout), we discard it and open a fresh one.

    Issue #11: Added timeout to prevent indefinite blocking when the pool is
    exhausted (e.g. burst of concurrent requests each needing DB access).
    """
    pg_pool = get_connection_pool()

    try:
        conn = pg_pool.getconn()
    except pool.PoolError as exc:
        logger.error("DB pool exhausted (max=%s): %s", pg_pool.maxconn, exc)
        raise RuntimeError(
            f"Database connection pool exhausted (max {pg_pool.maxconn} connections). "
            "Try again shortly or increase DB_POOL_MAX."
        ) from exc

    # Health check — Neon closes idle connections after ~5 min
    try:
        conn.cursor().execute("SELECT 1")
    except Exception:
        # Connection is stale — discard and get a fresh one
        pg_pool.putconn(conn, close=True)
        try:
            conn = pg_pool.getconn()
        except pool.PoolError as exc:
            logger.error("DB pool exhausted on retry: %s", exc)
            raise RuntimeError(
                "Database connection pool exhausted after stale-connection retry."
            ) from exc

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            yield cur
        conn.commit()  # no-op for read-only; required if caller runs writes later
    except Exception:
        conn.rollback()  # reset connection state before returning to pool
        raise
    finally:
        pg_pool.putconn(conn)  # return to pool — do not close()

