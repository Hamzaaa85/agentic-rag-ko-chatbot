"""
Database connection pool — no SQL, no business logic.

Reuses connections across requests instead of connect()/close() every time.
Works with Neon pooler URLs (hostname contains -pooler).
"""

from contextlib import contextmanager
from typing import Generator

import psycopg2
from psycopg2 import pool
from psycopg2.extensions import cursor as Cursor
from psycopg2.extras import RealDictCursor

from backend.app.config import get_settings

# Module-level pool — created once on first use
_connection_pool: pool.ThreadedConnectionPool | None = None


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

    Do not call conn.close() — use putconn() so the pool can reuse the connection.
    """
    pg_pool = get_connection_pool()
    conn = pg_pool.getconn()  # take from pool (blocks if maxconn exhausted)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            yield cur
        conn.commit()  # no-op for read-only; required if caller runs writes later
    except Exception:
        conn.rollback()  # reset connection state before returning to pool
        raise
    finally:
        pg_pool.putconn(conn)  # return to pool — do not close()
