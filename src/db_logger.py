"""
Phase 5a — Async MySQL logging for every prediction FastAPI serves.

Uses environment variables for connection details so credentials never get
hardcoded into source code (a real production API would pull these from a
secrets manager, not a .env file, but .env is the standard local-dev pattern).
"""

import json
import os
from datetime import datetime

import aiomysql

DB_CONFIG = {
    "host": os.environ.get("MYSQL_HOST", "localhost"),
    "port": int(os.environ.get("MYSQL_PORT", 3306)),
    "user": os.environ.get("MYSQL_USER", "churn_user"),
    "password": os.environ.get("MYSQL_PASSWORD", "churn_pass"),
    "db": os.environ.get("MYSQL_DATABASE", "churn_db"),
}

_pool = None  # a connection pool is reused across requests, not reconnected each time


async def get_pool():
    global _pool
    if _pool is None:
        _pool = await aiomysql.create_pool(**DB_CONFIG, minsize=1, maxsize=5)
    return _pool


async def init_db():
    """Creates the predictions table if it doesn't already exist. Call once at API startup."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    features JSON NOT NULL,
                    churn_probability FLOAT NOT NULL,
                    top_drivers JSON NOT NULL,
                    created_at DATETIME NOT NULL
                )
            """)
            await conn.commit()


async def log_prediction(features: dict, churn_probability: float, top_drivers: dict):
    """
    Fire-and-forget style call from the /predict endpoint. Uses the shared
    pool rather than opening a new connection per request, which would add
    real latency (TCP handshake + MySQL auth) to every single prediction.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO predictions (features, churn_probability, top_drivers, created_at)
                VALUES (%s, %s, %s, %s)
                """,
                (
                    json.dumps(features),
                    churn_probability,
                    json.dumps(top_drivers),
                    datetime.utcnow(),
                ),
            )
            await conn.commit()


async def close_pool():
    global _pool
    if _pool is not None:
        _pool.close()
        await _pool.wait_closed()
        _pool = None