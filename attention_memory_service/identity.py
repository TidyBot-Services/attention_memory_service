"""Opaque identity for an AttentionBench store shared with a remote daemon."""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path


def store_id(path: Path) -> str:
    with sqlite3.connect(path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS memory_service_identity "
            "(singleton INTEGER PRIMARY KEY CHECK(singleton=1), value TEXT NOT NULL)"
        )
        row = connection.execute(
            "SELECT value FROM memory_service_identity WHERE singleton=1"
        ).fetchone()
        if row is not None:
            return str(row[0])
        value = str(uuid.uuid4())
        connection.execute(
            "INSERT INTO memory_service_identity(singleton, value) VALUES (1, ?)",
            (value,),
        )
        return value
