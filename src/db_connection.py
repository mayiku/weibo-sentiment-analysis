"""DB-API compatibility helpers for Turso/libSQL."""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any


class MappingRow(Sequence):
    """Tuple-compatible row with sqlite3.Row-style named access."""

    def __init__(self, columns: list[str], values: Sequence[Any]):
        self._columns = tuple(columns)
        self._values = tuple(values)
        self._index = {name: index for index, name in enumerate(self._columns)}

    def keys(self):
        return self._columns

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._values[self._index[key]]
        return self._values[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return repr(self._values)


class LibSQLCursor:
    def __init__(self, cursor):
        self._cursor = cursor

    @property
    def description(self):
        return self._cursor.description

    @property
    def lastrowid(self):
        return self._cursor.lastrowid

    @property
    def rowcount(self):
        return self._cursor.rowcount

    def _columns(self) -> list[str]:
        return [column[0] for column in (self.description or [])]

    def _wrap(self, row):
        if row is None or isinstance(row, MappingRow):
            return row
        return MappingRow(self._columns(), row)

    def execute(self, sql, parameters=()):
        self._cursor.execute(sql, parameters)
        return self

    def executemany(self, sql, parameters):
        self._cursor.executemany(sql, parameters)
        return self

    def fetchone(self):
        return self._wrap(self._cursor.fetchone())

    def fetchall(self):
        return [self._wrap(row) for row in self._cursor.fetchall()]

    def close(self):
        return self._cursor.close()

    def __iter__(self):
        for row in self._cursor:
            yield self._wrap(row)


class LibSQLConnection:
    """Wrap libsql so existing sqlite3.Row-based code can remain unchanged."""

    is_turso = True

    def __init__(self, connection, *, close_connection: bool = True):
        self._connection = connection
        self._close_connection = close_connection

    def cursor(self):
        return LibSQLCursor(self._connection.cursor())

    def execute(self, sql, parameters=()):
        return LibSQLCursor(self._connection.execute(sql, parameters))

    def executemany(self, sql, parameters):
        return LibSQLCursor(self._connection.executemany(sql, parameters))

    def executescript(self, script):
        result = self._connection.executescript(script)
        return LibSQLCursor(result) if result is not None else None

    def commit(self):
        return self._connection.commit()

    def rollback(self):
        return self._connection.rollback()

    def close(self):
        if self._close_connection:
            return self._connection.close()
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.commit()
        else:
            self.rollback()
        self.close()
        return False
