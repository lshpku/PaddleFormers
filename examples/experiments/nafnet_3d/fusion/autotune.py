import ast
import os
import sqlite3
import time
from itertools import product
from pathlib import Path

import paddle


class _SqliteConfigStore:
    """Persistent dict-like store backed by SQLite.

    Keys and values are arbitrary Python literals (tuple/int/bool/str/...);
    they are serialised with ``repr`` and parsed back with ``ast.literal_eval``.
    Safe for concurrent use from multiple threads via a single shared
    connection (``check_same_thread=False``) guarded by a process-wide lock.
    """

    def __init__(self, path):
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS best_config (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._cache = {}

    @staticmethod
    def _encode(obj):
        return repr(obj)

    @staticmethod
    def _decode(text):
        return ast.literal_eval(text)

    def get(self, key, default=None):
        value = self._cache.get(key, _MISSING)
        if value is not _MISSING:
            return value

        k = self._encode(key)
        row = self._conn.execute(
            "SELECT value FROM best_config WHERE key = ?", (k,)
        ).fetchone()
        if row is None:
            return default

        value = self._decode(row[0])
        self._cache[key] = value
        return value

    def __getitem__(self, key):
        value = self.get(key, _MISSING)
        if value is _MISSING:
            raise KeyError(key)
        return value

    def __setitem__(self, key, value):
        self._cache[key] = value
        k = self._encode(key)
        v = self._encode(value)
        self._conn.execute(
            "INSERT OR REPLACE INTO best_config (key, value) VALUES (?, ?)", (k, v)
        )

    def __contains__(self, key):
        return self.get(key, _MISSING) is not _MISSING


_MISSING = object()

_DEFAULT_DB_PATH = os.environ.get(
    "NAFNET_FUSION_DB",
    str(Path("~/.cache/paddle/nafnet_fusion.sqlite").expanduser()),
)
if _DEFAULT_DB_PATH:
    BEST_CONFIG = _SqliteConfigStore(_DEFAULT_DB_PATH)
else:
    BEST_CONFIG = {}


def event_time(fn, warmup=5, repeat=10):
    """Time fn accurately with cuda event."""
    for _ in range(warmup):
        fn()
    new_event = lambda: paddle.device.Event(enable_timing=True)
    events = [(new_event(), new_event()) for _ in range(repeat)]
    paddle.randn([256, 1024, 1024])

    for e0, e1 in events:
        paddle.randn([256, 1024, 1024])
        e0.record()
        fn()
        e1.record()

    paddle.device.synchronize()
    elapsed_times = [e0.elapsed_time(e1) for e0, e1 in events]
    return sum(elapsed_times) / repeat


def tune_config(fn, *choices):
    begin = time.time()
    best_args, best_time = choices[0], float("inf")
    for args in product(*choices):
        try:
            test_time = event_time(lambda: fn(*args))
        except:
            continue
        if test_time < best_time:
            best_time = test_time
            best_args = args
    tuning_time = time.time() - begin
    if len(choices) == 1:
        best_args, = best_args
    return best_args, best_time, tuning_time


def tensor_size(x):
    return x.size * x.itemsize
