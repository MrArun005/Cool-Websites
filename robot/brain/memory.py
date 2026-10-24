"""The robot's memory.

A robot has two kinds of memory, just like a computer:

* Short-term memory lives in RAM. It is fast, but everything in it vanishes
  when the power goes off. Here it is a small queue of the last things heard.
* Long-term memory lives on storage (the SD card / SSD). It survives reboots.
  Here it is a SQLite database file: facts ("name" -> "Arun") and a diary of
  events ("waved at 10:42").
"""

import sqlite3
import time
from collections import deque


class Memory:
    def __init__(self, path="robot_memory.db", short_term_size=10):
        self.short_term = deque(maxlen=short_term_size)  # RAM: gone on power-off
        self.db = sqlite3.connect(path)                   # storage: kept forever
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS facts (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS diary (
                at REAL NOT NULL,
                text TEXT NOT NULL
            );
            """
        )

    # --- short-term (RAM) -------------------------------------------------
    def hear(self, text):
        self.short_term.append(text)

    def recent(self):
        return list(self.short_term)

    # --- long-term (storage) ----------------------------------------------
    def remember(self, key, value):
        key = key.strip().lower()
        with self.db:
            self.db.execute(
                "INSERT INTO facts (key, value, updated) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated = excluded.updated",
                (key, value.strip(), time.time()),
            )

    def recall(self, key):
        row = self.db.execute(
            "SELECT value FROM facts WHERE key = ?", (key.strip().lower(),)
        ).fetchone()
        return row[0] if row else None

    def forget(self, key):
        with self.db:
            cur = self.db.execute("DELETE FROM facts WHERE key = ?", (key.strip().lower(),))
        return cur.rowcount > 0

    def all_facts(self):
        return self.db.execute("SELECT key, value FROM facts ORDER BY key").fetchall()

    def log(self, text):
        with self.db:
            self.db.execute("INSERT INTO diary (at, text) VALUES (?, ?)", (time.time(), text))

    def diary(self, limit=10):
        rows = self.db.execute(
            "SELECT at, text FROM diary ORDER BY at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [(time.strftime("%Y-%m-%d %H:%M", time.localtime(at)), text) for at, text in rows]

    def close(self):
        self.db.close()
