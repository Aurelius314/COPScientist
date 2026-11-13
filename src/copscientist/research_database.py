"""Persistent repository for candidate lineage, evidence, and sources."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .utils import Candidate, Evaluation, Idea, Source


class ResearchDatabase:
    """SQLite storage shared by the research agents."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS candidates (
                id TEXT PRIMARY KEY, task TEXT NOT NULL, round INTEGER NOT NULL,
                parent_id TEXT, score REAL, status TEXT NOT NULL, payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT, task TEXT NOT NULL,
                category TEXT NOT NULL CHECK(category IN ('ideation','experimentation')),
                candidate_id TEXT, summary TEXT NOT NULL, evidence TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(task, category, candidate_id, summary)
            );
            CREATE TABLE IF NOT EXISTS sources (
                url TEXT PRIMARY KEY, payload TEXT NOT NULL
            );
        """)
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def backup(self, destination: Path) -> None:
        """Create a consistent SQLite snapshot for a checkpoint."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(destination)) as target:
            self.connection.backup(target)

    def record_sources(self, sources: list[Source]) -> None:
        self.connection.executemany(
            "INSERT OR REPLACE INTO sources(url,payload) VALUES(?,?)",
            [(source.url, json.dumps(asdict(source), ensure_ascii=False)) for source in sources],
        )
        self.connection.commit()

    def record_candidate(self, task: str, candidate: Candidate) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO candidates VALUES(?,?,?,?,?,?,?,?)",
            (candidate.id, task, candidate.round, candidate.parent_id,
             candidate.evaluation.score, candidate.evaluation.status,
             json.dumps(candidate.to_dict(), ensure_ascii=False),
             datetime.now(timezone.utc).isoformat()),
        )
        self.connection.commit()

    def record_memories(self, task: str, candidate_id: str, ideation: str,
                        experiment: str, evidence: dict) -> None:
        evidence_json = json.dumps(evidence, ensure_ascii=False)
        now = datetime.now(timezone.utc).isoformat()
        self.connection.executemany(
            "INSERT OR IGNORE INTO memories(task,category,candidate_id,summary,evidence,created_at) "
            "VALUES(?,?,?,?,?,?)",
            [(task, "ideation", candidate_id, ideation, evidence_json, now),
             (task, "experimentation", candidate_id, experiment, evidence_json, now)],
        )
        self.connection.commit()

    def retrieve(self, task: str, limit: int = 8) -> list[dict]:
        rows = self.connection.execute(
            "SELECT category,summary,evidence FROM memories WHERE task=? ORDER BY id DESC LIMIT ?",
            (task, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def candidates(self, task: str) -> list[Candidate]:
        rows = self.connection.execute(
            "SELECT payload FROM candidates WHERE task=? ORDER BY created_at,id", (task,)
        ).fetchall()
        result = []
        for row in rows:
            data = json.loads(row["payload"])
            data["idea"] = Idea(**data["idea"])
            data["evaluation"] = Evaluation(**data["evaluation"])
            result.append(Candidate(**data))
        return result
