import sqlite3
import threading
from pathlib import Path


class DatabaseService:
    def __init__(self, database_path: Path):
        self.database_path = database_path

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.lock = threading.RLock()

        self.initialize()

    # ========================================================
    # CONEXIÓN
    # ========================================================

    def connect(self):
        connection = sqlite3.connect(
            self.database_path,
            timeout=30,
        )

        connection.row_factory = sqlite3.Row

        connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        connection.execute(
            "PRAGMA journal_mode = WAL"
        )

        return connection

    # ========================================================
    # INICIALIZACIÓN
    # ========================================================

    def initialize(self):
        with self.lock:
            with self.connect() as connection:

                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,

                        station_id TEXT,
                        road TEXT,
                        reference TEXT,
                        city TEXT,

                        started_at TEXT NOT NULL,
                        ended_at TEXT,

                        duration_seconds REAL,

                        total_vehicles INTEGER NOT NULL
                            DEFAULT 0,

                        status TEXT NOT NULL
                    )
                    """
                )

                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,

                        session_id TEXT NOT NULL,

                        timestamp TEXT NOT NULL,

                        track_id INTEGER NOT NULL,

                        class_id INTEGER NOT NULL,
                        class_name TEXT NOT NULL,

                        direction TEXT NOT NULL,

                        confidence REAL,

                        start_distance_px REAL,
                        end_distance_px REAL,

                        displacement_px REAL,
                        normal_progress_px REAL,

                        FOREIGN KEY (session_id)
                            REFERENCES sessions(session_id)
                            ON DELETE CASCADE
                    )
                    """
                )

                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_events_session
                    ON events(session_id)
                    """
                )

                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_events_timestamp
                    ON events(timestamp)
                    """
                )

                connection.commit()

    # ========================================================
    # SESIONES
    # ========================================================

    def insert_session(
        self,
        session,
    ):
        with self.lock:
            with self.connect() as connection:

                connection.execute(
                    """
                    INSERT INTO sessions (
                        session_id,
                        station_id,
                        road,
                        reference,
                        city,
                        started_at,
                        ended_at,
                        duration_seconds,
                        total_vehicles,
                        status
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session["session_id"],
                        session["station_id"],
                        session["road"],
                        session["reference"],
                        session["city"],
                        session["started_at"],
                        session["ended_at"],
                        session["duration_seconds"],
                        session["total_vehicles"],
                        session["status"],
                    ),
                )

                connection.commit()

    def update_session(
        self,
        session_id,
        *,
        ended_at,
        duration_seconds,
        total_vehicles,
        status,
    ):
        with self.lock:
            with self.connect() as connection:

                connection.execute(
                    """
                    UPDATE sessions

                    SET
                        ended_at = ?,
                        duration_seconds = ?,
                        total_vehicles = ?,
                        status = ?

                    WHERE session_id = ?
                    """,
                    (
                        ended_at,
                        duration_seconds,
                        total_vehicles,
                        status,
                        session_id,
                    ),
                )

                connection.commit()

    def get_session(
        self,
        session_id,
    ):
        with self.lock:
            with self.connect() as connection:

                row = connection.execute(
                    """
                    SELECT *
                    FROM sessions
                    WHERE session_id = ?
                    """,
                    (session_id,),
                ).fetchone()

                if row is None:
                    return None

                return dict(row)

    def list_sessions(
        self,
        limit=100,
    ):
        with self.lock:
            with self.connect() as connection:

                rows = connection.execute(
                    """
                    SELECT *
                    FROM sessions

                    ORDER BY started_at DESC

                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()

                return [
                    dict(row)
                    for row in rows
                ]

    # ========================================================
    # EVENTOS
    # ========================================================

    def insert_event(
        self,
        event,
    ):
        with self.lock:
            with self.connect() as connection:

                cursor = connection.execute(
                    """
                    INSERT INTO events (
                        session_id,
                        timestamp,
                        track_id,
                        class_id,
                        class_name,
                        direction,
                        confidence,
                        start_distance_px,
                        end_distance_px,
                        displacement_px,
                        normal_progress_px
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event["session_id"],
                        event["timestamp"],
                        event["track_id"],
                        event["clase_id"],
                        event["clase"],
                        event["direccion"],
                        event["confianza"],
                        event["distancia_inicio_px"],
                        event["distancia_final_px"],
                        event["desplazamiento_px"],
                        event["progreso_normal_px"],
                    ),
                )

                connection.commit()

                return cursor.lastrowid

    def get_events(
        self,
        session_id,
    ):
        with self.lock:
            with self.connect() as connection:

                rows = connection.execute(
                    """
                    SELECT *
                    FROM events

                    WHERE session_id = ?

                    ORDER BY timestamp ASC
                    """,
                    (session_id,),
                ).fetchall()

                return [
                    dict(row)
                    for row in rows
                ]

    def count_events(
        self,
        session_id,
    ):
        with self.lock:
            with self.connect() as connection:

                row = connection.execute(
                    """
                    SELECT COUNT(*) AS total
                    FROM events
                    WHERE session_id = ?
                    """,
                    (session_id,),
                ).fetchone()

                return int(
                    row["total"]
                )