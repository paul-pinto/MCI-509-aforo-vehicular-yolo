import threading
from datetime import datetime


class SessionService:
    def __init__(
        self,
        database_service,
        traffic_config_service,
    ):
        self.database = (
            database_service
        )

        self.traffic_config_service = (
            traffic_config_service
        )

        self.lock = threading.RLock()

        self.active_session_id = None
        self.active_started_at = None

    # ========================================================
    # UTILIDADES
    # ========================================================

    @staticmethod
    def now():
        return (
            datetime.now()
            .astimezone()
        )

    @staticmethod
    def iso(
        value,
    ):
        return value.isoformat(
            timespec="milliseconds"
        )

    def find_config_value(
        self,
        obj,
        names,
        default=None,
    ):
        if isinstance(obj, dict):

            for key, value in obj.items():

                if (
                    str(key).lower()
                    in names
                ):
                    return value

            for value in obj.values():

                found = (
                    self.find_config_value(
                        value,
                        names,
                        None,
                    )
                )

                if found is not None:
                    return found

        elif isinstance(obj, list):

            for value in obj:

                found = (
                    self.find_config_value(
                        value,
                        names,
                        None,
                    )
                )

                if found is not None:
                    return found

        return default

    def metadata(self):
        config = (
            self.traffic_config_service
            .config
        )

        return {
            "station_id": str(
                self.find_config_value(
                    config,
                    {
                        "estacion_id",
                        "estacion",
                        "station_id",
                    },
                    "UADE-01",
                )
            ),

            "road": str(
                self.find_config_value(
                    config,
                    {
                        "via",
                        "avenida",
                        "road",
                    },
                    "Av. Independencia",
                )
            ),

            "reference": str(
                self.find_config_value(
                    config,
                    {
                        "referencia",
                        "reference",
                    },
                    "UADE",
                )
            ),

            "city": str(
                self.find_config_value(
                    config,
                    {
                        "ciudad",
                        "city",
                    },
                    "Buenos Aires",
                )
            ),
        }

    # ========================================================
    # INICIAR
    # ========================================================

    def start(self):
        with self.lock:

            if self.active_session_id is not None:
                raise RuntimeError(
                    "Ya existe una sesión de aforo activa."
                )

            started_at = self.now()

            session_id = (
                started_at.strftime(
                    "%Y%m%d_%H%M%S"
                )
            )

            metadata = (
                self.metadata()
            )

            session = {
                "session_id": session_id,

                "station_id": (
                    metadata["station_id"]
                ),

                "road": metadata["road"],

                "reference": (
                    metadata["reference"]
                ),

                "city": metadata["city"],

                "started_at": (
                    self.iso(started_at)
                ),

                "ended_at": None,

                "duration_seconds": None,

                "total_vehicles": 0,

                "status": "EN_CURSO",
            }

            self.database.insert_session(
                session
            )

            self.active_session_id = (
                session_id
            )

            self.active_started_at = (
                started_at
            )

            return session

    # ========================================================
    # DETENER
    # ========================================================

    def stop(
        self,
        total_vehicles,
        status="FINALIZADA",
    ):
        with self.lock:

            if self.active_session_id is None:
                raise RuntimeError(
                    "No existe una sesión activa."
                )

            ended_at = self.now()

            duration_seconds = (
                ended_at
                - self.active_started_at
            ).total_seconds()

            session_id = (
                self.active_session_id
            )

            self.database.update_session(
                session_id,

                ended_at=self.iso(
                    ended_at
                ),

                duration_seconds=round(
                    duration_seconds,
                    3,
                ),

                total_vehicles=int(
                    total_vehicles
                ),

                status=status,
            )

            session = (
                self.database
                .get_session(
                    session_id
                )
            )

            self.active_session_id = None
            self.active_started_at = None

            return session

    # ========================================================
    # EVENTOS
    # ========================================================

    def record_event(
        self,
        event,
    ):
        with self.lock:

            if self.active_session_id is None:
                return None

            event = dict(event)

            event["session_id"] = (
                self.active_session_id
            )

            event_id = (
                self.database
                .insert_event(
                    event
                )
            )

            return event_id

    # ========================================================
    # ESTADO
    # ========================================================

    def is_active(self):
        with self.lock:
            return (
                self.active_session_id
                is not None
            )

    def current(self):
        with self.lock:

            if self.active_session_id is None:
                return {
                    "active": False,
                    "session": None,
                }

            session = (
                self.database
                .get_session(
                    self.active_session_id
                )
            )

            return {
                "active": True,
                "session": session,
            }

    def get_session(
        self,
        session_id,
    ):
        return (
            self.database
            .get_session(
                session_id
            )
        )

    def list_sessions(
        self,
        limit=100,
    ):
        return (
            self.database
            .list_sessions(
                limit=limit
            )
        )

    def get_events(
        self,
        session_id,
    ):
        return (
            self.database
            .get_events(
                session_id
            )
        )