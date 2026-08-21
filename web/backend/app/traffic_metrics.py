import json

from datetime import datetime
from pathlib import Path


class TrafficMetricsService:
    def __init__(
        self,
        project_root: Path,
    ):
        self.project_root = (
            project_root
        )

        self.config_path = (
            project_root
            / "config_tpda.json"
        )

        self.config = {}

        self.k_vhp = None

        self.load_config()

    # ========================================================
    # CONFIGURACIÓN TPDA
    # ========================================================

    def load_config(self):
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"No existe: "
                f"{self.config_path}"
            )

        with open(
            self.config_path,
            "r",
            encoding="utf-8",
        ) as f:
            self.config = json.load(
                f
            )

        self.k_vhp = (
            self._find_numeric_value(
                self.config,
                {
                    "k_vhp",
                    "k",
                    "factor_k",
                },
            )
        )

        if self.k_vhp is None:
            raise RuntimeError(
                "No se pudo localizar "
                "k_vhp en config_tpda.json."
            )

        self.k_vhp = float(
            self.k_vhp
        )

        if self.k_vhp <= 0:
            raise RuntimeError(
                "k_vhp debe ser mayor que 0."
            )

    def _find_numeric_value(
        self,
        obj,
        names,
    ):
        if isinstance(
            obj,
            dict,
        ):
            for key, value in (
                obj.items()
            ):
                if (
                    str(key).lower()
                    in names
                    and isinstance(
                        value,
                        (int, float),
                    )
                ):
                    return value

            for value in (
                obj.values()
            ):
                result = (
                    self._find_numeric_value(
                        value,
                        names,
                    )
                )

                if result is not None:
                    return result

        elif isinstance(
            obj,
            list,
        ):
            for value in obj:
                result = (
                    self._find_numeric_value(
                        value,
                        names,
                    )
                )

                if result is not None:
                    return result

        return None

    # ========================================================
    # FECHAS
    # ========================================================

    @staticmethod
    def parse_datetime(
        value,
    ):
        if value is None:
            return None

        if isinstance(
            value,
            datetime,
        ):
            return value

        return datetime.fromisoformat(
            str(value)
        )

    def event_times(
        self,
        events,
    ):
        times = []

        for event in events:
            timestamp = (
                event.get(
                    "timestamp"
                )
            )

            if timestamp is None:
                continue

            try:
                parsed = (
                    self.parse_datetime(
                        timestamp
                    )
                )

            except Exception:
                continue

            times.append(
                parsed
            )

        times.sort()

        return times

    # ========================================================
    # VENTANAS TEMPORALES
    # ========================================================

    def window_metric(
        self,
        times,
        *,
        now,
        started_at,
        minutes,
    ):
        seconds = (
            minutes * 60
        )

        elapsed = (
            now - started_at
        ).total_seconds()

        # No extrapolamos una ventana
        # que todavía no fue observada completa.
        if elapsed < seconds:
            return {
                "available": False,
                "window_minutes": minutes,
                "vehicles": None,
                "veh_h": None,
            }

        start_window = (
            now.timestamp()
            - seconds
        )

        count = sum(
            1
            for timestamp in times
            if timestamp.timestamp()
            >= start_window
        )

        rate = (
            count
            * (
                60.0
                / minutes
            )
        )

        return {
            "available": True,
            "window_minutes": minutes,
            "vehicles": int(
                count
            ),
            "veh_h": round(
                float(rate),
                2,
            ),
        }

    # ========================================================
    # TH MÁXIMO ROLLING 60 MIN
    # ========================================================

    def max_hourly_volume(
        self,
        times,
        *,
        started_at,
        ended_at,
    ):
        duration = (
            ended_at
            - started_at
        ).total_seconds()

        if duration < 3600:
            return None

        if not times:
            return 0

        timestamps = [
            timestamp.timestamp()
            for timestamp in times
        ]

        left = 0
        max_count = 0

        for right in range(
            len(timestamps)
        ):
            while (
                timestamps[right]
                - timestamps[left]
                > 3600
            ):
                left += 1

            count = (
                right
                - left
                + 1
            )

            if count > max_count:
                max_count = count

        return int(
            max_count
        )

    # ========================================================
    # ANÁLISIS GENÉRICO
    # ========================================================

    def analyze(
        self,
        *,
        events,
        started_at,
        ended_at=None,
    ):
        started_at = (
            self.parse_datetime(
                started_at
            )
        )

        if started_at is None:
            return {
                "available": False,
                "reason": (
                    "No existe fecha "
                    "de inicio."
                ),
            }

        if ended_at is None:
            ended_at = (
                datetime.now()
                .astimezone()
            )

        else:
            ended_at = (
                self.parse_datetime(
                    ended_at
                )
            )

        times = (
            self.event_times(
                events
            )
        )

        duration_seconds = max(
            0.0,
            (
                ended_at
                - started_at
            ).total_seconds(),
        )

        q1 = (
            self.window_metric(
                times,
                now=ended_at,
                started_at=started_at,
                minutes=1,
            )
        )

        q5 = (
            self.window_metric(
                times,
                now=ended_at,
                started_at=started_at,
                minutes=5,
            )
        )

        q15 = (
            self.window_metric(
                times,
                now=ended_at,
                started_at=started_at,
                minutes=15,
            )
        )

        th = (
            self.window_metric(
                times,
                now=ended_at,
                started_at=started_at,
                minutes=60,
            )
        )

        max_th = (
            self.max_hourly_volume(
                times,
                started_at=(
                    started_at
                ),
                ended_at=(
                    ended_at
                ),
            )
        )

        tpda = None

        if max_th is not None:
            tpda = (
                max_th
                / self.k_vhp
            )

        return {
            "available": True,

            "duration_seconds": round(
                duration_seconds,
                3,
            ),

            "duration_hours": round(
                duration_seconds
                / 3600.0,
                4,
            ),

            "total_events": len(
                times
            ),

            "q1": q1,

            "q5": q5,

            "q15": q15,

            "th": th,

            "max_th": (
                None
                if max_th is None
                else {
                    "available": True,
                    "veh_h": float(
                        max_th
                    ),
                }
            ),

            "tpda": (
                None
                if tpda is None
                else {
                    "available": True,

                    "veh_day": round(
                        float(tpda),
                        2,
                    ),

                    "k_vhp": (
                        self.k_vhp
                    ),

                    "method": (
                        "max_th_proxy_vhp"
                    ),

                    "preliminary": True,
                }
            ),
        }

    # ========================================================
    # SESIÓN PERSISTIDA
    # ========================================================

    def analyze_session(
        self,
        session,
        events,
    ):
        if session is None:
            raise RuntimeError(
                "La sesión no existe."
            )

        started_at = (
            session.get(
                "started_at"
            )
        )

        ended_at = (
            session.get(
                "ended_at"
            )
        )

        analysis = (
            self.analyze(
                events=events,
                started_at=started_at,
                ended_at=ended_at,
            )
        )

        analysis[
            "session_id"
        ] = session.get(
            "session_id"
        )

        analysis[
            "session_status"
        ] = session.get(
            "status"
        )

        return analysis

    # ========================================================
    # INFO
    # ========================================================

    def get_info(self):
        return {
            "config_path": str(
                self.config_path
            ),

            "k_vhp": (
                self.k_vhp
            ),

            "method": (
                "VHP = max TH observado; "
                "TPDA = VHP / k"
            ),

            "preliminary": True,
        }