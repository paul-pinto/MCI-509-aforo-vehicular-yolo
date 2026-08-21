import math
import threading

from collections import (
    Counter,
    defaultdict,
    deque,
)

from datetime import datetime

import cv2


class TrafficCounter:
    def __init__(
        self,
        traffic_config_service,
        session_service,
        traffic_metrics_service,
    ):
        self.traffic_config_service = (
            traffic_config_service
        )

        self.session_service = (
            session_service
        )

        self.traffic_metrics_service = (
            traffic_metrics_service
        )

        # ====================================================
        # PARÁMETROS
        # ====================================================

        self.histeresis_linea_px = 18

        self.min_frames_track = 5

        self.min_desplazamiento_px = 25

        self.min_progreso_normal_px = 25

        self.historial_track = 12

        self.class_names = {
            0: "auto",
            1: "triciclo",
            2: "bus",
            3: "camion",
            4: "moto",
            5: "furgoneta",
        }

        self.lock = (
            threading.RLock()
        )

        self.reset()

    # ========================================================
    # RESET
    # ========================================================

    def reset(self):
        with self.lock:

            self.histories = (
                defaultdict(
                    lambda: deque(
                        maxlen=(
                            self.historial_track
                        )
                    )
                )
            )

            self.ages = (
                Counter()
            )

            self.last_stable_side = {}

            self.counted_ids = set()

            self.class_counts = (
                Counter()
            )

            self.direction_counts = (
                Counter()
            )

            self.events = []

    # ========================================================
    # GEOMETRÍA
    # ========================================================

    @staticmethod
    def signed_distance(
        point,
        line_start,
        line_end,
    ):
        px, py = point
        ax, ay = line_start
        bx, by = line_end

        vx = bx - ax
        vy = by - ay

        length = math.hypot(
            vx,
            vy,
        )

        if length == 0:
            return 0.0

        return (
            vx * (py - ay)
            - vy * (px - ax)
        ) / length

    def stable_side(
        self,
        distance,
    ):
        if (
            distance
            > self.histeresis_linea_px
        ):
            return 1

        if (
            distance
            < -self.histeresis_linea_px
        ):
            return -1

        return 0

    @staticmethod
    def point_in_roi(
        point,
        roi,
    ):
        return (
            cv2.pointPolygonTest(
                roi.reshape(
                    (-1, 1, 2)
                ),
                (
                    float(
                        point[0]
                    ),
                    float(
                        point[1]
                    ),
                ),
                False,
            )
            >= 0
        )

    # ========================================================
    # PROCESAMIENTO
    # ========================================================

    def process(
        self,
        result,
        frame_shape,
    ):
        if not (
            self.session_service
            .is_active()
        ):
            return

        height, width = (
            frame_shape[:2]
        )

        roi, line = (
            self.traffic_config_service
            .geometry(
                width,
                height,
            )
        )

        if (
            result.boxes is None
            or result.boxes.id is None
        ):
            return

        boxes = (
            result.boxes.xyxy
            .detach()
            .cpu()
            .numpy()
        )

        track_ids = (
            result.boxes.id
            .int()
            .cpu()
            .tolist()
        )

        classes = (
            result.boxes.cls
            .int()
            .cpu()
            .tolist()
        )

        confidences = (
            result.boxes.conf
            .detach()
            .cpu()
            .tolist()
        )

        line_start = tuple(
            line[0]
        )

        line_end = tuple(
            line[1]
        )

        with self.lock:

            for (
                box,
                track_id,
                class_id,
                confidence,
            ) in zip(
                boxes,
                track_ids,
                classes,
                confidences,
            ):
                x1, y1, x2, y2 = box

                point = (
                    float(
                        (x1 + x2)
                        / 2.0
                    ),
                    float(y2),
                )

                if not (
                    self.point_in_roi(
                        point,
                        roi,
                    )
                ):
                    continue

                self.ages[
                    track_id
                ] += 1

                self.histories[
                    track_id
                ].append(
                    point
                )

                distance_now = (
                    self.signed_distance(
                        point,
                        line_start,
                        line_end,
                    )
                )

                side_now = (
                    self.stable_side(
                        distance_now
                    )
                )

                if side_now == 0:
                    continue

                previous_side = (
                    self.last_stable_side
                    .get(
                        track_id
                    )
                )

                if previous_side is None:
                    self.last_stable_side[
                        track_id
                    ] = side_now

                    continue

                if (
                    previous_side
                    == side_now
                ):
                    continue

                if (
                    track_id
                    in self.counted_ids
                ):
                    self.last_stable_side[
                        track_id
                    ] = side_now

                    continue

                history = (
                    self.histories[
                        track_id
                    ]
                )

                if (
                    self.ages[
                        track_id
                    ]
                    < self.min_frames_track
                ):
                    self.last_stable_side[
                        track_id
                    ] = side_now

                    continue

                if len(history) < 2:
                    self.last_stable_side[
                        track_id
                    ] = side_now

                    continue

                point_start = (
                    history[0]
                )

                point_end = (
                    history[-1]
                )

                displacement = (
                    math.hypot(
                        point_end[0]
                        - point_start[0],

                        point_end[1]
                        - point_start[1],
                    )
                )

                if (
                    displacement
                    < self.min_desplazamiento_px
                ):
                    self.last_stable_side[
                        track_id
                    ] = side_now

                    continue

                distance_start = (
                    self.signed_distance(
                        point_start,
                        line_start,
                        line_end,
                    )
                )

                distance_end = (
                    self.signed_distance(
                        point_end,
                        line_start,
                        line_end,
                    )
                )

                normal_progress = abs(
                    distance_end
                    - distance_start
                )

                if (
                    normal_progress
                    < self.min_progreso_normal_px
                ):
                    self.last_stable_side[
                        track_id
                    ] = side_now

                    continue

                # ============================================
                # CRUCE VÁLIDO
                # ============================================

                direction = (
                    "A"
                    if previous_side
                    < side_now
                    else "B"
                )

                self.counted_ids.add(
                    track_id
                )

                self.direction_counts[
                    direction
                ] += 1

                class_name = (
                    self.class_names
                    .get(
                        int(
                            class_id
                        ),
                        str(
                            class_id
                        ),
                    )
                )

                self.class_counts[
                    class_name
                ] += 1

                event = {
                    "timestamp": (
                        datetime.now()
                        .astimezone()
                        .isoformat(
                            timespec=(
                                "milliseconds"
                            )
                        )
                    ),

                    "track_id": int(
                        track_id
                    ),

                    "clase_id": int(
                        class_id
                    ),

                    "clase": (
                        class_name
                    ),

                    "direccion": (
                        direction
                    ),

                    "confianza": round(
                        float(
                            confidence
                        ),
                        6,
                    ),

                    "distancia_inicio_px": round(
                        float(
                            distance_start
                        ),
                        3,
                    ),

                    "distancia_final_px": round(
                        float(
                            distance_end
                        ),
                        3,
                    ),

                    "desplazamiento_px": round(
                        float(
                            displacement
                        ),
                        3,
                    ),

                    "progreso_normal_px": round(
                        float(
                            normal_progress
                        ),
                        3,
                    ),
                }

                event_id = (
                    self.session_service
                    .record_event(
                        event
                    )
                )

                event[
                    "event_id"
                ] = event_id

                event[
                    "session_id"
                ] = (
                    self.session_service
                    .active_session_id
                )

                self.events.append(
                    event
                )

                self.last_stable_side[
                    track_id
                ] = side_now

    # ========================================================
    # MÉTRICAS TEMPORALES
    # ========================================================

    def get_metrics(self):
        current = (
            self.session_service
            .current()
        )

        if not current[
            "active"
        ]:
            return {
                "available": False,
                "reason": (
                    "No existe una "
                    "sesión activa."
                ),
            }

        session = (
            current[
                "session"
            ]
        )

        return (
            self.traffic_metrics_service
            .analyze(
                events=self.events,

                started_at=(
                    session[
                        "started_at"
                    ]
                ),

                ended_at=None,
            )
        )

    # ========================================================
    # ESTADO
    # ========================================================

    def get_status(self):
        with self.lock:

            active = (
                self.session_service
                .is_active()
            )

            metrics = (
                self.get_metrics()
                if active
                else {
                    "available": False,
                    "reason": (
                        "No existe "
                        "una sesión activa."
                    ),
                }
            )

            return {
                "aforo_activo": (
                    active
                ),

                "session_id": (
                    self.session_service
                    .active_session_id
                ),

                "total": len(
                    self.counted_ids
                ),

                "direcciones": {
                    "A": (
                        self.direction_counts[
                            "A"
                        ]
                    ),

                    "B": (
                        self.direction_counts[
                            "B"
                        ]
                    ),
                },

                "clases": dict(
                    self.class_counts
                ),

                "eventos": len(
                    self.events
                ),

                "metricas": (
                    metrics
                ),
            }

    def get_events(self):
        with self.lock:
            return list(
                self.events
            )

    # ========================================================
    # OVERLAY
    # ========================================================

    def draw_overlay(
        self,
        frame,
    ):
        status = (
            self.get_status()
        )

        overlay = (
            frame.copy()
        )

        cv2.rectangle(
            overlay,
            (0, 0),
            (
                frame.shape[1],
                145,
            ),
            (0, 0, 0),
            -1,
        )

        cv2.addWeighted(
            overlay,
            0.70,
            frame,
            0.30,
            0,
            frame,
        )

        if status[
            "aforo_activo"
        ]:
            state_text = (
                "AFORO EN CURSO"
            )

            state_color = (
                0,
                255,
                0,
            )

        else:
            state_text = (
                "AFORO DETENIDO"
            )

            state_color = (
                0,
                180,
                255,
            )

        cv2.putText(
            frame,
            state_text,
            (20, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            state_color,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            (
                f"TOTAL: "
                f"{status['total']}"
                f"   "
                f"A: "
                f"{status['direcciones']['A']}"
                f"   "
                f"B: "
                f"{status['direcciones']['B']}"
            ),
            (20, 57),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        classes_text = (
            "  ".join(
                f"{name}: {count}"
                for name, count
                in status[
                    "clases"
                ].items()
            )
        )

        if not classes_text:
            classes_text = (
                "Esperando cruces..."
            )

        cv2.putText(
            frame,
            classes_text,
            (20, 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # ====================================================
        # q15 / TH / TPDA
        # ====================================================

        metrics = (
            status.get(
                "metricas",
                {}
            )
        )

        if metrics.get(
            "available"
        ):
            q15 = (
                metrics.get(
                    "q15",
                    {}
                )
            )

            th = (
                metrics.get(
                    "th",
                    {}
                )
            )

            tpda = (
                metrics.get(
                    "tpda"
                )
            )

            q15_text = (
                f"{q15['veh_h']:.0f} veh/h"
                if q15.get(
                    "available"
                )
                else "pendiente"
            )

            th_text = (
                f"{th['veh_h']:.0f} veh/h"
                if th.get(
                    "available"
                )
                else "pendiente"
            )

            tpda_text = (
                f"{tpda['veh_day']:.0f} veh/dia"
                if tpda
                else "pendiente"
            )

            metrics_text = (
                f"q15: {q15_text}   "
                f"TH: {th_text}   "
                f"TPDA: {tpda_text}"
            )

        else:
            metrics_text = (
                "q15: --   "
                "TH: --   "
                "TPDA: --"
            )

        cv2.putText(
            frame,
            metrics_text,
            (20, 118),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 220, 80),
            1,
            cv2.LINE_AA,
        )