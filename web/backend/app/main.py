import csv
import io

from contextlib import (
    asynccontextmanager,
)

from datetime import datetime
from pathlib import Path

import cv2

from fastapi import (
    FastAPI,
    HTTPException,
    Query,
)

from fastapi.middleware.cors import (
    CORSMiddleware,
)

from fastapi.responses import (
    Response,
    StreamingResponse,
)

from pydantic import BaseModel


from .database_service import (
    DatabaseService,
)

from .model_service import (
    ModelService,
)

from .session_service import (
    SessionService,
)

from .source_resolver import (
    SourceResolver,
)

from .stream_service import (
    StreamService,
)

from .traffic_config import (
    TrafficConfigService,
)

from .traffic_counter import (
    TrafficCounter,
)

from .traffic_metrics import (
    TrafficMetricsService,
)

from .video_service import (
    VideoService,
)

from .webrtc_service import (
    WebRTCService,
)


# ============================================================
# RUTAS
# ============================================================

BACKEND_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

WEB_DIR = BACKEND_DIR.parent

PROJECT_ROOT = WEB_DIR.parent

DATA_DIR = (
    BACKEND_DIR
    / "data"
)

DATABASE_PATH = (
    DATA_DIR
    / "aforo.db"
)


# ============================================================
# SERVICIOS
# ============================================================

database_service = (
    DatabaseService(
        DATABASE_PATH
    )
)

model_service = (
    ModelService(
        PROJECT_ROOT
    )
)

source_resolver = (
    SourceResolver(
        PROJECT_ROOT
    )
)

stream_service = (
    StreamService()
)

traffic_config_service = (
    TrafficConfigService(
        PROJECT_ROOT
    )
)

traffic_metrics_service = (
    TrafficMetricsService(
        PROJECT_ROOT
    )
)

session_service = (
    SessionService(
        database_service=(
            database_service
        ),
        traffic_config_service=(
            traffic_config_service
        ),
    )
)

traffic_counter = (
    TrafficCounter(
        traffic_config_service=(
            traffic_config_service
        ),
        session_service=(
            session_service
        ),
        traffic_metrics_service=(
            traffic_metrics_service
        ),
    )
)

video_service = (
    VideoService(
        model_service=(
            model_service
        ),
        stream_service=(
            stream_service
        ),
        traffic_config_service=(
            traffic_config_service
        ),
        traffic_counter=(
            traffic_counter
        ),
    )
)

webrtc_service = (
    WebRTCService(
        video_service=(
            video_service
        )
    )
)


# ============================================================
# SCHEMAS
# ============================================================

class SourceRequest(
    BaseModel
):
    source: str


class WebRTCOfferRequest(
    BaseModel
):
    sdp: str
    type: str


# ============================================================
# LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    print("=" * 65)
    print("CARGANDO MODELO YOLO11")
    print("=" * 65)

    model_service.load()

    model_info = (
        model_service
        .get_info()
    )

    print(
        "Modelo cargado:",
        model_info["loaded"],
    )

    print(
        "Dispositivo:",
        model_info["device"],
    )

    print(
        "GPU:",
        model_info["gpu"],
    )

    print(
        "Ruta:",
        model_info[
            "model_path"
        ],
    )

    print("=" * 65)
    print("BASE DE DATOS")
    print("=" * 65)

    print(
        "SQLite:",
        DATABASE_PATH,
    )

    print("=" * 65)
    print(
        "INGENIERÍA DE TRÁFICO"
    )
    print("=" * 65)

    print(
        "k VHP:",
        traffic_metrics_service.k_vhp,
    )

    yield

    await webrtc_service.close_all()

    video_service.stop()
    stream_service.close()

    if session_service.is_active():
        try:
            session_service.stop(
                total_vehicles=(
                    traffic_counter
                    .get_status()[
                        "total"
                    ]
                ),
                status=(
                    "INTERRUMPIDA_SERVIDOR"
                ),
            )

        except Exception:
            pass

    print("Cerrando backend...")


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title=(
        "Sistema de Aforo Vehicular"
    ),

    description=(
        "Plataforma de detección, "
        "seguimiento, aforo vehicular "
        "y análisis de tráfico mediante "
        "YOLO11 + ByteTrack."
    ),

    version="1.1.0",

    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://aforo.paulpinto.ia.bo",
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# BASE
# ============================================================

@app.get("/")
def root():
    return {
        "name": (
            "Sistema de Aforo Vehicular"
        ),
        "status": "online",
        "version": "1.1.0",
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",

        "timestamp": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),

        "model_loaded": (
            model_service.loaded
        ),

        "database": str(
            DATABASE_PATH
        ),
    }


@app.get("/api/info")
def info():
    return {
        "project_root": str(
            PROJECT_ROOT
        ),

        "version": "1.1.0",

        "components": {
            "detector": "YOLO11 V2",
            "tracker": "ByteTrack",
            "source_resolver": "yt-dlp",
            "traffic_config": "ready",
            "traffic_counter": "ready",
            "traffic_metrics": "ready",
            "sessions": "ready",
            "database": "SQLite",
            "stream": "ready",
            "video": "ready",
            "webrtc": "ready",
            "tpda": "ready",
        },
    }


# ============================================================
# DASHBOARD
# ============================================================

@app.get("/api/dashboard")
def dashboard():
    traffic = (
        traffic_counter
        .get_status()
    )

    current_session = (
        session_service
        .current()
    )

    source = (
        stream_service
        .get_info()
    )

    video = (
        video_service
        .get_status()
    )

    model = (
        model_service
        .get_info()
    )

    station = (
        session_service
        .metadata()
    )

    events = (
        traffic_counter
        .get_events()
    )

    last_event = (
        events[-1]
        if events
        else None
    )

    return {
        "timestamp": (
            datetime.now()
            .astimezone()
            .isoformat(
                timespec="milliseconds"
            )
        ),

        "station": station,
        "session": current_session,
        "traffic": traffic,
        "source": source,
        "video": video,
        "model": model,

        "engineering": {
            "k_vhp": (
                traffic_metrics_service
                .k_vhp
            ),

            "preliminary_tpda": True,

            "method": (
                "TPDA = VHP / k; "
                "VHP aproximado mediante "
                "máximo TH observado"
            ),
        },

        "last_event": last_event,
    }


# ============================================================
# WEBRTC
# ============================================================

@app.post(
    "/api/webrtc/offer"
)
async def webrtc_offer(
    request: WebRTCOfferRequest,
):
    try:
        return (
            await webrtc_service
            .create_answer(
                sdp=request.sdp,
                type_=request.type,
            )
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


@app.get(
    "/api/webrtc/status"
)
async def webrtc_status():
    return (
        await webrtc_service
        .get_status()
    )


# ============================================================
# MODELO
# ============================================================

@app.get("/api/model")
def model_info():
    return (
        model_service
        .get_info()
    )


# ============================================================
# CONFIGURACIÓN
# ============================================================

@app.get(
    "/api/traffic/config"
)
def traffic_config():
    return (
        traffic_config_service
        .get_info()
    )


@app.get(
    "/api/traffic/metrics/config"
)
def traffic_metrics_config():
    return (
        traffic_metrics_service
        .get_info()
    )


# ============================================================
# SESIONES
# ============================================================

@app.post(
    "/api/sessions/start"
)
def start_session():
    if not (
        stream_service
        .get_info()[
            "opened"
        ]
    ):
        raise HTTPException(
            status_code=400,

            detail=(
                "Primero configura "
                "una fuente de video."
            ),
        )

    try:
        traffic_counter.reset()

        session = (
            session_service
            .start()
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    return {
        "ok": True,
        "session": session,

        "traffic": (
            traffic_counter
            .get_status()
        ),
    }


@app.post(
    "/api/sessions/stop"
)
def stop_session():
    try:
        total = (
            traffic_counter
            .get_status()[
                "total"
            ]
        )

        session = (
            session_service
            .stop(
                total_vehicles=total,
                status="FINALIZADA",
            )
        )

        events = (
            session_service
            .get_events(
                session[
                    "session_id"
                ]
            )
        )

        analysis = (
            traffic_metrics_service
            .analyze_session(
                session,
                events,
            )
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    return {
        "ok": True,
        "session": session,
        "analysis": analysis,
    }


@app.get(
    "/api/sessions/current"
)
def current_session():
    return (
        session_service
        .current()
    )


@app.get("/api/sessions")
def sessions(
    limit: int = Query(
        default=100,
        ge=1,
        le=1000,
    ),
):
    return {
        "sessions": (
            session_service
            .list_sessions(
                limit=limit
            )
        )
    }


@app.get(
    "/api/sessions/{session_id}"
)
def get_session(
    session_id: str,
):
    session = (
        session_service
        .get_session(
            session_id
        )
    )

    if session is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Sesión no encontrada."
            ),
        )

    return session


@app.get(
    "/api/sessions/{session_id}/events"
)
def session_events(
    session_id: str,
):
    session = (
        session_service
        .get_session(
            session_id
        )
    )

    if session is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Sesión no encontrada."
            ),
        )

    return {
        "session_id": session_id,

        "events": (
            session_service
            .get_events(
                session_id
            )
        ),
    }


@app.get(
    "/api/sessions/{session_id}/analysis"
)
def session_analysis(
    session_id: str,
):
    session = (
        session_service
        .get_session(
            session_id
        )
    )

    if session is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Sesión no encontrada."
            ),
        )

    events = (
        session_service
        .get_events(
            session_id
        )
    )

    return (
        traffic_metrics_service
        .analyze_session(
            session,
            events,
        )
    )


# ============================================================
# EXPORTAR CSV
# ============================================================

@app.get(
    "/api/sessions/"
    "{session_id}/export.csv"
)
def export_session_csv(
    session_id: str,
):
    session = (
        session_service
        .get_session(
            session_id
        )
    )

    if session is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Sesión no encontrada."
            ),
        )

    events = (
        session_service
        .get_events(
            session_id
        )
    )

    output = io.StringIO()

    fieldnames = [
        "session_id",
        "timestamp",
        "track_id",
        "class_id",
        "class_name",
        "direction",
        "confidence",
        "start_distance_px",
        "end_distance_px",
        "displacement_px",
        "normal_progress_px",
    ]

    writer = csv.DictWriter(
        output,
        fieldnames=fieldnames,
        extrasaction="ignore",
    )

    writer.writeheader()

    for event in events:
        row = dict(event)

        row[
            "session_id"
        ] = session_id

        writer.writerow(
            row
        )

    content = (
        "\ufeff"
        + output.getvalue()
    )

    filename = (
        f"aforo_{session_id}.csv"
    )

    return Response(
        content=content,

        media_type=(
            "text/csv; "
            "charset=utf-8"
        ),

        headers={
            "Content-Disposition": (
                f'attachment; '
                f'filename="{filename}"'
            )
        },
    )


# ============================================================
# TRÁFICO
# ============================================================

@app.get(
    "/api/traffic/status"
)
def traffic_status():
    return (
        traffic_counter
        .get_status()
    )


@app.get(
    "/api/traffic/metrics"
)
def traffic_metrics():
    if not (
        session_service
        .is_active()
    ):
        return {
            "available": False,

            "reason": (
                "No existe "
                "una sesión activa."
            ),
        }

    return (
        traffic_counter
        .get_metrics()
    )


@app.get(
    "/api/traffic/events"
)
def traffic_events():
    return {
        "events": (
            traffic_counter
            .get_events()
        )
    }


# ============================================================
# FUENTE
# ============================================================

@app.post("/api/source")
def set_source(
    request: SourceRequest,
):
    if session_service.is_active():
        raise HTTPException(
            status_code=400,

            detail=(
                "Detén la sesión de aforo "
                "antes de cambiar la fuente."
            ),
        )

    video_service.stop()
    stream_service.close()

    try:
        resolution = (
            source_resolver
            .resolve(
                request.source
            )
        )

        print("=" * 65)
        print("FUENTE DE VIDEO")
        print("=" * 65)

        print(
            "Original:",
            resolution[
                "original_source"
            ],
        )

        print(
            "Resolver:",
            resolution[
                "resolver"
            ],
        )

        print("Abriendo stream...")

        metadata = {
            key: value
            for key, value
            in resolution.items()
            if key not in {
                "original_source",
                "resolved_source",
                "resolver",
            }
        }

        stream_service.open(
            resolution[
                "resolved_source"
            ],

            original_source=(
                resolution[
                    "original_source"
                ]
            ),

            resolver=(
                resolution[
                    "resolver"
                ]
            ),

            metadata=metadata,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    traffic_counter.reset()

    return {
        "ok": True,

        **stream_service
        .get_info(),
    }


@app.get("/api/source")
def get_source():
    return (
        stream_service
        .get_info()
    )


@app.delete("/api/source")
def close_source():
    if session_service.is_active():
        raise HTTPException(
            status_code=400,

            detail=(
                "Detén la sesión de aforo "
                "antes de cerrar la fuente."
            ),
        )

    video_service.stop()
    stream_service.close()

    return {
        "ok": True,
        "opened": False,
        "source": None,
    }


# ============================================================
# FRAME
# ============================================================

@app.get("/api/frame")
def get_frame():
    if not model_service.loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                "Modelo no cargado."
            ),
        )

    try:
        frame = (
            stream_service.read()
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    result = (
        model_service.model.predict(
            source=frame,
            conf=0.35,
            imgsz=640,
            device=(
                model_service.device
            ),
            verbose=False,
        )[0]
    )

    annotated = result.plot()

    height, width = (
        annotated.shape[:2]
    )

    roi, line = (
        traffic_config_service.geometry(
            width,
            height,
        )
    )

    cv2.polylines(
        annotated,
        [
            roi.reshape(
                (-1, 1, 2)
            )
        ],
        True,
        (0, 255, 255),
        3,
    )

    cv2.line(
        annotated,
        tuple(line[0]),
        tuple(line[1]),
        (255, 0, 255),
        4,
    )

    ok, buffer = cv2.imencode(
        ".jpg",
        annotated,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            80,
        ],
    )

    if not ok:
        raise HTTPException(
            status_code=500,
            detail=(
                "No se pudo codificar "
                "el frame."
            ),
        )

    return Response(
        content=(
            buffer.tobytes()
        ),
        media_type="image/jpeg",
    )


# ============================================================
# MJPEG FALLBACK
# ============================================================

@app.get("/api/video")
def video():
    if not model_service.loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                "Modelo no cargado."
            ),
        )

    if not (
        stream_service
        .get_info()[
            "opened"
        ]
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Primero configura "
                "una fuente."
            ),
        )

    video_service.start()

    return StreamingResponse(
        video_service.generate(),

        media_type=(
            "multipart/"
            "x-mixed-replace;"
            " boundary=frame"
        ),
    )


@app.post(
    "/api/video/stop"
)
def stop_video():
    video_service.stop()

    return {
        "ok": True,

        **video_service
        .get_status(),
    }


@app.get(
    "/api/video/status"
)
def video_status():
    return (
        video_service
        .get_status()
    )