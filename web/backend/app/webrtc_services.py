import asyncio
import fractions
import time
import uuid

from av import VideoFrame

from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCRtpSender,
    RTCSessionDescription,
    VideoStreamTrack,
)


VIDEO_CLOCK_RATE = 90000


class ProcessedVideoTrack(
    VideoStreamTrack
):
    def __init__(
        self,
        video_service,
        fps=15.0,
    ):
        super().__init__()

        self.video_service = (
            video_service
        )

        self.fps = max(
            1.0,
            float(fps),
        )

        self.time_base = (
            fractions.Fraction(
                1,
                VIDEO_CLOCK_RATE,
            )
        )

        self.timestamp_step = int(
            VIDEO_CLOCK_RATE
            / self.fps
        )

        self.timestamp = 0

        self.started_at = None

    async def recv(self):
        if self.started_at is None:
            self.started_at = (
                time.monotonic()
            )

            self.timestamp = 0

        else:
            self.timestamp += (
                self.timestamp_step
            )

            target_time = (
                self.started_at
                + (
                    self.timestamp
                    / VIDEO_CLOCK_RATE
                )
            )

            delay = (
                target_time
                - time.monotonic()
            )

            if delay > 0:
                await asyncio.sleep(
                    delay
                )

        frame = None

        while frame is None:
            frame = (
                self.video_service
                .get_latest_frame()
            )

            if frame is None:
                await asyncio.sleep(
                    0.02
                )

        video_frame = (
            VideoFrame.from_ndarray(
                frame,
                format="bgr24",
            )
        )

        video_frame.pts = (
            self.timestamp
        )

        video_frame.time_base = (
            self.time_base
        )

        return video_frame


class WebRTCService:
    def __init__(
        self,
        video_service,
    ):
        self.video_service = (
            video_service
        )

        self.peer_connections = {}

        self.lock = (
            asyncio.Lock()
        )

    # ========================================================
    # CONFIGURACIÓN ICE
    # ========================================================

    def create_peer_connection(self):
        configuration = (
            RTCConfiguration(
                iceServers=[
                    RTCIceServer(
                        urls=[
                            (
                                "stun:"
                                "stun.l.google.com:"
                                "19302"
                            )
                        ]
                    ),
                ]
            )
        )

        return RTCPeerConnection(
            configuration=configuration
        )

    # ========================================================
    # H264 SI ESTÁ DISPONIBLE
    # ========================================================

    @staticmethod
    def prefer_h264(
        pc,
        sender,
    ):
        capabilities = (
            RTCRtpSender
            .getCapabilities(
                "video"
            )
        )

        h264 = [
            codec
            for codec
            in capabilities.codecs
            if codec.mimeType.lower()
            == "video/h264"
        ]

        if not h264:
            return

        transceiver = next(
            (
                item
                for item
                in pc.getTransceivers()
                if item.sender
                == sender
            ),
            None,
        )

        if transceiver is not None:
            transceiver.setCodecPreferences(
                h264
            )

    # ========================================================
    # ESPERAR ICE
    # ========================================================

    @staticmethod
    async def wait_for_ice_gathering(
        pc,
        timeout=5.0,
    ):
        if (
            pc.iceGatheringState
            == "complete"
        ):
            return

        event = asyncio.Event()

        @pc.on(
            "icegatheringstatechange"
        )
        def on_ice_gathering_state_change():
            if (
                pc.iceGatheringState
                == "complete"
            ):
                event.set()

        try:
            await asyncio.wait_for(
                event.wait(),
                timeout=timeout,
            )

        except asyncio.TimeoutError:
            pass

    # ========================================================
    # CREAR RESPUESTA
    # ========================================================

    async def create_answer(
        self,
        sdp,
        type_,
    ):
        if not (
            self.video_service
            .stream_service
            .get_info()["opened"]
        ):
            raise RuntimeError(
                "Primero configura "
                "una fuente de video."
            )

        self.video_service.start()

        peer_id = (
            str(uuid.uuid4())
        )

        pc = (
            self.create_peer_connection()
        )

        async with self.lock:
            self.peer_connections[
                peer_id
            ] = pc

        print("=" * 65)
        print("WEBRTC")
        print("=" * 65)
        print(
            "Peer:",
            peer_id,
        )

        @pc.on(
            "connectionstatechange"
        )
        async def on_connectionstatechange():
            print(
                "WebRTC",
                peer_id,
                "connection:",
                pc.connectionState,
            )

            if pc.connectionState in {
                "failed",
                "closed",
            }:
                await self.close_peer(
                    peer_id
                )

        @pc.on(
            "iceconnectionstatechange"
        )
        async def on_iceconnectionstatechange():
            print(
                "WebRTC",
                peer_id,
                "ICE:",
                pc.iceConnectionState,
            )

        offer = RTCSessionDescription(
            sdp=sdp,
            type=type_,
        )

        await pc.setRemoteDescription(
            offer
        )

        track = ProcessedVideoTrack(
            video_service=(
                self.video_service
            ),
            fps=(
                self.video_service
                .target_fps
            ),
        )

        sender = pc.addTrack(
            track
        )

        self.prefer_h264(
            pc,
            sender,
        )

        answer = (
            await pc.createAnswer()
        )

        await pc.setLocalDescription(
            answer
        )

        await self.wait_for_ice_gathering(
            pc
        )

        return {
            "peer_id": peer_id,

            "sdp": (
                pc.localDescription.sdp
            ),

            "type": (
                pc.localDescription.type
            ),
        }

    # ========================================================
    # CERRAR PEER
    # ========================================================

    async def close_peer(
        self,
        peer_id,
    ):
        async with self.lock:
            pc = (
                self.peer_connections
                .pop(
                    peer_id,
                    None,
                )
            )

        if pc is not None:
            await pc.close()

    # ========================================================
    # CERRAR TODOS
    # ========================================================

    async def close_all(self):
        async with self.lock:
            items = list(
                self.peer_connections
                .items()
            )

            self.peer_connections.clear()

        for _, pc in items:
            try:
                await pc.close()

            except Exception:
                pass

    # ========================================================
    # STATUS
    # ========================================================

    async def get_status(self):
        async with self.lock:
            peers = {
                peer_id: {
                    "connection_state": (
                        pc.connectionState
                    ),

                    "ice_state": (
                        pc.iceConnectionState
                    ),
                }
                for peer_id, pc
                in self.peer_connections.items()
            }

        return {
            "peers": peers,
            "total": len(peers),
        }