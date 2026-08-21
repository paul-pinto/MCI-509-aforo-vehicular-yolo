import json
import os
import subprocess
import sys

from pathlib import Path
from urllib.parse import urlparse


class SourceResolver:
    def __init__(
        self,
        project_root: Path,
    ):
        self.project_root = project_root

    # ========================================================
    # DETECCIÓN DE YOUTUBE
    # ========================================================

    @staticmethod
    def is_youtube_url(
        source: str,
    ):
        try:
            parsed = urlparse(source)

            host = (
                parsed.netloc
                .lower()
                .split(":")[0]
            )

            return (
                host == "youtube.com"
                or host.endswith(".youtube.com")
                or host == "youtu.be"
                or host.endswith(".youtu.be")
            )

        except Exception:
            return False

    # ========================================================
    # RESOLVER GENERAL
    # ========================================================

    def resolve(
        self,
        source: str,
    ):
        source = source.strip()

        if not source:
            raise RuntimeError(
                "La fuente está vacía."
            )

        if self.is_youtube_url(source):
            return self.resolve_youtube(
                source
            )

        return {
            "original_source": source,
            "resolved_source": source,
            "resolver": "direct",
            "format_id": None,
        }

    # ========================================================
    # COMANDO BASE YT-DLP
    # ========================================================

    def build_base_command(self):
        command = [
            sys.executable,
            "-m",
            "yt_dlp",

            "--no-playlist",

            "--js-runtimes",
            "node",

            "--remote-components",
            "ejs:github",

            "--extractor-args",
            "youtube:player_client=default,web_safari",
        ]

        # ====================================================
        # COOKIES DESDE ARCHIVO
        # ====================================================

        cookies_path = os.environ.get(
            "YTDLP_COOKIES_PATH"
        )

        if cookies_path:
            path = Path(cookies_path)

            if path.exists():
                command.extend(
                    [
                        "--cookies",
                        str(path),
                    ]
                )

        # ====================================================
        # COOKIES DESDE NAVEGADOR
        # ====================================================

        cookies_browser = os.environ.get(
            "YTDLP_COOKIES_FROM_BROWSER"
        )

        if cookies_browser:
            command.extend(
                [
                    "--cookies-from-browser",
                    cookies_browser,
                ]
            )

        return command

    # ========================================================
    # EJECUTAR YT-DLP
    # ========================================================

    @staticmethod
    def run_command(
        command,
        timeout=90,
    ):
        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )

        except subprocess.TimeoutExpired:
            raise RuntimeError(
                "yt-dlp tardó demasiado "
                "en resolver la URL de YouTube."
            )

        return process

    # ========================================================
    # OBTENER METADATOS Y FORMATOS
    # ========================================================

    def get_youtube_info(
        self,
        url: str,
    ):
        command = (
            self.build_base_command()
            + [
                "--dump-single-json",
                "--skip-download",
                url,
            ]
        )

        process = self.run_command(
            command,
            timeout=120,
        )

        if process.returncode != 0:
            error = (
                process.stderr.strip()
                or process.stdout.strip()
                or "yt-dlp no pudo consultar los formatos."
            )

            raise RuntimeError(error)

        try:
            info = json.loads(
                process.stdout
            )

        except json.JSONDecodeError:
            raise RuntimeError(
                "yt-dlp devolvió metadatos "
                "que no pudieron interpretarse."
            )

        return info

    # ========================================================
    # ELEGIR FORMATO
    # ========================================================

    @staticmethod
    def choose_format(
        formats,
    ):
        candidates = []

        for fmt in formats:
            format_id = fmt.get(
                "format_id"
            )

            url = fmt.get(
                "url"
            )

            vcodec = fmt.get(
                "vcodec"
            )

            protocol = (
                fmt.get("protocol")
                or ""
            ).lower()

            height = (
                fmt.get("height")
                or 0
            )

            fps = (
                fmt.get("fps")
                or 0
            )

            if not format_id:
                continue

            if not url:
                continue

            # Solo formatos que tengan video
            if (
                not vcodec
                or vcodec == "none"
            ):
                continue

            # Evitamos resoluciones enormes
            if height and height > 1080:
                continue

            score = 0

            # Preferir HLS si todavía está disponible
            if (
                "m3u8" in protocol
                or ".m3u8" in url
            ):
                score += 1000

            # Luego HTTP directo
            elif (
                protocol.startswith("http")
            ):
                score += 700

            # Preferencia por resoluciones razonables
            if height:
                if height <= 720:
                    score += 300
                elif height <= 1080:
                    score += 200

                score += min(
                    height,
                    1080,
                ) / 10

            if fps:
                score += min(
                    fps,
                    60,
                )

            candidates.append(
                (
                    score,
                    fmt,
                )
            )

        if not candidates:
            raise RuntimeError(
                "YouTube no devolvió ningún formato "
                "de video reproducible."
            )

        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return candidates[0][1]

    # ========================================================
    # RESOLVER URL CON FORMAT_ID
    # ========================================================

    def get_format_url(
        self,
        url: str,
        format_id: str,
    ):
        command = (
            self.build_base_command()
            + [
                "--get-url",
                "-f",
                format_id,
                url,
            ]
        )

        process = self.run_command(
            command
        )

        if process.returncode != 0:
            error = (
                process.stderr.strip()
                or process.stdout.strip()
                or (
                    "No se pudo obtener "
                    "la URL del formato seleccionado."
                )
            )

            raise RuntimeError(error)

        urls = [
            line.strip()
            for line in process.stdout.splitlines()
            if line.strip().startswith(
                (
                    "http://",
                    "https://",
                )
            )
        ]

        if not urls:
            raise RuntimeError(
                "yt-dlp no devolvió "
                "la URL del stream."
            )

        return urls[0]

    # ========================================================
    # YOUTUBE
    # ========================================================

    def resolve_youtube(
        self,
        url: str,
    ):
        print("=" * 65)
        print("RESOLVIENDO YOUTUBE")
        print("=" * 65)

        print(
            "URL:",
            url,
        )

        print(
            "Runtime JS: Node"
        )

        print(
            "Consultando formatos disponibles..."
        )

        info = self.get_youtube_info(
            url
        )

        formats = info.get(
            "formats",
            []
        )

        if not formats:
            raise RuntimeError(
                "YouTube no devolvió "
                "formatos disponibles."
            )

        print(
            "Formatos encontrados:",
            len(formats),
        )

        selected = self.choose_format(
            formats
        )

        format_id = selected.get(
            "format_id"
        )

        height = selected.get(
            "height"
        )

        fps = selected.get(
            "fps"
        )

        protocol = selected.get(
            "protocol"
        )

        ext = selected.get(
            "ext"
        )

        print(
            "Formato seleccionado:",
            format_id,
        )

        print(
            "Resolución:",
            height,
        )

        print(
            "FPS:",
            fps,
        )

        print(
            "Protocolo:",
            protocol,
        )

        print(
            "Contenedor:",
            ext,
        )

        resolved_url = (
            self.get_format_url(
                url,
                format_id,
            )
        )

        print(
            "Stream resuelto correctamente."
        )

        return {
            "original_source": url,
            "resolved_source": resolved_url,
            "resolver": "yt-dlp",
            "format_id": format_id,
            "height": height,
            "fps": fps,
            "protocol": protocol,
        }