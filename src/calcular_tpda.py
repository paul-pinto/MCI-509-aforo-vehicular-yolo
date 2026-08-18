from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analisis de aforo y estimacion preliminar de TPDA."
    )

    parser.add_argument(
        "--events",
        type=Path,
        default=Path("eventos_trafico.csv"),
    )

    parser.add_argument(
        "--sessions",
        type=Path,
        default=Path("sesiones_aforo.csv"),
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config_tpda.json"),
    )

    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Sesion a analizar. Si se omite, usa la ultima.",
    )

    return parser.parse_args()


def leer_csv(ruta: Path):
    if not ruta.exists():
        raise FileNotFoundError(ruta)

    with ruta.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as archivo:
        return list(csv.DictReader(archivo))


def obtener_sesion(sesiones, session_id):
    if not sesiones:
        raise RuntimeError("No existen sesiones registradas.")

    if session_id is None:
        return sesiones[-1]

    for sesion in sesiones:
        if sesion["session_id"] == session_id:
            return sesion

    raise RuntimeError(
        f"No existe la sesion {session_id}"
    )


def eventos_por_sesion(filas_eventos):
    resultado = {}

    for row in filas_eventos:
        sid = row.get("session_id")

        if not sid:
            continue

        evento = {
            "timestamp": datetime.fromisoformat(
                row["timestamp"]
            ),
            "clase": row["clase"],
            "direccion": row["direccion"],
        }

        resultado.setdefault(
            sid,
            [],
        ).append(evento)

    for eventos in resultado.values():
        eventos.sort(
            key=lambda x: x["timestamp"]
        )

    return resultado


def contar_ventana(eventos, fin, segundos):
    limite = fin.timestamp() - segundos

    return sum(
        1
        for e in eventos
        if limite
        <= e["timestamp"].timestamp()
        <= fin.timestamp()
    )


def maximo_th_rodante(eventos, inicio, fin):
    """
    Busca el maximo numero de vehiculos observado
    dentro de cualquier ventana completa de 60 minutos.

    Solo se utiliza si la sesion duro al menos 3600 s.
    """

    if (
        not eventos
        or (fin - inicio).total_seconds() < 3600
    ):
        return None

    tiempos = [
        e["timestamp"].timestamp()
        for e in eventos
    ]

    izquierda = 0
    maximo = 0

    inicio_ts = inicio.timestamp()

    for derecha, t_fin in enumerate(tiempos):

        # No aceptar ventanas que todavía no tengan
        # 60 minutos completos desde el inicio real.
        if t_fin - inicio_ts < 3600:
            continue

        limite = t_fin - 3600

        while (
            izquierda <= derecha
            and tiempos[izquierda] < limite
        ):
            izquierda += 1

        cantidad = derecha - izquierda + 1

        if cantidad > maximo:
            maximo = cantidad

    # Evaluar también la ventana que termina exactamente
    # con el cierre real de la sesión.
    limite_final = fin.timestamp() - 3600

    cantidad_final = sum(
        1
        for t in tiempos
        if limite_final <= t <= fin.timestamp()
    )

    maximo = max(
        maximo,
        cantidad_final,
    )

    return maximo


def main():
    args = parse_args()

    sesiones = leer_csv(
        args.sessions
    )

    filas_eventos = leer_csv(
        args.events
    )

    eventos_sesiones = eventos_por_sesion(
        filas_eventos
    )

    sesion = obtener_sesion(
        sesiones,
        args.session,
    )

    sid = sesion["session_id"]

    inicio = datetime.fromisoformat(
        sesion["inicio"]
    )

    fin = datetime.fromisoformat(
        sesion["fin"]
    )

    duracion_seg = float(
        sesion["duracion_segundos"]
    )

    duracion_h = (
        duracion_seg / 3600.0
        if duracion_seg > 0
        else 0.0
    )

    eventos = eventos_sesiones.get(
        sid,
        [],
    )

    total = len(eventos)

    clases = Counter(
        e["clase"]
        for e in eventos
    )

    direcciones = Counter(
        e["direccion"]
        for e in eventos
    )

    print()
    print("=" * 68)
    print("AFORO VEHICULAR")
    print("=" * 68)

    print(f"Sesion:    {sid}")
    print(f"Estacion:  {sesion['estacion_id']}")
    print(f"Via:       {sesion['via']}")
    print(f"Inicio:    {inicio.isoformat()}")
    print(f"Fin:       {fin.isoformat()}")
    print(
        f"Duracion:  {duracion_seg:.0f} s "
        f"({duracion_h:.3f} h)"
    )
    print(f"Estado:    {sesion['estado']}")
    print(f"Vehiculos: {total}")

    print()
    print("Composicion vehicular:")

    for clase, cantidad in clases.most_common():
        porcentaje = (
            100.0 * cantidad / total
            if total
            else 0.0
        )

        print(
            f"  {clase:<12} "
            f"{cantidad:>5} "
            f"({porcentaje:6.2f}%)"
        )

    print()
    print("Distribucion direccional:")

    for direccion, cantidad in direcciones.items():
        porcentaje = (
            100.0 * cantidad / total
            if total
            else 0.0
        )

        print(
            f"  {direccion}: "
            f"{cantidad} "
            f"({porcentaje:.2f}%)"
        )

    print()
    print("=" * 68)
    print("METRICAS TEMPORALES")
    print("=" * 68)

    if duracion_h > 0:
        q_periodo = total / duracion_h

        print(
            f"Tasa equivalente del periodo: "
            f"{q_periodo:.1f} veh/h"
        )
    else:
        print(
            "Tasa equivalente del periodo: --"
        )

    if duracion_seg >= 900:
        n15 = contar_ventana(
            eventos,
            fin,
            900,
        )

        q15 = n15 / 0.25

        print(
            f"Ultimos 15 min:               "
            f"{n15} veh"
        )

        print(
            f"q15:                          "
            f"{q15:.1f} veh/h"
        )
    else:
        print(
            "q15:                           "
            "-- (<15 min)"
        )

    if duracion_seg >= 3600:
        th = contar_ventana(
            eventos,
            fin,
            3600,
        )

        print(
            f"TH actual (ultimos 60 min):   "
            f"{th} veh/h"
        )
    else:
        th = None

        print(
            "TH actual:                     "
            "-- (<60 min)"
        )

    print()
    print("=" * 68)
    print("VHP PROXY HISTORICO")
    print("=" * 68)

    maximos = []

    for s in sesiones:

        s_inicio = datetime.fromisoformat(
            s["inicio"]
        )

        s_fin = datetime.fromisoformat(
            s["fin"]
        )

        s_duracion = (
            s_fin - s_inicio
        ).total_seconds()

        if s_duracion < 3600:
            continue

        s_sid = s["session_id"]

        s_eventos = eventos_sesiones.get(
            s_sid,
            [],
        )

        max_th = maximo_th_rodante(
            s_eventos,
            s_inicio,
            s_fin,
        )

        if max_th is not None:
            maximos.append(
                (
                    max_th,
                    s_sid,
                )
            )

    if maximos:
        vhp_proxy, sesion_vhp = max(
            maximos,
            key=lambda x: x[0],
        )

        print(
            f"Maximo TH observado:          "
            f"{vhp_proxy} veh/h"
        )

        print(
            f"Sesion de referencia:          "
            f"{sesion_vhp}"
        )
    else:
        vhp_proxy = None

        print(
            "Maximo TH observado:          "
            "--"
        )

        print(
            "Se requiere al menos una "
            "sesion continua >= 60 min."
        )

    print()
    print("=" * 68)
    print("TPDA PRELIMINAR")
    print("=" * 68)

    if not args.config.exists():
        print(
            "No existe config_tpda.json."
        )
        return

    config = json.loads(
        args.config.read_text(
            encoding="utf-8"
        )
    )

    k = config.get(
        "k_vhp"
    )

    tipo_via = config.get(
        "tipo_via_referencia",
        "sin definir",
    )

    nota = config.get(
        "nota_k",
        "",
    )

    if vhp_proxy is None:
        print(
            "No calculado:"
        )

        print(
            "todavia no existe un volumen "
            "horario completo de referencia."
        )

        return

    if k is None:
        print(
            "No calculado: falta k_vhp."
        )
        return

    k = float(k)

    if k <= 0:
        raise ValueError(
            "k_vhp debe ser mayor que cero."
        )

    tpda_preliminar = (
        vhp_proxy / k
    )

    print(
        f"VHP proxy:                    "
        f"{vhp_proxy} veh/h"
    )

    print(
        f"k:                            "
        f"{k:.3f}"
    )

    print(
        f"Tipo de via de referencia:    "
        f"{tipo_via}"
    )

    print()
    print(
        "TPDA preliminar = VHP proxy / k"
    )

    print(
        f"TPDA preliminar = "
        f"{vhp_proxy} / {k:.3f}"
    )

    print()
    print(
        f"TPDA PRELIMINAR: "
        f"{tpda_preliminar:,.0f} veh/dia"
    )

    print()
    print(
        "IMPORTANTE: resultado estimado, "
        "no TPDA anual observado."
    )

    if nota:
        print(
            f"Nota: {nota}"
        )


if __name__ == "__main__":
    main()
