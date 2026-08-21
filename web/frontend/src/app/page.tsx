"use client";

import {
  useEffect,
  useMemo,
  useState,
} from "react";


const API =
  process.env.NEXT_PUBLIC_API_URL ??
  "http://127.0.0.1:8000";


const GITHUB_URL =
  "https://github.com/paul-pinto/MCI-509-aforo-vehicular-yolo";


type SessionData = {
  session_id: string;
  station_id: string;
  road: string;
  reference: string;
  city: string;
  started_at: string;
  ended_at: string | null;
  duration_seconds: number | null;
  total_vehicles: number;
  status: string;
};


type TrafficMetrics = {
  available: boolean;
  duration_seconds?: number;
  duration_hours?: number;
  total_events?: number;

  q1?: {
    available: boolean;
    vehicles: number | null;
    veh_h: number | null;
  };

  q5?: {
    available: boolean;
    vehicles: number | null;
    veh_h: number | null;
  };

  q15?: {
    available: boolean;
    vehicles: number | null;
    veh_h: number | null;
  };

  th?: {
    available: boolean;
    vehicles: number | null;
    veh_h: number | null;
  };

  max_th?: {
    available: boolean;
    veh_h: number;
  } | null;

  tpda?: {
    available: boolean;
    veh_day: number;
    k_vhp: number;
    preliminary: boolean;
  } | null;
};


type TrafficStatus = {
  aforo_activo: boolean;
  session_id: string | null;
  total: number;

  direcciones: {
    A: number;
    B: number;
  };

  clases: Record<
    string,
    number
  >;

  eventos: number;

  metricas?: TrafficMetrics;
};


type SourceInfo = {
  opened: boolean;
  source: string | null;
  resolver?: string | null;
  type?: string;
  fps_source?: number | null;
  width?: number | null;
  height?: number | null;
  format_id?: string | null;
  protocol?: string | null;
};


type ModelInfo = {
  loaded: boolean;
  model_path: string;
  device: string | null;
  cuda_available: boolean;
  gpu: string | null;

  classes: Record<
    string,
    string
  >;
};


type VideoStatus = {
  running: boolean;
  frames: number;
  fps: number;
};


type LastEvent = {
  timestamp: string;
  track_id: number;
  clase_id: number;
  clase: string;
  direccion: string;
  confianza: number;
};


type DashboardData = {
  timestamp: string;

  station: {
    station_id: string;
    road: string;
    reference: string;
    city: string;
  };

  session: {
    active: boolean;
    session: SessionData | null;
  };

  traffic: TrafficStatus;

  source: SourceInfo;

  video: VideoStatus;

  model: ModelInfo;

  engineering: {
    k_vhp: number;
    preliminary_tpda: boolean;
    method: string;
  };

  last_event: LastEvent | null;
};


type SessionsResponse = {
  sessions: SessionData[];
};


export default function Home() {
  const [
    mounted,
    setMounted,
  ] =
    useState(false);


  const [
    now,
    setNow,
  ] =
    useState(0);


  const [
    source,
    setSource,
  ] =
    useState(
      "https://www.youtube.com/watch?v=NfsyRx50gAI"
    );


  const [
    dashboard,
    setDashboard,
  ] =
    useState<DashboardData | null>(
      null
    );


  const [
    sessions,
    setSessions,
  ] =
    useState<SessionData[]>(
      []
    );


  const [
    loading,
    setLoading,
  ] =
    useState(false);


  const [
    message,
    setMessage,
  ] =
    useState("");


  // =========================================================
  // MOUNT
  // =========================================================

  useEffect(() => {
    setMounted(true);
    setNow(Date.now());
  }, []);


  // =========================================================
  // RELOJ
  // =========================================================

  useEffect(() => {
    if (!mounted) {
      return;
    }

    const timer =
      window.setInterval(
        () => {
          setNow(
            Date.now()
          );
        },
        1000
      );

    return () =>
      window.clearInterval(
        timer
      );
  }, [mounted]);


  // =========================================================
  // DASHBOARD
  // =========================================================

  async function refreshDashboard() {
    try {
      const response =
        await fetch(
          `${API}/api/dashboard`,
          {
            cache: "no-store",
          }
        );

      if (!response.ok) {
        return;
      }

      const data =
        await response.json();

      setDashboard(
        data
      );

      if (
        data.source?.source
        && !source
      ) {
        setSource(
          data.source.source
        );
      }

    } catch {
      // Backend puede estar reiniciando.
    }
  }


  // =========================================================
  // HISTORIAL
  // =========================================================

  async function refreshSessions() {
    try {
      const response =
        await fetch(
          `${API}/api/sessions?limit=8`,
          {
            cache: "no-store",
          }
        );

      if (!response.ok) {
        return;
      }

      const data:
        SessionsResponse =
        await response.json();

      setSessions(
        data.sessions ?? []
      );

    } catch {
      //
    }
  }


  // =========================================================
  // POLLING
  // =========================================================

  useEffect(() => {
    if (!mounted) {
      return;
    }

    refreshDashboard();
    refreshSessions();

    const dashboardTimer =
      window.setInterval(
        refreshDashboard,
        1000
      );

    const sessionsTimer =
      window.setInterval(
        refreshSessions,
        10000
      );

    return () => {
      window.clearInterval(
        dashboardTimer
      );

      window.clearInterval(
        sessionsTimer
      );
    };
  }, [mounted]);


  // =========================================================
  // CONFIGURAR FUENTE
  // =========================================================

  async function configureSource() {
    if (!source.trim()) {
      setMessage(
        "Introduce una fuente de video."
      );

      return;
    }

    setLoading(true);
    setMessage("");

    try {
      const response =
        await fetch(
          `${API}/api/source`,
          {
            method: "POST",

            headers: {
              "Content-Type":
                "application/json",
            },

            body:
              JSON.stringify({
                source:
                  source.trim(),
              }),
          }
        );

      const data =
        await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail ??
          "No se pudo configurar la fuente."
        );
      }

      setMessage(
        "Fuente configurada correctamente."
      );

      await refreshDashboard();

    } catch (error) {
      setMessage(
        error instanceof Error
          ? error.message
          : "Error configurando la fuente."
      );

    } finally {
      setLoading(false);
    }
  }


  // =========================================================
  // INICIAR SESIÓN
  // =========================================================

  async function startSession() {
    setLoading(true);
    setMessage("");

    try {
      const response =
        await fetch(
          `${API}/api/sessions/start`,
          {
            method: "POST",
          }
        );

      const data =
        await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail ??
          "No se pudo iniciar el aforo."
        );
      }

      setMessage(
        `Aforo iniciado: ${
          data.session.session_id
        }`
      );

      await refreshDashboard();
      await refreshSessions();

    } catch (error) {
      setMessage(
        error instanceof Error
          ? error.message
          : "Error iniciando el aforo."
      );

    } finally {
      setLoading(false);
    }
  }


  // =========================================================
  // DETENER SESIÓN
  // =========================================================

  async function stopSession() {
    setLoading(true);
    setMessage("");

    try {
      const response =
        await fetch(
          `${API}/api/sessions/stop`,
          {
            method: "POST",
          }
        );

      const data =
        await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail ??
          "No se pudo detener el aforo."
        );
      }

      setMessage(
        `Sesión finalizada: ${
          data.session.session_id
        }`
      );

      await refreshDashboard();
      await refreshSessions();

    } catch (error) {
      setMessage(
        error instanceof Error
          ? error.message
          : "Error deteniendo el aforo."
      );

    } finally {
      setLoading(false);
    }
  }


  // =========================================================
  // ESTADOS DERIVADOS
  // =========================================================

  const traffic =
    dashboard?.traffic;


  const currentSession =
    dashboard?.session
      ?.session;


  const sourceInfo =
    dashboard?.source;


  const model =
    dashboard?.model;


  const video =
    dashboard?.video;


  const station =
    dashboard?.station;


  const metrics =
    traffic?.metricas;


  const active =
    dashboard?.session
      ?.active === true;


  const sourceOpened =
    sourceInfo?.opened === true;


  const videoUrl =
    useMemo(
      () => {
        if (
          !mounted
          || !sourceOpened
        ) {
          return null;
        }

        return (
          `${API}/api/video`
        );
      },
      [
        mounted,
        sourceOpened,
      ]
    );


  const elapsed =
    currentSession
    && active
    && now > 0
      ? formatDuration(
          Math.max(
            0,
            (
              now
              - new Date(
                currentSession
                  .started_at
              ).getTime()
            ) / 1000
          )
        )
      : currentSession
        ?.duration_seconds
        ? formatDuration(
            currentSession
              .duration_seconds
          )
        : "--:--:--";


  const configureDisabled =
    !mounted
    || loading
    || active;


  const startDisabled =
    !mounted
    || loading
    || !sourceOpened
    || active;


  const stopDisabled =
    !mounted
    || loading
    || !active;


  // =========================================================
  // UI
  // =========================================================

  return (
    <main
      className="
        min-h-screen
        bg-[#020617]
        text-slate-100
      "
    >
      <div
        className="
          mx-auto
          max-w-[1500px]
          px-5
          py-6
          lg:px-8
        "
      >

        {/* ================================================= */}
        {/* HEADER */}
        {/* ================================================= */}

        <header
          className="
            mb-6
            flex
            flex-col
            justify-between
            gap-5
            lg:flex-row
            lg:items-center
          "
        >
          <div>
            <div
              className="
                mb-2
                flex
                flex-wrap
                items-center
                gap-3
              "
            >
              <span
                className="
                  text-xs
                  font-bold
                  uppercase
                  tracking-[0.24em]
                  text-sky-400
                "
              >
                Sistema de Aforo Vehicular
              </span>

              <span
                className="
                  rounded-full
                  border
                  border-slate-700
                  bg-slate-900
                  px-3
                  py-1
                  text-[11px]
                  text-slate-400
                "
              >
                YOLO11 V2 + ByteTrack
              </span>
            </div>

            <h1
              className="
                text-3xl
                font-bold
                tracking-tight
                lg:text-4xl
              "
            >
              {station?.station_id
                ?? "Estación de aforo"}
            </h1>

            <p
              className="
                mt-2
                text-sm
                text-slate-400
              "
            >
              {station
                ? (
                  `${station.road} · `
                  + `${station.reference} · `
                  + `${station.city}`
                )
                : (
                  "Cargando información "
                  + "de la estación..."
                )
              }
            </p>
          </div>


          <div
            className={`
              flex
              items-center
              gap-3
              rounded-2xl
              border
              px-5
              py-3

              ${
                active
                  ? (
                    "border-emerald-500/30 "
                    + "bg-emerald-500/10"
                  )
                  : (
                    "border-slate-800 "
                    + "bg-slate-900"
                  )
              }
            `}
          >
            <div
              className={`
                h-2.5
                w-2.5
                rounded-full

                ${
                  active
                    ? (
                      "bg-emerald-400 "
                      + "shadow-[0_0_14px_#34d399]"
                    )
                    : "bg-slate-600"
                }
              `}
            />

            <div>
              <p
                className="
                  text-[10px]
                  uppercase
                  tracking-wider
                  text-slate-500
                "
              >
                Estado
              </p>

              <p
                className={`
                  text-sm
                  font-semibold

                  ${
                    active
                      ? "text-emerald-300"
                      : "text-slate-300"
                  }
                `}
              >
                {active
                  ? "AFORO EN CURSO"
                  : "AFORO DETENIDO"}
              </p>
            </div>
          </div>
        </header>


        {/* ================================================= */}
        {/* BARRA DE CONTROL */}
        {/* ================================================= */}

        <section
          className="
            mb-6
            rounded-2xl
            border
            border-slate-800
            bg-slate-900/80
            p-5
          "
        >
          <div
            className="
              flex
              flex-col
              gap-3
              xl:flex-row
            "
          >
            <input
              value={source}

              onChange={
                (
                  event
                ) =>
                  setSource(
                    event.target.value
                  )
              }

              placeholder={
                "URL YouTube, HLS, "
                + "archivo local o cámara IP"
              }

              disabled={
                active
              }

              className="
                min-w-0
                flex-1
                rounded-xl
                border
                border-slate-700
                bg-slate-950
                px-4
                py-3
                text-sm
                outline-none
                transition
                focus:border-sky-500
                disabled:opacity-60
              "
            />

            <button
              type="button"

              disabled={
                configureDisabled
              }

              onClick={
                configureSource
              }

              className="
                rounded-xl
                bg-sky-600
                px-6
                py-3
                text-sm
                font-semibold
                transition
                hover:bg-sky-500
                disabled:
                  cursor-not-allowed
                disabled:
                  opacity-40
              "
            >
              {loading
                ? "Procesando..."
                : "Configurar fuente"}
            </button>
          </div>


          {message && (
            <p
              className="
                mt-3
                break-words
                text-sm
                text-sky-300
              "
            >
              {message}
            </p>
          )}


          {sourceOpened && (
            <div
              className="
                mt-4
                flex
                flex-wrap
                gap-x-5
                gap-y-2
                text-xs
                text-slate-500
              "
            >
              <InfoInline
                label="Estado"
                value="Fuente abierta"
              />

              <InfoInline
                label="Resolver"
                value={
                  sourceInfo
                    ?.resolver
                  ?? "direct"
                }
              />

              <InfoInline
                label="Resolución"
                value={
                  sourceInfo?.width
                  && sourceInfo?.height
                    ? (
                      `${sourceInfo.width}`
                      + `×`
                      + `${sourceInfo.height}`
                    )
                    : "--"
                }
              />

              <InfoInline
                label="FPS origen"
                value={
                  sourceInfo
                    ?.fps_source
                    ?.toFixed(1)
                  ?? "--"
                }
              />

              {sourceInfo
                ?.format_id && (
                <InfoInline
                  label="Formato"
                  value={
                    sourceInfo
                      .format_id
                  }
                />
              )}
            </div>
          )}
        </section>


        {/* ================================================= */}
        {/* OPERACIÓN */}
        {/* ================================================= */}

        <div
          className="
            grid
            gap-6
            xl:grid-cols-[minmax(0,2fr)_420px]
          "
        >

          <section
            className="
              overflow-hidden
              rounded-2xl
              border
              border-slate-800
              bg-black
            "
          >
            {videoUrl ? (
              <img
                src={videoUrl}
                alt="Video de aforo vehicular"

                className="
                  block
                  aspect-video
                  w-full
                  object-contain
                "
              />
            ) : (
              <div
                className="
                  flex
                  aspect-video
                  items-center
                  justify-center
                  p-8
                  text-center
                  text-sm
                  text-slate-500
                "
              >
                Configura una fuente
                para iniciar el monitoreo.
              </div>
            )}


            <div
              className="
                grid
                grid-cols-2
                gap-px
                border-t
                border-slate-800
                bg-slate-800
                sm:grid-cols-4
              "
            >
              <VideoInfo
                label="Modelo"
                value="YOLO11 V2"
              />

              <VideoInfo
                label="Tracker"
                value="ByteTrack"
              />

              <VideoInfo
                label="Procesamiento"
                value={
                  model?.device
                  ?? "--"
                }
              />

              <VideoInfo
                label="FPS IA"
                value={
                  video?.fps
                    ?.toFixed(1)
                  ?? "--"
                }
              />
            </div>
          </section>


          <aside
            className="
              flex
              flex-col
              gap-4
            "
          >
            <MetricCard
              label="Vehículos"

              value={
                traffic?.total
                ?? 0
              }

              large
            />


            <div
              className="
                grid
                grid-cols-2
                gap-4
              "
            >
              <MetricCard
                label="Sentido A"

                value={
                  traffic
                    ?.direcciones
                    ?.A
                  ?? 0
                }
              />

              <MetricCard
                label="Sentido B"

                value={
                  traffic
                    ?.direcciones
                    ?.B
                  ?? 0
                }
              />
            </div>


            <section
              className="
                rounded-2xl
                border
                border-slate-800
                bg-slate-900
                p-5
              "
            >
              <p
                className="
                  mb-4
                  text-xs
                  font-semibold
                  uppercase
                  tracking-wider
                  text-slate-500
                "
              >
                Sesión de aforo
              </p>


              <div
                className="
                  space-y-3
                "
              >
                <DataRow
                  label="Inicio"

                  value={
                    currentSession
                      ? formatDateTime(
                          currentSession
                            .started_at,
                          mounted
                        )
                      : "--"
                  }
                />

                <DataRow
                  label="Tiempo"
                  value={elapsed}
                  strong
                />

                <DataRow
                  label="Eventos"

                  value={
                    String(
                      traffic?.eventos
                      ?? 0
                    )
                  }
                />

                <DataRow
                  label="Sesión"

                  value={
                    currentSession
                      ?.session_id
                    ?? "--"
                  }

                  mono
                />
              </div>


              <div
                className="
                  mt-5
                  grid
                  grid-cols-2
                  gap-3
                "
              >
                <button
                  type="button"

                  onClick={
                    startSession
                  }

                  disabled={
                    startDisabled
                  }

                  className="
                    rounded-xl
                    bg-emerald-600
                    px-4
                    py-3
                    text-sm
                    font-semibold
                    transition
                    hover:bg-emerald-500
                    disabled:
                      cursor-not-allowed
                    disabled:
                      opacity-35
                  "
                >
                  Iniciar aforo
                </button>

                <button
                  type="button"

                  onClick={
                    stopSession
                  }

                  disabled={
                    stopDisabled
                  }

                  className="
                    rounded-xl
                    bg-red-700
                    px-4
                    py-3
                    text-sm
                    font-semibold
                    transition
                    hover:bg-red-600
                    disabled:
                      cursor-not-allowed
                    disabled:
                      opacity-35
                  "
                >
                  Detener aforo
                </button>
              </div>
            </section>
          </aside>
        </div>


        {/* ================================================= */}
        {/* INGENIERÍA DE TRÁFICO */}
        {/* ================================================= */}

        <section
          className="
            mt-6
            rounded-2xl
            border
            border-slate-800
            bg-slate-900
            p-5
          "
        >
          <div
            className="
              mb-5
              flex
              flex-col
              justify-between
              gap-2
              sm:flex-row
              sm:items-center
            "
          >
            <div>
              <h2
                className="
                  text-sm
                  font-bold
                  uppercase
                  tracking-wider
                  text-slate-300
                "
              >
                Ingeniería de tráfico
              </h2>

              <p
                className="
                  mt-1
                  text-xs
                  text-slate-500
                "
              >
                Métricas temporales
                observadas durante la sesión
              </p>
            </div>

            <span
              className="
                rounded-full
                border
                border-amber-500/20
                bg-amber-500/10
                px-3
                py-1
                text-[11px]
                text-amber-300
              "
            >
              TPDA preliminar · k ={" "}
              {
                dashboard
                  ?.engineering
                  ?.k_vhp
                ?? 0.08
              }
            </span>
          </div>


          <div
            className="
              grid
              gap-4
              sm:grid-cols-2
              lg:grid-cols-3
              2xl:grid-cols-6
            "
          >
            <EngineeringCard
              label="q1"
              metric={
                metrics?.q1
              }
            />

            <EngineeringCard
              label="q5"
              metric={
                metrics?.q5
              }
            />

            <EngineeringCard
              label="q15"
              metric={
                metrics?.q15
              }
            />

            <EngineeringCard
              label="TH"
              metric={
                metrics?.th
              }
            />

            <MetricCard
              label="TH máximo"

              value={
                metrics
                  ?.max_th
                  ?.available
                  ? (
                    `${metrics.max_th
                      .veh_h
                      .toFixed(0)} veh/h`
                  )
                  : "Pendiente"
              }
            />

            <MetricCard
              label="TPDA preliminar"

              value={
                metrics
                  ?.tpda
                  ?.available
                  ? (
                    `${metrics.tpda
                      .veh_day
                      .toFixed(0)} veh/día`
                  )
                  : "Pendiente"
              }
            />
          </div>


          <p
            className="
              mt-4
              text-[11px]
              leading-relaxed
              text-slate-500
            "
          >
            El TPDA mostrado por el sistema
            es una estimación preliminar.
            Se utiliza el máximo volumen
            horario observado como proxy
            del VHP y la relación
            TPDA = VHP / k. No representa
            un TPDA anual medido.
          </p>
        </section>


        {/* ================================================= */}
        {/* INFORMACIÓN OPERATIVA */}
        {/* ================================================= */}

        <div
          className="
            mt-6
            grid
            gap-6
            xl:grid-cols-3
          "
        >
          <DetailPanel
            title="Estación de aforo"
          >
            <DataRow
              label="Estación"
              value={
                station
                  ?.station_id
                ?? "--"
              }
            />

            <DataRow
              label="Vía"
              value={
                station
                  ?.road
                ?? "--"
              }
            />

            <DataRow
              label="Referencia"
              value={
                station
                  ?.reference
                ?? "--"
              }
            />

            <DataRow
              label="Ciudad"
              value={
                station
                  ?.city
                ?? "--"
              }
            />
          </DetailPanel>


          <DetailPanel
            title="Motor de visión"
          >
            <DataRow
              label="Detector"
              value="YOLO11 V2"
            />

            <DataRow
              label="Seguimiento"
              value="ByteTrack"
            />

            <DataRow
              label="Dispositivo"
              value={
                model?.device
                ?? "--"
              }
            />

            <DataRow
              label="GPU"
              value={
                model?.gpu
                ?? "CPU"
              }
            />

            <DataRow
              label="FPS"
              value={
                video
                  ? (
                    `${video.fps
                      .toFixed(1)}`
                  )
                  : "--"
              }
            />
          </DetailPanel>


          <DetailPanel
            title="Último cruce"
          >
            {dashboard
              ?.last_event ? (
                <>
                  <DataRow
                    label="Hora"

                    value={
                      formatTime(
                        dashboard
                          .last_event
                          .timestamp,
                        mounted
                      )
                    }
                  />

                  <DataRow
                    label="Clase"

                    value={
                      dashboard
                        .last_event
                        .clase
                    }
                  />

                  <DataRow
                    label="Dirección"

                    value={
                      dashboard
                        .last_event
                        .direccion
                    }
                  />

                  <DataRow
                    label="Track ID"

                    value={
                      String(
                        dashboard
                          .last_event
                          .track_id
                      )
                    }
                  />

                  <DataRow
                    label="Confianza"

                    value={
                      (
                        dashboard
                          .last_event
                          .confianza
                        * 100
                      ).toFixed(1)
                      + "%"
                    }
                  />
                </>
              ) : (
                <p
                  className="
                    text-sm
                    text-slate-500
                  "
                >
                  Aún no se registró
                  ningún cruce.
                </p>
              )
            }
          </DetailPanel>
        </div>


        {/* ================================================= */}
        {/* COMPOSICIÓN */}
        {/* ================================================= */}

        <section
          className="
            mt-6
            rounded-2xl
            border
            border-slate-800
            bg-slate-900
            p-5
          "
        >
          <h2
            className="
              mb-4
              text-sm
              font-bold
              uppercase
              tracking-wider
              text-slate-300
            "
          >
            Composición vehicular
          </h2>


          <div
            className="
              grid
              grid-cols-2
              gap-3
              sm:grid-cols-3
              lg:grid-cols-6
            "
          >
            {[
              "auto",
              "moto",
              "bus",
              "camion",
              "furgoneta",
              "triciclo",
            ].map(
              (
                name
              ) => (
                <div
                  key={name}

                  className="
                    rounded-xl
                    border
                    border-slate-800
                    bg-slate-950/60
                    p-4
                  "
                >
                  <p
                    className="
                      text-[11px]
                      font-semibold
                      uppercase
                      tracking-wider
                      text-slate-500
                    "
                  >
                    {name}
                  </p>

                  <p
                    className="
                      mt-2
                      text-2xl
                      font-bold
                    "
                  >
                    {
                      traffic
                        ?.clases
                        ?.[name]
                      ?? 0
                    }
                  </p>
                </div>
              )
            )}
          </div>
        </section>


        {/* ================================================= */}
        {/* HISTORIAL */}
        {/* ================================================= */}

        <section
          className="
            mt-6
            overflow-hidden
            rounded-2xl
            border
            border-slate-800
            bg-slate-900
          "
        >
          <div
            className="
              border-b
              border-slate-800
              p-5
            "
          >
            <h2
              className="
                text-sm
                font-bold
                uppercase
                tracking-wider
                text-slate-300
              "
            >
              Historial de aforos
            </h2>

            <p
              className="
                mt-1
                text-xs
                text-slate-500
              "
            >
              Últimas sesiones
              registradas en SQLite
            </p>
          </div>


          <div
            className="
              overflow-x-auto
            "
          >
            <table
              className="
                w-full
                min-w-[950px]
                text-left
                text-sm
              "
            >
              <thead
                className="
                  bg-slate-950/60
                  text-[11px]
                  uppercase
                  tracking-wider
                  text-slate-500
                "
              >
                <tr>
                  <th className="px-5 py-3">
                    Sesión
                  </th>

                  <th className="px-5 py-3">
                    Inicio
                  </th>

                  <th className="px-5 py-3">
                    Fin
                  </th>

                  <th className="px-5 py-3">
                    Duración
                  </th>

                  <th className="px-5 py-3">
                    Vehículos
                  </th>

                  <th className="px-5 py-3">
                    Estado
                  </th>

                  <th className="px-5 py-3">
                    Exportar
                  </th>
                </tr>
              </thead>


              <tbody
                className="
                  divide-y
                  divide-slate-800
                "
              >
                {sessions.length
                  === 0 ? (
                    <tr>
                      <td
                        colSpan={7}

                        className="
                          px-5
                          py-8
                          text-center
                          text-slate-500
                        "
                      >
                        No existen sesiones
                        registradas.
                      </td>
                    </tr>
                  ) : (
                    sessions.map(
                      (
                        session
                      ) => (
                        <tr
                          key={
                            session
                              .session_id
                          }

                          className="
                            transition
                            hover:bg-slate-800/30
                          "
                        >
                          <td
                            className="
                              px-5
                              py-4
                              font-mono
                              text-xs
                              text-sky-300
                            "
                          >
                            {
                              session
                                .session_id
                            }
                          </td>

                          <td
                            className="
                              px-5
                              py-4
                              text-slate-300
                            "
                          >
                            {
                              formatDateTime(
                                session
                                  .started_at,
                                mounted
                              )
                            }
                          </td>

                          <td
                            className="
                              px-5
                              py-4
                              text-slate-400
                            "
                          >
                            {
                              session
                                .ended_at
                                ? formatDateTime(
                                    session
                                      .ended_at,
                                    mounted
                                  )
                                : "--"
                            }
                          </td>

                          <td
                            className="
                              px-5
                              py-4
                              text-slate-300
                            "
                          >
                            {
                              session
                                .duration_seconds
                                ? formatDuration(
                                    session
                                      .duration_seconds
                                  )
                                : (
                                  session
                                    .status
                                  === "EN_CURSO"
                                    ? "En curso"
                                    : "--"
                                )
                            }
                          </td>

                          <td
                            className="
                              px-5
                              py-4
                              font-semibold
                            "
                          >
                            {
                              session
                                .total_vehicles
                            }
                          </td>

                          <td
                            className="
                              px-5
                              py-4
                            "
                          >
                            <SessionBadge
                              status={
                                session
                                  .status
                              }
                            />
                          </td>

                          <td
                            className="
                              px-5
                              py-4
                            "
                          >
                            <a
                              href={
                                `${API}`
                                + `/api/sessions/`
                                + `${session.session_id}`
                                + `/export.csv`
                              }

                              className="
                                rounded-lg
                                border
                                border-slate-700
                                bg-slate-950
                                px-3
                                py-2
                                text-xs
                                font-medium
                                text-slate-300
                                transition
                                hover:border-sky-500
                                hover:text-sky-300
                              "
                            >
                              CSV
                            </a>
                          </td>
                        </tr>
                      )
                    )
                  )
                }
              </tbody>
            </table>
          </div>
        </section>


        {/* ================================================= */}
        {/* FOOTER ACADÉMICO */}
        {/* ================================================= */}

        <footer
          className="
            mt-8
            overflow-hidden
            rounded-2xl
            border
            border-slate-800
            bg-slate-950
          "
        >
          <div
            className="
              px-6
              py-8
              text-center
            "
          >
            <p
              className="
                text-base
                font-bold
                text-slate-100
              "
            >
              Sistema de Detección
              y Aforo Vehicular
            </p>


            <p
              className="
                mt-2
                text-sm
                text-slate-400
              "
            >
              Procesamiento de Imágenes
              y Visión Computacional
            </p>


            <p
              className="
                mt-1
                text-sm
                text-slate-400
              "
            >
              Maestría en Ciencia de Datos
              e Inteligencia Artificial
            </p>


            <p
              className="
                mt-1
                text-sm
                font-medium
                text-slate-300
              "
            >
              Universidad Católica Boliviana
              &quot;San Pablo&quot;
            </p>


            <div
              className="
                mx-auto
                my-6
                h-px
                max-w-2xl
                bg-slate-800
              "
            />


            <p
              className="
                text-xs
                uppercase
                tracking-[0.18em]
                text-slate-500
              "
            >
              Desarrollado por
            </p>


            <div
              className="
                mt-3
                flex
                flex-col
                items-center
                justify-center
                gap-1
                text-sm
                sm:flex-row
                sm:gap-2
              "
            >
              <span
                className="
                  font-semibold
                  text-sky-300
                "
              >
                Jhonny Paul Pinto Phillips
              </span>

              <span
                className="
                  hidden
                  text-slate-600
                  sm:inline
                "
              >
                ·
              </span>

              <span
                className="
                  font-semibold
                  text-sky-300
                "
              >
                Ronald Marcelo Pinto Delgadillo
              </span>
            </div>


            <nav
              className="
                mt-6
                flex
                flex-wrap
                items-center
                justify-center
                gap-x-3
                gap-y-2
                text-xs
              "
            >
              <a
                href={GITHUB_URL}
                target="_blank"
                rel="noreferrer"

                className="
                  font-semibold
                  text-sky-400
                  transition
                  hover:text-sky-300
                "
              >
                GitHub
              </a>

              <span
                className="
                  text-slate-700
                "
              >
                ·
              </span>

              <a
                href={`${API}/docs`}
                target="_blank"
                rel="noreferrer"

                className="
                  font-semibold
                  text-sky-400
                  transition
                  hover:text-sky-300
                "
              >
                API Docs
              </a>

              <span
                className="
                  text-slate-700
                "
              >
                ·
              </span>

              <span
                className="
                  text-slate-500
                "
              >
                YOLO11 V2
              </span>

              <span
                className="
                  text-slate-700
                "
              >
                ·
              </span>

              <span
                className="
                  text-slate-500
                "
              >
                ByteTrack
              </span>

              <span
                className="
                  text-slate-700
                "
              >
                ·
              </span>

              <span
                className="
                  text-slate-500
                "
              >
                FastAPI
              </span>

              <span
                className="
                  text-slate-700
                "
              >
                ·
              </span>

              <span
                className="
                  text-slate-500
                "
              >
                Next.js
              </span>
            </nav>
          </div>


          <div
            className="
              flex
              flex-col
              items-center
              justify-between
              gap-2
              border-t
              border-slate-800
              bg-slate-900/50
              px-6
              py-4
              text-[11px]
              text-slate-500
              sm:flex-row
            "
          >
            <span>
              © 2026 Jhonny Paul Pinto Phillips
              y Ronald Marcelo Pinto Delgadillo
            </span>

            <span>
              Versión 1.0.0
            </span>
          </div>
        </footer>
      </div>
    </main>
  );
}


// ===========================================================
// COMPONENTES
// ===========================================================

function MetricCard({
  label,
  value,
  large = false,
}: {
  label: string;
  value: string | number;
  large?: boolean;
}) {
  return (
    <div
      className="
        rounded-2xl
        border
        border-slate-800
        bg-slate-900
        p-5
      "
    >
      <p
        className="
          text-[11px]
          font-semibold
          uppercase
          tracking-wider
          text-slate-500
        "
      >
        {label}
      </p>

      <p
        className={`
          mt-2
          font-bold
          tracking-tight

          ${
            large
              ? "text-4xl"
              : "text-3xl"
          }
        `}
      >
        {value}
      </p>
    </div>
  );
}


function EngineeringCard({
  label,
  metric,
}: {
  label: string;

  metric?: {
    available: boolean;
    vehicles: number | null;
    veh_h: number | null;
  };
}) {
  return (
    <MetricCard
      label={label}

      value={
        metric?.available
        && metric.veh_h !== null
          ? (
            `${metric.veh_h
              .toFixed(0)} veh/h`
          )
          : "Pendiente"
      }
    />
  );
}


function DetailPanel({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section
      className="
        rounded-2xl
        border
        border-slate-800
        bg-slate-900
        p-5
      "
    >
      <h2
        className="
          mb-4
          text-xs
          font-bold
          uppercase
          tracking-wider
          text-slate-400
        "
      >
        {title}
      </h2>

      <div
        className="
          space-y-3
        "
      >
        {children}
      </div>
    </section>
  );
}


function DataRow({
  label,
  value,
  strong = false,
  mono = false,
}: {
  label: string;
  value: string;
  strong?: boolean;
  mono?: boolean;
}) {
  return (
    <div
      className="
        flex
        items-start
        justify-between
        gap-4
      "
    >
      <span
        className="
          text-xs
          text-slate-500
        "
      >
        {label}
      </span>

      <span
        className={`
          text-right
          text-sm
          text-slate-200

          ${
            strong
              ? "font-bold text-sky-300"
              : ""
          }

          ${
            mono
              ? "font-mono text-xs"
              : ""
          }
        `}
      >
        {value}
      </span>
    </div>
  );
}


function VideoInfo({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div
      className="
        bg-slate-950
        px-4
        py-3
      "
    >
      <p
        className="
          text-[10px]
          uppercase
          tracking-wider
          text-slate-600
        "
      >
        {label}
      </p>

      <p
        className="
          mt-1
          truncate
          text-xs
          font-semibold
          text-slate-300
        "
      >
        {value}
      </p>
    </div>
  );
}


function InfoInline({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <span>
      <span
        className="
          text-slate-600
        "
      >
        {label}:{" "}
      </span>

      <span
        className="
          text-slate-400
        "
      >
        {value}
      </span>
    </span>
  );
}


function SessionBadge({
  status,
}: {
  status: string;
}) {
  const active =
    status === "EN_CURSO";

  const interrupted =
    status.includes(
      "INTERRUMPIDA"
    );

  return (
    <span
      className={`
        rounded-full
        px-2.5
        py-1
        text-[10px]
        font-semibold

        ${
          active
            ? (
              "bg-emerald-500/10 "
              + "text-emerald-300"
            )
            : interrupted
              ? (
                "bg-amber-500/10 "
                + "text-amber-300"
              )
              : (
                "bg-sky-500/10 "
                + "text-sky-300"
              )
        }
      `}
    >
      {status}
    </span>
  );
}


// ===========================================================
// FORMATO
// ===========================================================

function formatDuration(
  seconds: number,
) {
  const total =
    Math.max(
      0,
      Math.floor(seconds)
    );

  const hours =
    Math.floor(
      total / 3600
    );

  const minutes =
    Math.floor(
      (
        total % 3600
      ) / 60
    );

  const secs =
    total % 60;

  return [
    hours,
    minutes,
    secs,
  ]
    .map(
      (
        value
      ) =>
        String(value)
          .padStart(
            2,
            "0"
          )
    )
    .join(":");
}


function formatDateTime(
  value: string,
  mounted: boolean,
) {
  if (!mounted) {
    return "--";
  }

  const date =
    new Date(value);

  if (
    Number.isNaN(
      date.getTime()
    )
  ) {
    return "--";
  }

  return new Intl.DateTimeFormat(
    "es-BO",
    {
      day: "2-digit",
      month: "2-digit",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    }
  ).format(date);
}


function formatTime(
  value: string,
  mounted: boolean,
) {
  if (!mounted) {
    return "--";
  }

  const date =
    new Date(value);

  if (
    Number.isNaN(
      date.getTime()
    )
  ) {
    return "--";
  }

  return new Intl.DateTimeFormat(
    "es-BO",
    {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    }
  ).format(date);
}