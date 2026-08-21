import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Aforo Vehicular",
  description:
    "Sistema de detección, seguimiento y aforo vehicular mediante YOLO11 y ByteTrack.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="es">
      <body>{children}</body>
    </html>
  );
}