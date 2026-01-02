"""  
Sistema de Procesamiento Headless para Detección en Tiempo Real
===============================================================
Script optimizado para ejecución en servidor sin interfaz gráfica.
Procesa stream RTSP con validación estricta de zonas poligonales.
"""

import cv2
import time
import datetime
import csv
import torch
import gc
import os
import sys
from shapely.geometry import Point, Polygon
from ultralytics import solutions

# ==========================================
# 1. CONFIGURACIÓN DEL SISTEMA
# ==========================================
# Parámetros de conexión RTSP, modelo YOLO y umbrales de detección
RTSP_URL = "rtsp://usertf:Tfo.-2525@172.16.10.22:554/cam/realmonitor?channel=5&subtype=0"
MODEL_PATH = "/home/aaravenatk/yolo/best.pt"
CONF_THRESHOLD = 0.7  # Exigencia alta

# Nombre de la carpeta de salida
OUTPUT_FOLDER = "detecciones"

# Definición de Zonas (Polígonos fijos)
polygon_personas = [
    (445, 299),
    (812, 341),
    (1061, 330),
    (1046, 153),
    (428, 177)
]

polygon_teleferico = [
    (488, 520),
    (460, 165),
    (810, 145),
    (832, 492)
]

# Solo para visualización interna de ultralytics (si se usara show=True)
region_points = {
    "personas": polygon_personas,
    "teleferico": polygon_teleferico,
}

# Crear objetos polígono de Shapely
person_poly = Polygon(polygon_personas)
tele_poly = Polygon(polygon_teleferico)

# Mapeo de clases (Asegúrate de que coincida con tu modelo 'best.pt')
# 0: person, 1: teleferico (ajusta si tu modelo tiene otro orden)
CLASS_MAP = {0: "person", 1: "teleferico"}

# ==========================================
# 2. FUNCIONES AUXILIARES
# ==========================================
# Utilidades para conversión de datos y manejo de conexiones
def to_list(x):
    """Convierte tensores/arrays a listas limpias."""
    if x is None: return []
    try:
        if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
    except: pass
    try:
        import numpy as np
        if isinstance(x, np.ndarray): return x.tolist()
    except: pass
    if isinstance(x, (list, tuple)): return list(x)
    return [x]

def conectar_camara_robusto(url, max_intentos=10, espera_seg=5):
    """Conexión con reintentos para tolerancia a fallos."""
    print(f"--- Intentando conectar al stream RTSP... ---")
    for i in range(max_intentos):
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"✅ Conexión exitosa en el intento {i+1}")
                return cap
            else:
                print(f"⚠️ Conectado, pero sin imagen. Reintentando...")
        cap.release()
        print(f"❌ Fallo intento {i+1}/{max_intentos}. Esperando {espera_seg}s...")
        time.sleep(espera_seg)
    return None

# ==========================================
# 3. PREPARACIÓN DE ARCHIVOS
# ==========================================

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"📁 Carpeta de salida: {OUTPUT_FOLDER}")

timestamp_inicio = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"log_inferencia_{timestamp_inicio}.csv"
summary_filename = f"resumen_{timestamp_inicio}.csv"

LOG_PATH = os.path.join(OUTPUT_FOLDER, log_filename)
SUMMARY_PATH = os.path.join(OUTPUT_FOLDER, summary_filename)

print(f"📄 Guardando en: {LOG_PATH}")

try:
    user_input = input("Ingrese minutos a capturar (Enter para 1 min): ")
    user_minutes = float(user_input) if user_input.strip() else 1.0
except ValueError:
    print("Valor no válido, usando 1 minuto.")
    user_minutes = 1.0

duration_seconds = user_minutes * 60

print(f"--- MODO SERVIDOR (Headless) ---")
print(f"--- Confianza mínima: {CONF_THRESHOLD} ---")

# D. Cargar Modelo
try:
    print(f"Cargando modelo desde: {MODEL_PATH}")
    # Nota: Usamos RegionCounter principalmente para el tracking, pero el conteo
    # final lo haremos con nuestra lógica manual 'shapely' para mayor control.
    rc = solutions.RegionCounter(
        model=MODEL_PATH,
        region=region_points,
        show=False,
        tracker="bytetrack.yaml",
        conf=CONF_THRESHOLD,
    )
except Exception as e:
    print(f"⛔ Error cargando modelo: {e}")
    sys.exit(1)

# E. Conectar Cámara
cap = conectar_camara_robusto(RTSP_URL)
if cap is None:
    print("⛔ No se pudo conectar a la cámara. Revisa la IP/Red.")
    sys.exit(1)

# F. Abrir CSV de Log
log_f = open(LOG_PATH, 'w', newline='')
log_writer = csv.writer(log_f)
log_writer.writerow(["Timestamp", "Frame_ID", "Clase", "Track_ID", "Confianza", "Coordenadas_XY"])

# Variables de estado (Sets para IDs únicos)
unique_ids_detected = {
    "person": set(), 
    "teleferico": set()
}

start_time = time.time()
frame_count = 0

# ==========================================
# 4. BUCLE PRINCIPAL DE INFERENCIA
# ==========================================
# Procesa frames, ejecuta detección y registra resultados en CSV
try:
    with torch.no_grad():
        while True:
            current_time = time.time()
            elapsed = current_time - start_time

            if elapsed >= duration_seconds:
                print("\n⏰ Tiempo límite alcanzado.")
                break

            ret, frame = cap.read()
            if not ret:
                print("\n⚠️ Pérdida de señal de video RTSP.")
                break

            # Procesar frame con ultralytics
            res = rc(frame)

            # Extraer listas limpias
            boxes = to_list(getattr(rc, "boxes", []))
            clss = to_list(getattr(rc, "clss", []))
            track_ids = to_list(getattr(rc, "track_ids", []))
            confs = to_list(getattr(rc, "confs", []))

            n = min(len(boxes), len(clss), len(track_ids))
            if len(confs) < n: confs = [0.0] * n

            timestamp_now = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]

            for i in range(n):
                box = boxes[i]
                cls_idx = int(clss[i])
                tid = int(track_ids[i])
                conf_val = float(confs[i])
                
                cls_name = CLASS_MAP.get(cls_idx, "unknown")

                # Calcular centroide (pies o centro, aquí usamos centro)
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                pt = Point(cx, cy)

                # ========================================================
                # LÓGICA DE EXCLUSIÓN ESTRICTA (SOLUCIÓN AL PROBLEMA)
                # ========================================================
                
                # CASO 1: ES PERSONA (Clase 0)
                if cls_name == "person":
                    # Regla A: Solo verificar intersección con Polígono Personas
                    if person_poly.contains(pt):
                        # Regla B: Verificar que este ID no haya sido contado antes como Teleférico
                        # (Esto evita duplicados si el modelo se confunde momentáneamente)
                        if tid not in unique_ids_detected["teleferico"]:
                            if tid not in unique_ids_detected["person"]:
                                unique_ids_detected["person"].add(tid)
                                # Registrar evento nuevo
                                log_writer.writerow([
                                    timestamp_now, frame_count, cls_name, tid,
                                    f"{conf_val:.4f}", f"{int(cx)},{int(cy)}"
                                ])

                # CASO 2: ES TELEFERICO (Clase 1)
                elif cls_name == "teleferico":
                    # Regla A: Solo verificar intersección con Polígono Teleférico
                    if tele_poly.contains(pt):
                        # Regla B: Verificar que este ID no sea una persona mal clasificada
                        if tid not in unique_ids_detected["person"]:
                            if tid not in unique_ids_detected["teleferico"]:
                                unique_ids_detected["teleferico"].add(tid)
                                # Registrar evento nuevo
                                log_writer.writerow([
                                    timestamp_now, frame_count, cls_name, tid,
                                    f"{conf_val:.4f}", f"{int(cx)},{int(cy)}"
                                ])

            # Feedback en consola
            if frame_count % 30 == 0:
                p_count = len(unique_ids_detected['person'])
                t_count = len(unique_ids_detected['teleferico'])
                print(f"\r[En Proceso] T: {elapsed:.0f}/{duration_seconds:.0f}s | Pers: {p_count} | Tele: {t_count}", end="")

            # Limpieza periódica de memoria
            if frame_count % 500 == 0:
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            frame_count += 1

except KeyboardInterrupt:
    print("\n\n🛑 Detenido manualmente (Ctrl+C).")

except Exception as e:
    print(f"\n⛔ Error inesperado: {e}")

finally:
    print("\n--- Finalizando sesión... ---")
    try: log_f.close()
    except: pass
    if 'cap' in locals() and cap is not None: cap.release()

    final_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_real_time = time.time() - start_time
    total_personas = len(unique_ids_detected["person"])
    total_teleferico = len(unique_ids_detected["teleferico"])

    try:
        with open(SUMMARY_PATH, 'w', newline='') as f:
            s_writer = csv.writer(f)
            s_writer.writerow(["Fecha_Hora_Fin", "Duracion_Segundos", "Total_Personas", "Total_Teleferico"])
            s_writer.writerow([final_ts, round(total_real_time, 2), total_personas, total_teleferico])
        print(f"✅ Resumen guardado: {SUMMARY_PATH}")
        print(f"✅ Log detallado: {LOG_PATH}")
    except Exception as e:
        print(f"⚠️ Error guardando resumen: {e}")
