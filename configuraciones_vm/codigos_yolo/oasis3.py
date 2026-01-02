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
# 1. CONFIGURACIÓN EXACTA
# ==========================================
RTSP_URL = "rtsp://usertf:Tfo.-2525@172.16.10.22:554/cam/realmonitor?channel=5&subtype=0"
MODEL_PATH = "/home/aaravenatk/yolo/best.pt"
CONF_THRESHOLD = 0.7  # Nivel de exigencia alto

# --- NUEVO: FILTRO ANTI-RUIDO ---
# Si el video va a 15fps, 30 frames equivalen a 2 segundos de permanencia mínima.
MIN_FRAMES = 30 
# --------------------------------

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

polygon_teleferico = [(149, 136), (435, 114), (445, 417), (161, 443)]

region_points = {
    "personas": polygon_personas,
    "teleferico": polygon_teleferico,
}

person_poly = Polygon(polygon_personas)
tele_poly = Polygon(polygon_teleferico)
CLASS_MAP = {0: "person", 1: "teleferico"}

# ==========================================
# 2. FUNCIONES AUXILIARES
# ==========================================
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

# A. Crear carpeta si no existe
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"📁 Carpeta de salida: {OUTPUT_FOLDER}")

# B. Generar nombres de archivo únicos (Timestamp)
timestamp_inicio = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"log_inferencia_filtrado_{timestamp_inicio}.csv" # Nombre actualizado
summary_filename = f"resumen_filtrado_{timestamp_inicio}.csv"

LOG_PATH = os.path.join(OUTPUT_FOLDER, log_filename)
SUMMARY_PATH = os.path.join(OUTPUT_FOLDER, summary_filename)

print(f"📄 Guardando en: {LOG_PATH}")

# ---------------------------------------------------------
# C. CONFIGURACIÓN DE TIEMPO (AHORA EN MINUTOS)
# ---------------------------------------------------------
if len(sys.argv) > 1:
    try:
        minutes_input = float(sys.argv[1])
        duration_seconds = minutes_input * 60
        print(f"⏱️  Tiempo límite configurado: {minutes_input} minutos ({duration_seconds} segundos)")
    except ValueError:
        print("❌ Error: El argumento debe ser un número (minutos).")
        sys.exit(1)
else:
    print("\n❌ Error: Debes especificar el tiempo de ejecución en MINUTOS.")
    print("👉 Uso correcto: python main.py <minutos>")
    sys.exit(1)
# ---------------------------------------------------------

print(f"--- MODO SERVIDOR (Headless) ---")
print(f"--- Confianza mínima: {CONF_THRESHOLD} ---")
print(f"--- Filtro de persistencia: {MIN_FRAMES} frames (~2s) ---")
print(f"--- Para detener presiona CTRL + C ---")

# D. Cargar Modelo
try:
    print(f"Cargando modelo desde: {MODEL_PATH}")
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

# Variables de estado
unique_ids_detected = {"person": set(), "teleferico": set()}
# --- NUEVO: Diccionario para historial de frames por ID ---
track_history = {} 
# ----------------------------------------------------------

start_time = time.time()
frame_count = 0

# ==========================================
# 4. BUCLE DE INFERENCIA
# ==========================================
try:
    with torch.no_grad(): # Ahorro de memoria
        while True:
            current_time = time.time()
            elapsed = current_time - start_time

            # 1. Chequeo de tiempo
            if elapsed >= duration_seconds:
                print("\n⏰ Tiempo límite alcanzado.")
                break

            # 2. Leer Frame
            ret, frame = cap.read()
            if not ret:
                print("\n⚠️ Pérdida de señal de video RTSP.")
                break

            # 3. Inferencia
            res = rc(frame)

            # 4. Extraer datos
            boxes = to_list(getattr(rc, "boxes", []))
            clss = to_list(getattr(rc, "clss", []))
            track_ids = to_list(getattr(rc, "track_ids", []))
            confs = to_list(getattr(rc, "confs", []))

            # Relleno de seguridad
            n = min(len(boxes), len(clss), len(track_ids))
            if len(confs) < n: confs = [0.0] * n

            timestamp_now = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]

            for i in range(n):
                box = boxes[i]
                cls_idx = int(clss[i])
                tid = int(track_ids[i])
                conf_val = float(confs[i])
                cls_name = CLASS_MAP.get(cls_idx, "unknown")

                # Centroide
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                pt = Point(cx, cy)

                # Validar Zonas
                is_inside = False
                if cls_name == "person" and person_poly.contains(pt):
                    is_inside = True
                elif cls_name == "teleferico" and tele_poly.contains(pt):
                    is_inside = True

                if is_inside:
                    # --- NUEVA LÓGICA DE FILTRADO ---
                    
                    # 1. Inicializar contador para ID nuevo
                    if tid not in track_history:
                        track_history[tid] = 0
                    
                    # 2. Aumentar contador de permanencia
                    track_history[tid] += 1

                    # 3. SOLO procesar si supera el umbral (filtro de ruido)
                    if track_history[tid] >= MIN_FRAMES:
                        
                        # Agregar al set de IDs únicos (para el conteo final)
                        unique_ids_detected[cls_name].add(tid)

                        # Escribir en el LOG (Solo escribimos si es una detección "confirmada")
                        log_writer.writerow([
                            timestamp_now, frame_count, cls_name, tid,
                            f"{conf_val:.4f}", f"{int(cx)},{int(cy)}"
                        ])
                    # --------------------------------

            # 5. Feedback en consola (cada 1 seg aprox)
            if frame_count % 30 == 0:
                p_count = len(unique_ids_detected['person'])
                t_count = len(unique_ids_detected['teleferico'])
                print(f"\r[En Proceso] T: {elapsed:.0f}/{duration_seconds:.0f}s | Pers (Filtradas): {p_count} | Tele: {t_count}", end="")

            # 6. Limpieza de memoria (VRAM/RAM)
            if frame_count % 500 == 0:
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                
                # Limpieza de historial para IDs viejos (Opcional, para no saturar memoria en sesiones de horas)
                # Eliminamos del historial IDs que no se han visto recientemente si quisieras optimizar más,
                # pero para scripts de minutos no es crítico.

            frame_count += 1

except KeyboardInterrupt:
    print("\n\n🛑 Detenido manualmente (Ctrl+C).")

except Exception as e:
    print(f"\n⛔ Error inesperado: {e}")

finally:
    # ==========================================
    # 5. CIERRE Y RESUMEN
    # ==========================================
    print("\n--- Finalizando sesión... ---")

    try:
        log_f.close()
    except: pass

    if 'cap' in locals() and cap is not None: cap.release()

    final_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_real_time = time.time() - start_time
    total_personas = len(unique_ids_detected["person"])
    total_teleferico = len(unique_ids_detected["teleferico"])

    try:
        with open(SUMMARY_PATH, 'w', newline='') as f:
            s_writer = csv.writer(f)
            s_writer.writerow(["Fecha_Hora_Fin", "Duracion_Segundos", "Total_Personas_Filtradas", "Total_Teleferico"])
            s_writer.writerow([
                final_ts,
                round(total_real_time, 2),
                total_personas,
                total_teleferico
            ])
        print(f"✅ Resumen guardado: {SUMMARY_PATH}")
        print(f"✅ Log limpio guardado: {LOG_PATH}")
    except Exception as e:
        print(f"⚠️ Error guardando resumen: {e}")
