import cv2
import time
import datetime
import numpy as np
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
CONF_THRESHOLD = 0.7 

# --- AJUSTE DE FILTRO ---
# Bajamos a 20 frames (~1.3 seg) para acercarnos a tu conteo manual de 195 personas
MIN_FRAMES = 20 
# ------------------------

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

timestamp_inicio = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"log_inferencia_filtrado_{timestamp_inicio}.csv"
summary_filename = f"resumen_filtrado_{timestamp_inicio}.csv"
# --- NUEVO: Nombre del archivo de video ---
video_filename = f"grabacion_inferencia_{timestamp_inicio}.mp4" 
VIDEO_PATH = os.path.join(OUTPUT_FOLDER, video_filename)
# ------------------------------------------

LOG_PATH = os.path.join(OUTPUT_FOLDER, log_filename)
SUMMARY_PATH = os.path.join(OUTPUT_FOLDER, summary_filename)

# Configuración de tiempo
if len(sys.argv) > 1:
    try:
        minutes_input = float(sys.argv[1])
        duration_seconds = minutes_input * 60
        print(f"⏱️  Tiempo límite: {minutes_input} min ({duration_seconds} s)")
    except ValueError:
        print("❌ Error: El argumento debe ser un número (minutos).")
        sys.exit(1)
else:
    print("\n❌ Error: Debes especificar el tiempo en MINUTOS (ej: python main.py 5)")
    sys.exit(1)

print(f"--- MODO VIDEO + LOG ---")
print(f"--- Umbral Frames: {MIN_FRAMES} ---")

# Cargar Modelo
try:
    print(f"Cargando modelo: {MODEL_PATH}")
    rc = solutions.RegionCounter(
        model=MODEL_PATH,
        region=region_points,
        show=False, 
        tracker="bytetrack.yaml",
        conf=CONF_THRESHOLD,
    )
except Exception as e:
    print(f"⛔ Error modelo: {e}")
    sys.exit(1)

# Conectar Cámara
cap = conectar_camara_robusto(RTSP_URL)
if cap is None: sys.exit(1)

# --- NUEVO: Configurar VideoWriter ---
# Obtenemos ancho y alto del video original
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_video = 15 # Asumimos 15fps como dijiste, o puedes usar cap.get(cv2.CAP_PROP_FPS)

# Codec mp4v es compatible con la mayoría de reproductores
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out_video = cv2.VideoWriter(VIDEO_PATH, fourcc, fps_video, (frame_width, frame_height))
print(f"🎥 Grabando video en: {VIDEO_PATH}")
# -------------------------------------

# Abrir Log
log_f = open(LOG_PATH, 'w', newline='')
log_writer = csv.writer(log_f)
log_writer.writerow(["Timestamp", "Frame_ID", "Clase", "Track_ID", "Confianza", "Coordenadas_XY"])

unique_ids_detected = {"person": set(), "teleferico": set()}
track_history = {} 

start_time = time.time()
frame_count = 0

# ==========================================
# 4. BUCLE DE INFERENCIA
# ==========================================
try:
    with torch.no_grad():
        while True:
            elapsed = time.time() - start_time
            if elapsed >= duration_seconds:
                print("\n⏰ Tiempo límite alcanzado.")
                break

            ret, frame = cap.read()
            if not ret:
                print("\n⚠️ Fin del stream.")
                break

            # Inferencia
            res = rc(frame)
            
            # Copia del frame para dibujar (opcional, ultralytics a veces dibuja sobre el original)
            frame_draw = frame.copy()

            # --- DIBUJAR POLÍGONOS DE ZONAS (Visualización) ---
            cv2.polylines(frame_draw, [np.array(polygon_personas, np.int32)], True, (0, 255, 0), 2) # Verde para personas
            cv2.polylines(frame_draw, [np.array(polygon_teleferico, np.int32)], True, (0, 0, 255), 2) # Rojo para teleferico

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
                
                # Coordenadas caja
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                pt = Point(cx, cy)

                # Validar Zona
                is_inside = False
                color_box = (255, 255, 255) # Blanco por defecto

                if cls_name == "person" and person_poly.contains(pt):
                    is_inside = True
                    color_box = (0, 255, 0) # Verde
                elif cls_name == "teleferico" and tele_poly.contains(pt):
                    is_inside = True
                    color_box = (0, 0, 255) # Rojo

                if is_inside:
                    # Lógica de Filtro
                    if tid not in track_history: track_history[tid] = 0
                    track_history[tid] += 1

                    if track_history[tid] >= MIN_FRAMES:
                        unique_ids_detected[cls_name].add(tid)
                        log_writer.writerow([
                            timestamp_now, frame_count, cls_name, tid,
                            f"{conf_val:.4f}", f"{int(cx)},{int(cy)}"
                        ])
                        
                        # --- DIBUJAR CAJA Y ID ---
                        # Solo dibujamos si es un ID válido y confirmado (o si quieres ver todo, saca el if)
                        cv2.rectangle(frame_draw, (x1, y1), (x2, y2), color_box, 2)
                        label = f"{cls_name} ID:{tid}"
                        cv2.putText(frame_draw, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 2)
                        # -------------------------

            # --- GUARDAR FRAME EN VIDEO ---
            out_video.write(frame_draw)
            # ------------------------------

            if frame_count % 30 == 0:
                p_count = len(unique_ids_detected['person'])
                t_count = len(unique_ids_detected['teleferico'])
                print(f"\r[Rec] T: {elapsed:.0f}/{duration_seconds:.0f}s | Pers: {p_count} | Tele: {t_count}", end="")

            if frame_count % 500 == 0:
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            frame_count += 1

except KeyboardInterrupt:
    print("\n🛑 Detenido manualmente.")

except Exception as e:
    print(f"\n⛔ Error: {e}")

finally:
    print("\n--- Finalizando... ---")
    try: log_f.close()
    except: pass
    
    # --- CERRAR VIDEO ---
    if 'out_video' in locals() and out_video.isOpened():
        out_video.release()
        print(f"✅ Video guardado: {VIDEO_PATH}")
    # --------------------

    if 'cap' in locals() and cap is not None: cap.release()

    # Guardar Resumen
    final_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_real_time = time.time() - start_time
    total_personas = len(unique_ids_detected["person"])
    total_teleferico = len(unique_ids_detected["teleferico"])

    try:
        with open(SUMMARY_PATH, 'w', newline='') as f:
            s_writer = csv.writer(f)
            s_writer.writerow(["Fecha_Hora_Fin", "Duracion_Segundos", "Total_Personas_Filtradas", "Total_Teleferico"])
            s_writer.writerow([final_ts, round(total_real_time, 2), total_personas, total_teleferico])
        print(f"✅ Resumen guardado.")
    except: pass
