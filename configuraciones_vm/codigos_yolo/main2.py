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
# Tus parámetros solicitados:
RTSP_URL = "rtsp://usertf:Tfo.-2525@172.16.10.22:554/cam/realmonitor?channel=5&subtype=0"
MODEL_PATH = "/home/aaravenatk/yolo/best.pt"
CONF_THRESHOLD = 0.7  # Nivel de exigencia alto (solo detecciones muy claras)

# Nombre de la carpeta de salida
OUTPUT_FOLDER = "detecciones"

# Definición de Zonas (Polígonos fijos)
polygon_personas = polygon_personas = [
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
            # Leer un frame real para confirmar que no sea pantalla negra
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
log_filename = f"log_inferencia_{timestamp_inicio}.csv"
summary_filename = f"resumen_{timestamp_inicio}.csv"

LOG_PATH = os.path.join(OUTPUT_FOLDER, log_filename)
SUMMARY_PATH = os.path.join(OUTPUT_FOLDER, summary_filename)

print(f"📄 Guardando en: {LOG_PATH}")

# C. Input de usuario
try:
    user_input = input("Ingrese minutos a capturar (Enter para 1 min): ")
    user_minutes = float(user_input) if user_input.strip() else 1.0
except ValueError:
    print("Valor no válido, usando 1 minuto.")
    user_minutes = 1.0

duration_seconds = user_minutes * 60

print(f"--- MODO SERVIDOR (Headless) ---")
print(f"--- Confianza mínima: {CONF_THRESHOLD} ---")
print(f"--- Para detener presiona CTRL + C ---")

# D. Cargar Modelo
try:
    print(f"Cargando modelo desde: {MODEL_PATH}")
    rc = solutions.RegionCounter(
        model=MODEL_PATH,
        region=region_points,
        show=False,
        tracker="bytetrack.yaml",
        conf=CONF_THRESHOLD,  # <--- Aquí se aplica tu 0.8
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

            # (Opcional) Reducir resolución si va lento
            # frame = cv2.resize(frame, (1280, 720))

            # 3. Inferencia
            # El objeto 'rc' procesa internamente tracking + zonas
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
                    unique_ids_detected[cls_name].add(tid)
                    
                    # Escribir detección individual
                    log_writer.writerow([
                        timestamp_now, frame_count, cls_name, tid, 
                        f"{conf_val:.4f}", f"{int(cx)},{int(cy)}"
                    ])

            # 5. Feedback en consola (cada 1 seg aprox)
            if frame_count % 30 == 0:
                p_count = len(unique_ids_detected['person'])
                t_count = len(unique_ids_detected['teleferico'])
                print(f"\r[En Proceso] T: {elapsed:.0f}/{duration_seconds:.0f}s | Pers: {p_count} | Tele: {t_count}", end="")

            # 6. Limpieza de memoria (VRAM/RAM)
            if frame_count % 500 == 0:
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            
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
    
    # Cerrar Log
    try:
        log_f.close()
    except: pass
    
    # Liberar cámara
    if 'cap' in locals() and cap is not None: cap.release()
    
    # Calcular totales
    final_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_real_time = time.time() - start_time
    total_personas = len(unique_ids_detected["person"])
    total_teleferico = len(unique_ids_detected["teleferico"])
    
    # Escribir Resumen de esta sesión
    try:
        with open(SUMMARY_PATH, 'w', newline='') as f:
            s_writer = csv.writer(f)
            s_writer.writerow(["Fecha_Hora_Fin", "Duracion_Segundos", "Total_Personas", "Total_Teleferico"])
            s_writer.writerow([
                final_ts, 
                round(total_real_time, 2),
                total_personas, 
                total_teleferico
            ])
        print(f"✅ Resumen guardado: {SUMMARY_PATH}")
        print(f"✅ Log detallado: {LOG_PATH}")
    except Exception as e:
        print(f"⚠️ Error guardando resumen: {e}")
