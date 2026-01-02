import cv2
import os
import sys
import numpy as np
from shapely.geometry import Polygon
from ultralytics import solutions

# ==========================================
# 1. TUS PARÁMETROS
# ==========================================
RTSP_URL = "rtsp://usertf:Tfo.-2525@172.16.10.22:554/cam/realmonitor?channel=5&subtype=0"
MODEL_PATH = "/home/aaravenatk/yolo/best.pt"
CONF_THRESHOLD = 0.7  # <--- Tu configuración solicitada
OUTPUT_FOLDER = "runs"

# ==========================================
# 2. DEFINICIÓN DE ZONAS
# ==========================================
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


region_points = {
    "personas": polygon_personas,
    "teleferico": polygon_teleferico,
}

# ==========================================
# 3. INICIALIZACIÓN
# ==========================================
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

print(f"Cargando modelo desde: {MODEL_PATH}")
try:
    # Inicializamos el contador con tus parámetros
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

print(f"Conectando a cámara...")
cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print("⛔ Error: No se puede conectar al RTSP.")
    sys.exit(1)

# ==========================================
# 4. CAPTURA Y DIBUJO
# ==========================================
print(f"📸 Iniciando captura de 10 frames con Confianza > {CONF_THRESHOLD}...")
frames_procesados = 0
max_frames = 10

try:
    while frames_procesados < max_frames:
        ret, frame = cap.read()
        if not ret:
            print("Error leyendo frame.")
            break
        
        # A. Ejecutar inferencia (Detecta y cuenta internamente)
        # Esto genera las cajas según el conf=0.8
        _ = rc(frame)
        
        # B. Recuperar la imagen visual
        # Intentamos sacar la imagen pintada por YOLO (im0). 
        # Si no existe, usamos el frame original limpio.
        frame_visual = getattr(rc, "im0", frame.copy())

        # C. DIBUJAR POLÍGONOS MANUALMENTE
        # Dibujamos las zonas ENCIMA de todo para que veas la calibración real
        
        # Zona Personas (VERDE)
        pts_p = np.array(polygon_personas, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame_visual, [pts_p], isClosed=True, color=(0, 255, 0), thickness=3)
        
        # Zona Teleférico (ROJO)
        pts_t = np.array(polygon_teleferico, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame_visual, [pts_t], isClosed=True, color=(0, 0, 255), thickness=3)

        # D. Guardar
        filename = f"{OUTPUT_FOLDER}/test_conf08_{frames_procesados+1}.jpg"
        cv2.imwrite(filename, frame_visual)
        
        print(f"✅ Guardado: {filename}")
        frames_procesados += 1

except Exception as e:
    print(f"Error procesando: {e}")

cap.release()
print("\n=== LISTO ===")
print(f"Revisa la carpeta '{OUTPUT_FOLDER}'")
