import cv2
import time
import datetime
import csv
import torch
import gc
import os
import sys
import numpy as np
from collections import deque
from shapely.geometry import Point, Polygon, LineString
from ultralytics import solutions

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
RTSP_URL = "rtsp://usertf:Tfo.-2525@172.16.10.22:554/cam/realmonitor?channel=5&subtype=0"
MODEL_PATH = "/home/aaravenatk/yolo/best.pt"
CONF_THRESHOLD = 0.7 
OUTPUT_FOLDER = "detecciones"

FPS_VIDEO = 15.0

# --- FILTRO DE OPERADORES ---
UMBRAL_OPERADOR_SEG = 45.0 

# Zonas
polygon_personas_coords = [
    (477, 171), (478, 345), (765, 372), (1020, 264), 
    (1034, 126), (681, 111), (671, 175)
]

# Línea Gatillo
linea_teleferico_coords = [
    (561, 110),
    (570, 465)
]

polygon_teleferico_coords = [(149, 136), (435, 114), (445, 417), (161, 443)]

region_points = {
    "personas": polygon_personas_coords,
    "teleferico": polygon_teleferico_coords, 
}

poly_personas = Polygon(polygon_personas_coords)
linea_trigger = LineString(linea_teleferico_coords)

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

# ==========================================
# 3. PREPARACIÓN DE ARCHIVOS
# ==========================================
if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)

timestamp_inicio = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
summary_filename = f"resumen_filtrado_{timestamp_inicio}.csv"
video_filename = f"evidencia_filtrada_{timestamp_inicio}.mp4"

SUMMARY_PATH = os.path.join(OUTPUT_FOLDER, summary_filename)
VIDEO_PATH = os.path.join(OUTPUT_FOLDER, video_filename)

if len(sys.argv) > 1:
    try:
        duration_seconds = float(sys.argv[1]) * 60
    except ValueError:
        duration_seconds = 300 
else:
    duration_seconds = 300

try:
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

cap = cv2.VideoCapture(RTSP_URL)
if not cap.isOpened():
    print("⛔ No se pudo conectar a la cámara RTSP.")
    sys.exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out_vid = cv2.VideoWriter(VIDEO_PATH, fourcc, FPS_VIDEO, (width, height))

summary_f = open(SUMMARY_PATH, 'w', newline='')
summary_writer = csv.writer(summary_f)
summary_writer.writerow(["Inicio_Ciclo", "Fin_Ciclo", "ID_Cabina", "Total_Contados", "Total_Operadores_Excluidos"])

# --- VARIABLES DE ESTADO ---
prev_teleferico_pos = {} 
crossed_ids = set() 
cycle_start_time = datetime.datetime.now().strftime("%H:%M:%S")

person_entry_times = {} 
unique_people_in_cycle = set()

start_time = time.time()
frame_count = 0
ciclos_completados = 0
total_global_personas = 0 
historial_cabinas = deque(maxlen=5) 

print(f"--- INICIO DE MONITOREO ---")
print(f"--- Visualización TOTAL activada (Gris=Fuera, Verde/Rojo=Dentro) ---")

try:
    with torch.no_grad():
        while True:
            current_time_loop = time.time()
            elapsed = current_time_loop - start_time
            if elapsed >= duration_seconds:
                print("\n⏰ Tiempo cumplido.")
                break

            ret, frame = cap.read()
            if not ret:
                print("⚠️ Frame vacío o pérdida de señal.")
                break

            annotated_frame = frame.copy()

            # --- DIBUJO DE ZONAS ---
            pts_poly = np.array(polygon_personas_coords, np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [pts_poly], True, (0, 255, 0), 2)
            
            pt1 = (int(linea_teleferico_coords[0][0]), int(linea_teleferico_coords[0][1]))
            pt2 = (int(linea_teleferico_coords[1][0]), int(linea_teleferico_coords[1][1]))
            cv2.line(annotated_frame, pt1, pt2, (0, 255, 255), 3)

            # Inferencia
            res = rc(frame)
            boxes = to_list(getattr(rc, "boxes", []))
            clss = to_list(getattr(rc, "clss", []))
            track_ids = to_list(getattr(rc, "track_ids", []))
            n = min(len(boxes), len(clss), len(track_ids))
            
            current_telefericos = [] 

            for i in range(n):
                box = boxes[i]
                cls_idx = int(clss[i])
                tid = int(track_ids[i])
                cls_name = CLASS_MAP.get(cls_idx, "unknown")
                
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                pt = Point(cx, cy)

                # ====================================================
                # LOGICA DE VISUALIZACIÓN "SIEMPRE VISIBLE"
                # ====================================================

                # --- CASO PERSONA ---
                if cls_name == "person":
                    # Registrar tiempo desde que aparece EN LA IMAGEN (en cualquier lugar)
                    if tid not in person_entry_times:
                        person_entry_times[tid] = current_time_loop
                    
                    tiempo_en_imagen = current_time_loop - person_entry_times[tid]
                    
                    if poly_personas.contains(pt):
                        # DENTRO DE LA ZONA (Lógica de conteo)
                        if tiempo_en_imagen > UMBRAL_OPERADOR_SEG:
                            color_box = (0, 0, 255) # Rojo (Operador)
                            label_extra = " (OP)"
                        else:
                            color_box = (0, 255, 0) # Verde (Pasajero)
                            label_extra = ""
                            unique_people_in_cycle.add(tid)

                        # Dibujo DENTRO con ID visible
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color_box, 2)
                        cv2.putText(annotated_frame, f"ID:{tid}{label_extra}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 2)
                    else:
                        # FUERA DE LA ZONA (Solo visualización)
                        # Color Gris/Blanco para diferenciar
                        color_box = (200, 200, 200) 
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color_box, 2)
                        cv2.putText(annotated_frame, f"ID:{tid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 2)

                # --- CASO TELEFERICO ---
                elif cls_name == "teleferico":
                    current_telefericos.append((tid, pt))
                    # Siempre visible en Naranja
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cv2.putText(annotated_frame, f"Cabina {tid}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

            # --- LÓGICA DE CIERRE DE CICLO ---
            now_str = datetime.datetime.now().strftime("%H:%M:%S")
            
            for tid, pt in current_telefericos:
                if tid in prev_teleferico_pos:
                    prev_pt = prev_teleferico_pos[tid]
                    movement_line = LineString([(prev_pt.x, prev_pt.y), (pt.x, pt.y)])
                    
                    if movement_line.intersects(linea_trigger):
                        if prev_pt.x < pt.x and tid not in crossed_ids:
                            
                            pasajeros_reales = 0
                            operadores_detectados = 0
                            for pid in unique_people_in_cycle:
                                t_inicio = person_entry_times.get(pid, current_time_loop)
                                if (current_time_loop - t_inicio) <= UMBRAL_OPERADOR_SEG:
                                    pasajeros_reales += 1
                                else:
                                    operadores_detectados += 1

                            total_global_personas += pasajeros_reales

                            historial_cabinas.append(f"Cabina {tid}: {pasajeros_reales} pax")
                            
                            summary_writer.writerow([cycle_start_time, now_str, tid, pasajeros_reales, operadores_detectados])
                            summary_f.flush()
                            
                            unique_people_in_cycle = set()
                            cycle_start_time = now_str
                            crossed_ids.add(tid)
                            ciclos_completados += 1
                            
                            print(f"🚡 Cabina {tid} registrada. Historial: {list(historial_cabinas)}")

                prev_teleferico_pos[tid] = pt

            # --- PANEL DERECHO ---
            pax_esperando_validos = 0
            for pid in unique_people_in_cycle:
                t_inicio = person_entry_times.get(pid, current_time_loop)
                if (current_time_loop - t_inicio) <= UMBRAL_OPERADOR_SEG:
                    pax_esperando_validos += 1
            
            x_panel = width - 380 
            y_panel = 60

            cv2.putText(annotated_frame, f"Esperando: {pax_esperando_validos} ...", (x_panel, y_panel), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            y_panel += 40 

            for linea_texto in reversed(historial_cabinas):
                cv2.putText(annotated_frame, linea_texto, (x_panel, y_panel), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                y_panel += 30 

            out_vid.write(annotated_frame)

            if frame_count % 30 == 0:
                print(f"\r[Rec] T: {elapsed:.0f}s | Cabinas: {ciclos_completados} | Pax: {pax_esperando_validos}", end="")

            if frame_count % 300 == 0:
                gc.collect()
            
            frame_count += 1

except KeyboardInterrupt:
    print("\n🛑 Detenido manualmente.")

except Exception as e:
    print(f"\n⛔ Error: {e}")

finally:
    print("\n--- Generando Resumen Final... ---")
    try:
        summary_writer.writerow([]) 
        summary_writer.writerow([]) 
        summary_writer.writerow(["=== RESUMEN TOTAL ==="])
        summary_writer.writerow(["TOTAL_CABINAS", "TOTAL_PERSONAS"])
        summary_writer.writerow([ciclos_completados, total_global_personas])
    except Exception as e:
        print(f"Error escribiendo resumen final: {e}")

    summary_f.close()
    if 'out_vid' in locals(): out_vid.release()
    cap.release()
    print(f"✅ Finalizado. Archivos guardados.")
