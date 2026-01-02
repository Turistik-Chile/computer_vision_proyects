"""
Sistema de Conteo de Pasajeros para Teleférico - Versión Server
Optimizado para Ubuntu Server sin interfaz gráfica.
Integra filtrado avanzado de operadores y conteo direccional.
"""

import cv2
import time
import datetime
import csv
import torch
import gc
import os
import sys
import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from shapely.geometry import Point, Polygon, LineString
from ultralytics import YOLO

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
# VIDEO_PATH se puede pasar como argumento: python claude.py video.mp4
VIDEO_PATH = "/home/aaravenatk/video/Oasis1_1718.mp4"  # Ruta por defecto, se puede cambiar por argumento
MODEL_PATH = "/home/aaravenatk/yolo/best.pt"
CONF_THRESHOLD = 0.7
OUTPUT_FOLDER = "detecciones"
FPS_VIDEO = None  # Se detecta automáticamente del video, o usar valor fijo si falla

# --- FILTRO DE OPERADORES ---
# El ÚNICO criterio para filtrar operadores es el tiempo en escena.
# Si una persona lleva más de este tiempo en la imagen, se considera operador.
UMBRAL_OPERADOR_SEG = 45.0          # Segundos máximos para ser considerado pasajero

# --- DIRECCIÓN DEL CRUCE DE CABINAS ---
# "izq_a_der" = cabinas van de izquierda a derecha (prev_x < current_x)
# "der_a_izq" = cabinas van de derecha a izquierda (prev_x > current_x)
DIRECCION_CRUCE = "izq_a_der"  # Cambiar según la dirección real de las cabinas

# Zona donde se detectan personas (área de embarque/fila)
polygon_personas_coords = [
    (477, 171), (478, 345), (765, 372), (1020, 264),
    (1034, 126), (681, 111), (671, 175)
]

# Línea de cruce (gatillo) - cuando la cabina cruza esta línea, se cierra el ciclo
linea_teleferico_coords = [
    (561, 110),
    (570, 465)
]

# Zona de la cabina/teleférico
polygon_teleferico_coords = [(149, 136), (435, 114), (445, 417), (161, 443)]

CLASS_MAP = {0: "person", 1: "teleferico"}

# ==========================================
# 2. CLASE PARA TRACKING AVANZADO DE PERSONAS
# ==========================================
@dataclass
class PersonTrack:
    """Representa una persona trackeada - simplificado para filtro por tiempo"""
    track_id: int
    positions: deque = field(default_factory=lambda: deque(maxlen=150))  # ~10 seg a 15fps
    first_seen: float = 0
    last_seen: float = 0
    was_in_zone: bool = False
    is_in_zone: bool = False  # Estado actual
    is_operator: bool = False
    boarded: bool = False  # Si ya abordó una cabina
    zone_exit_time: Optional[float] = None  # Cuando salió de la zona
    
    def add_position(self, x: float, y: float, timestamp: float, in_zone: bool):
        """Agrega una posición al historial"""
        self.positions.append((x, y, timestamp))
        self.last_seen = timestamp
        
        if self.first_seen == 0:
            self.first_seen = timestamp
        
        # Detectar cuando SALE de la zona (posible embarque)
        if self.is_in_zone and not in_zone:
            self.zone_exit_time = timestamp
        
        # Actualizar estados
        if in_zone:
            self.was_in_zone = True  # Una vez en zona, siempre true
        self.is_in_zone = in_zone
    
    def time_in_scene(self, current_time: float) -> float:
        """Tiempo total en escena - ÚNICO CRITERIO PARA FILTRAR OPERADORES"""
        return current_time - self.first_seen


class AdvancedPassengerCounter:
    """Sistema de conteo con filtro por tiempo en escena"""
    
    def __init__(self, config: dict):
        self.config = config
        
        # Geometrías
        self.poly_personas = Polygon(config['polygon_personas'])
        self.linea_trigger = LineString(config['linea_teleferico'])
        
        # Tracking de personas
        self.person_tracks: Dict[int, PersonTrack] = {}
        
        # Estado del ciclo actual
        # CAMBIO CRÍTICO: Guardamos personas que SALIERON de la zona (probable embarque)
        self.persons_who_left_zone: Set[int] = set()  # Personas que salieron de zona en este ciclo
        self.persons_in_zone: Set[int] = set()  # Personas actualmente en zona
        self.cycle_start_time = 0.0
        
        # Estado de teleféricos
        self.prev_teleferico_pos: Dict[int, Point] = {}
        self.crossed_cabin_ids: Set[int] = set()
        self.cabin_in_zone: bool = False  # Si hay cabina en zona de embarque
        self.current_cabin_id: Optional[int] = None
        
        # Estadísticas
        self.total_passengers = 0
        self.total_operators_filtered = 0
        self.cycles_completed = 0
        self.cabin_history: deque = deque(maxlen=10)
        
    def update_person(self, track_id: int, bbox: Tuple[int, int, int, int], 
                      current_time: float) -> Tuple[str, str]:
        """
        Actualiza tracking de persona y retorna su clasificación.
        ÚNICO CRITERIO: Tiempo en escena > umbral = operador
        Returns: (status, reason) donde status es 'passenger', 'operator', 'unknown'
        """
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        pt = Point(cx, cy)
        
        in_zone = self.poly_personas.contains(pt)
        
        # Crear o actualizar track
        if track_id not in self.person_tracks:
            self.person_tracks[track_id] = PersonTrack(track_id=track_id)
        
        track = self.person_tracks[track_id]
        was_in_zone_before = track.is_in_zone
        track.add_position(cx, cy, current_time, in_zone)
        
        # === CLASIFICACIÓN POR TIEMPO EN ESCENA ===
        time_in_scene = track.time_in_scene(current_time)
        
        if time_in_scene > self.config['umbral_operador_seg']:
            track.is_operator = True
            # Remover de sets si estaba
            self.persons_in_zone.discard(track_id)
            self.persons_who_left_zone.discard(track_id)
            return ('operator', f'{time_in_scene:.0f}s')
        
        # Lógica de zona
        if in_zone:
            self.persons_in_zone.add(track_id)
            return ('passenger', f'{time_in_scene:.0f}s')
        else:
            # Si SALIÓ de la zona (estaba antes, ya no está)
            if was_in_zone_before and not in_zone:
                # Persona salió de la zona → probable embarque
                if not track.boarded:
                    self.persons_who_left_zone.add(track_id)
            
            self.persons_in_zone.discard(track_id)
            return ('unknown', 'fuera_zona')
    
    def check_cabin_crossing(self, cabin_id: int, cabin_point: Point, 
                             current_time: float) -> Optional[dict]:
        """
        Verifica si una cabina cruzó la línea y cierra el ciclo.
        Returns: dict con info del ciclo si se cerró, None si no.
        """
        result = None
        
        if cabin_id in self.prev_teleferico_pos:
            prev_pt = self.prev_teleferico_pos[cabin_id]
            movement_line = LineString([(prev_pt.x, prev_pt.y), (cabin_point.x, cabin_point.y)])
            
            intersects = movement_line.intersects(self.linea_trigger)
            
            if intersects:
                # Verificar dirección del cruce según configuración
                direccion = self.config.get('direccion_cruce', 'izq_a_der')
                
                if direccion == "izq_a_der":
                    # Cabinas van de izquierda a derecha
                    cruce_valido = prev_pt.x < cabin_point.x
                else:
                    # Cabinas van de derecha a izquierda
                    cruce_valido = prev_pt.x > cabin_point.x
                
                if cruce_valido and cabin_id not in self.crossed_cabin_ids:
                    # === CERRAR CICLO ===
                    result = self._close_cycle(cabin_id, current_time)
                    self.crossed_cabin_ids.add(cabin_id)
        
        self.prev_teleferico_pos[cabin_id] = cabin_point
        return result
    
    def _close_cycle(self, cabin_id: int, current_time: float) -> dict:
        """Cierra el ciclo actual y cuenta pasajeros"""
        
        passengers = 0
        operators = 0
        passenger_ids = []
        operator_ids = []
        
        # CAMBIO CRÍTICO: Contar personas que SALIERON de la zona durante este ciclo
        # Estas son las que probablemente abordaron la cabina
        persons_to_count = self.persons_who_left_zone.copy()
        
        for person_id in persons_to_count:
            if person_id not in self.person_tracks:
                continue
                
            track = self.person_tracks[person_id]
            
            if track.is_operator:
                operators += 1
                operator_ids.append(person_id)
            else:
                passengers += 1
                passenger_ids.append(person_id)
                track.boarded = True  # Marcar como abordado
        
        # Actualizar estadísticas
        self.total_passengers += passengers
        self.total_operators_filtered += operators
        self.cycles_completed += 1
        
        cycle_info = {
            'cabin_id': cabin_id,
            'passengers': passengers,
            'operators_filtered': operators,
            'passenger_ids': passenger_ids,
            'operator_ids': operator_ids,
            'cycle_duration': current_time - self.cycle_start_time if self.cycle_start_time > 0 else 0,
            'timestamp': datetime.datetime.now().strftime("%H:%M:%S")
        }
        
        self.cabin_history.append(f"Cabina {cabin_id}: {passengers} pax")
        
        # Reset para nuevo ciclo - SOLO limpiar los que abordaron
        self.persons_who_left_zone.clear()
        self.cycle_start_time = current_time
        
        return cycle_info
    
    def get_waiting_count(self, current_time: float) -> int:
        """Cuenta personas válidas esperando actualmente en zona"""
        count = 0
        for person_id in self.persons_in_zone:
            if person_id in self.person_tracks:
                track = self.person_tracks[person_id]
                if not track.is_operator and not track.boarded:
                    count += 1
        return count
    
    def cleanup_old_tracks(self, current_time: float, max_age: float = 60.0):
        """Limpia tracks antiguos"""
        to_remove = []
        for track_id, track in self.person_tracks.items():
            if current_time - track.last_seen > max_age:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.person_tracks[track_id]
            self.persons_in_zone.discard(track_id)
            self.persons_who_left_zone.discard(track_id)


# ==========================================
# 3. FUNCIONES AUXILIARES
# ==========================================
def to_list(x):
    """Convierte tensores/arrays a lista"""
    if x is None:
        return []
    try:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except:
        pass
    try:
        if isinstance(x, np.ndarray):
            return x.tolist()
    except:
        pass
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def setup_output_folder(folder: str) -> Tuple[str, str]:
    """Crea carpeta y retorna paths de archivos de salida"""
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_path = os.path.join(folder, f"conteo_pasajeros_{timestamp}.csv")
    video_path = os.path.join(folder, f"evidencia_{timestamp}.mp4")
    
    return summary_path, video_path


def draw_annotations(frame: np.ndarray, config: dict, counter: AdvancedPassengerCounter,
                     detections: list, current_time: float) -> np.ndarray:
    """Dibuja anotaciones en el frame"""
    annotated = frame.copy()
    height, width = frame.shape[:2]
    
    # Dibujar zona de personas
    pts = np.array(config['polygon_personas'], np.int32).reshape((-1, 1, 2))
    cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)
    
    # Dibujar línea trigger
    p1 = tuple(config['linea_teleferico'][0])
    p2 = tuple(config['linea_teleferico'][1])
    cv2.line(annotated, p1, p2, (0, 255, 255), 3)
    
    # Dibujar detecciones
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cls_name = det['class']
        track_id = det['track_id']
        status = det.get('status', 'unknown')
        reason = det.get('reason', '')
        
        if cls_name == "person":
            if status == 'operator':
                color = (0, 0, 255)  # Rojo
                label = f"ID:{track_id} [OP:{reason[:10]}]"
            elif status == 'passenger':
                color = (0, 255, 0)  # Verde
                label = f"ID:{track_id}"
            else:
                color = (200, 200, 200)  # Gris
                label = f"ID:{track_id}"
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Dibujar trayectoria si está en zona
            if track_id in counter.person_tracks and status == 'passenger':
                track = counter.person_tracks[track_id]
                positions = list(track.positions)[-20:]  # Últimas 20 posiciones
                for i in range(1, len(positions)):
                    pt1 = (int(positions[i-1][0]), int(positions[i-1][1]))
                    pt2 = (int(positions[i][0]), int(positions[i][1]))
                    cv2.line(annotated, pt1, pt2, (255, 255, 0), 1)
        
        elif cls_name == "teleferico":
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(annotated, f"Cabina {track_id}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    
    # Panel de información
    waiting = counter.get_waiting_count(current_time)
    x_panel = width - 400
    y_panel = 30
    
    # Fondo semi-transparente para el panel
    overlay = annotated.copy()
    cv2.rectangle(overlay, (x_panel - 10, 10), (width - 10, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)
    
    cv2.putText(annotated, f"Esperando: {waiting}", (x_panel, y_panel),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    y_panel += 35
    cv2.putText(annotated, f"Total Pax: {counter.total_passengers}", (x_panel, y_panel),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    y_panel += 30
    cv2.putText(annotated, f"Ciclos: {counter.cycles_completed}", (x_panel, y_panel),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    y_panel += 35
    cv2.putText(annotated, "Historial:", (x_panel, y_panel),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    for hist_line in reversed(list(counter.cabin_history)[-5:]):
        y_panel += 22
        cv2.putText(annotated, hist_line, (x_panel, y_panel),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return annotated


# ==========================================
# 4. FUNCIÓN PRINCIPAL
# ==========================================
def main():
    print("="*60)
    print("SISTEMA DE CONTEO DE PASAJEROS - TELEFÉRICO")
    print("="*60)
    
    # Configuración simplificada - solo tiempo en escena como filtro
    config = {
        'polygon_personas': polygon_personas_coords,
        'linea_teleferico': linea_teleferico_coords,
        'umbral_operador_seg': UMBRAL_OPERADOR_SEG,
        'direccion_cruce': DIRECCION_CRUCE,
    }
    
    # Parsear ruta de video desde argumentos
    # Uso: python claude.py <ruta_video>
    if len(sys.argv) > 1:
        video_input = sys.argv[1]
    else:
        video_input = VIDEO_PATH
    
    # Verificar que el archivo existe
    if not os.path.exists(video_input):
        print(f"⛔ Error: No se encontró el archivo de video: {video_input}")
        print(f"\nUso: python {sys.argv[0]} <ruta_video.mp4>")
        sys.exit(1)
    
    print(f"Video de entrada: {video_input}")
    
    # Preparar archivos de salida
    summary_path, video_path = setup_output_folder(OUTPUT_FOLDER)
    print(f"CSV: {summary_path}")
    print(f"Video: {video_path}")
    
    # Cargar modelo YOLO
    print(f"\nCargando modelo: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
        print(f"Clases del modelo: {model.names}")
    except Exception as e:
        print(f"⛔ Error cargando modelo: {e}")
        sys.exit(1)
    
    # Abrir archivo de video
    print(f"\nAbriendo video: {video_input}")
    cap = cv2.VideoCapture(video_input)
    if not cap.isOpened():
        print(f"⛔ No se pudo abrir el archivo de video: {video_input}")
        sys.exit(1)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_original = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_video = total_frames / fps_original if fps_original > 0 else 0
    
    # Usar FPS del video original o el configurado
    fps_output = FPS_VIDEO if FPS_VIDEO else fps_original
    
    print(f"Resolución: {width}x{height}")
    print(f"FPS original: {fps_original:.1f}")
    print(f"Total frames: {total_frames}")
    print(f"Duración: {duration_video/60:.1f} minutos")
    
    # Configurar video de salida
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(video_path, fourcc, fps_output, (width, height))
    
    # Configurar CSV
    summary_f = open(summary_path, 'w', newline='', encoding='utf-8')
    summary_writer = csv.writer(summary_f)
    summary_writer.writerow([
        "Timestamp", "ID_Cabina", "Pasajeros", "Operadores_Filtrados",
        "IDs_Pasajeros", "IDs_Operadores", "Duracion_Ciclo_seg"
    ])
    
    # Inicializar contador avanzado
    counter = AdvancedPassengerCounter(config)
    
    # Variables de control
    start_time = time.time()
    frame_count = 0
    
    # IMPORTANTE: Usar tiempo del video, no tiempo real
    # Esto es crítico para videos pregrabados
    video_time = 0.0  # Tiempo simulado basado en frames
    
    print(f"\n{'='*60}")
    print("PROCESANDO VIDEO...")
    print(f"{'='*60}\n")
    
    try:
        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("\n✅ Video procesado completamente.")
                    break
                
                # CRÍTICO: Usar tiempo del video, no tiempo real
                # Cada frame avanza 1/fps segundos en el video
                video_time = frame_count / fps_original
                current_time = video_time  # Usar tiempo del video para toda la lógica
                
                # Ejecutar YOLO con tracking
                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=CONF_THRESHOLD,
                    verbose=False
                )
                
                # Procesar detecciones
                detections = []
                
                for r in results:
                    if r.boxes is None:
                        continue
                    
                    boxes = r.boxes
                    
                    for box in boxes:
                        cls_idx = int(box.cls[0])
                        conf = float(box.conf[0])
                        bbox = tuple(box.xyxy[0].cpu().numpy().astype(int))
                        track_id = int(box.id[0]) if box.id is not None else None
                        
                        if track_id is None:
                            continue
                        
                        cls_name = CLASS_MAP.get(cls_idx, "unknown")
                        
                        det = {
                            'bbox': bbox,
                            'class': cls_name,
                            'track_id': track_id,
                            'confidence': conf
                        }
                        
                        if cls_name == "person":
                            status, reason = counter.update_person(track_id, bbox, current_time)
                            det['status'] = status
                            det['reason'] = reason
                        
                        elif cls_name == "teleferico":
                            cx = (bbox[0] + bbox[2]) / 2
                            cy = (bbox[1] + bbox[3]) / 2
                            cabin_point = Point(cx, cy)
                            
                            cycle_info = counter.check_cabin_crossing(track_id, cabin_point, current_time)
                            
                            if cycle_info:
                                # Escribir al CSV
                                summary_writer.writerow([
                                    cycle_info['timestamp'],
                                    cycle_info['cabin_id'],
                                    cycle_info['passengers'],
                                    cycle_info['operators_filtered'],
                                    str(cycle_info['passenger_ids']),
                                    str(cycle_info['operator_ids']),
                                    f"{cycle_info['cycle_duration']:.1f}"
                                ])
                                summary_f.flush()
                                
                                print(f"\n🚡 CABINA {cycle_info['cabin_id']}: "
                                      f"{cycle_info['passengers']} pasajeros "
                                      f"({cycle_info['operators_filtered']} operadores filtrados)")
                        
                        detections.append(det)
                
                # Anotar frame
                annotated_frame = draw_annotations(frame, config, counter, detections, current_time)
                
                # Guardar frame
                out_vid.write(annotated_frame)
                
                # Progreso
                if frame_count % 45 == 0:  # Cada ~3 segundos
                    waiting = counter.get_waiting_count(current_time)
                    progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                    print(f"\r[{progress:.1f}%] Frame {frame_count}/{total_frames} | "
                          f"Ciclos: {counter.cycles_completed} | "
                          f"Total Pax: {counter.total_passengers} | "
                          f"Esperando: {waiting}", end="", flush=True)
                
                # Limpieza periódica
                if frame_count % 450 == 0:  # Cada ~30 segundos
                    counter.cleanup_old_tracks(current_time)
                    gc.collect()
                
                frame_count += 1
    
    except KeyboardInterrupt:
        print("\n\n🛑 Detenido manualmente.")
    
    except Exception as e:
        print(f"\n⛔ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Resumen final
        print("\n\n" + "="*60)
        print("RESUMEN FINAL")
        print("="*60)
        print(f"Ciclos completados: {counter.cycles_completed}")
        print(f"Total pasajeros: {counter.total_passengers}")
        print(f"Operadores filtrados: {counter.total_operators_filtered}")
        print(f"Promedio por cabina: {counter.total_passengers/max(1, counter.cycles_completed):.1f}")
        
        # Escribir resumen al CSV
        summary_writer.writerow([])
        summary_writer.writerow(["=== RESUMEN TOTAL ==="])
        summary_writer.writerow(["Total_Ciclos", "Total_Pasajeros", "Total_Operadores_Filtrados", "Promedio_por_Cabina"])
        summary_writer.writerow([
            counter.cycles_completed,
            counter.total_passengers,
            counter.total_operators_filtered,
            f"{counter.total_passengers/max(1, counter.cycles_completed):.1f}"
        ])
        
        # Cerrar recursos
        summary_f.close()
        out_vid.release()
        cap.release()
        
        print(f"\n✅ Archivos guardados en: {OUTPUT_FOLDER}/")
        print("="*60)


if __name__ == "__main__":
    main()
