#!/usr/bin/env python3
"""
Sistema Avanzado de Conteo de Pasajeros en Teleférico v2.2 (Final - Fixed)
==========================================================================
Lógica de conteo:
1. Detección de cabinas cruzando línea trigger.
2. Tracking de personas con persistencia y re-identificación.
3. Filtrado de operadores: Estrictamente por tiempo de permanencia (>90s).
4. Conteo: Acumulativo (personas que desaparecen en zona de embarque).
5. Visualización: Zonas transparentes para mejor supervisión.

Uso: python teleferico_counter_v2.py <minutos>
"""

import cv2
import numpy as np
import argparse
import time
import csv
import os
import tempfile
from datetime import datetime
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
from ultralytics import YOLO
import logging

# ============================================================================
# CONFIGURACIÓN DE BYTETRACK
# ============================================================================
BYTETRACK_CONFIG = """
tracker_type: bytetrack
track_high_thresh: 0.5
track_low_thresh: 0.1
new_track_thresh: 0.6
track_buffer: 45
match_thresh: 0.8
fuse_score: true
"""

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURACIÓN GENERAL
# ============================================================================

RTSP_URL = "rtsp://usertf:Tfo.-2525@172.16.10.22:554/cam/realmonitor?channel=5&subtype=0"
MODEL_PATH = "/home/aaravenatk/yolo/best.pt"
CONF_THRESHOLD = 0.6
OUTPUT_FOLDER = "detecciones"
FPS_VIDEO = 15.0
OPACIDAD_ZONAS = 0.08  # Transparencia de las zonas (0.0 a 1.0).

# --- Geometría de la escena ---
LINEA_TELEFERICO = [(561, 110), (570, 465)]

ZONA_EMBARQUE = [
    (570, 168), (809, 164), (897, 159),
    (984, 185), (997, 388), (575, 359)
]

ZONA_PAX1 = [(862, 717), (1022, 569), (1030, 452), (995, 330), (657, 314), (681, 375), (471, 436), (286, 498), (141, 544), (2, 591), (2, 719), (306, 719), (550, 717)]
TRANSVERSALES1 = [(5, 2), (5, 1), (6, 0), (7, 12), (8, 11), (9, 11)]

ZONA_PAX2 = [(1278, 719), (1279, 594), (1282, 504), (1274, 390), (1183, 344), (1082, 296), (970, 390), (1028, 485), (1014, 549), (904, 654), (829, 719)]
TRANSVERSALES2 = [(9, 1), (8, 2), (7, 3), (7, 4)]

# --- Parámetros de comportamiento ---
TIEMPO_OPERADOR = 90.0          # Si está >90s en escena, es operador (no se cuenta)
TIEMPO_REFRESCO_TRACKS = 60.0   # Limpieza de memoria
TIEMPO_REIDENTIFICACION = 10.0  # Buffer para recuperar tracks perdidos
DISTANCIA_MAX_REIDENT = 200     # Pixeles max para re-identificar
SIMILARIDAD_BBOX_MIN = 0.6      

# Clases del modelo
CLASE_CABEZA = 0
CLASE_CABINA = 1

# ============================================================================
# ESTRUCTURAS DE DATOS
# ============================================================================

@dataclass
class PersonTracker:
    """Tracking avanzado de una persona."""
    track_id: int
    first_seen: float
    last_seen: float
    history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Estados
    in_zone: bool = False
    is_operator: bool = False
    is_passenger: bool = False
    
    # Tracking de zonas PAX
    in_pax1: bool = False
    in_pax2: bool = False
    transversales_pax1_crossed: Set[int] = field(default_factory=set)
    transversales_pax2_crossed: Set[int] = field(default_factory=set)
    
    # Para análisis de dirección
    avg_velocity: Tuple[float, float] = (0.0, 0.0)
    
    # Tracking de zonas
    zone_entry_count: int = 0
    zone_exit_count: int = 0
    was_in_zone: bool = False
    
    # Para re-identificación
    last_bbox_size: Tuple[int, int] = (0, 0)
    is_lost: bool = False
    lost_timestamp: float = 0.0
    
    def add_position(self, x: float, y: float, timestamp: float, bbox: Tuple[int, int, int, int] = None):
        self.history.append((x, y, timestamp))
        self.last_seen = timestamp
        if bbox:
            self.last_bbox_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
        self._update_velocity()
    
    def predict_position(self, delta_time: float) -> Tuple[float, float]:
        if not self.last_position:
            return (0, 0)
        pred_x = self.last_position[0] + self.avg_velocity[0] * delta_time
        pred_y = self.last_position[1] + self.avg_velocity[1] * delta_time
        return (pred_x, pred_y)
    
    def bbox_similarity(self, other_bbox_size: Tuple[int, int]) -> float:
        if self.last_bbox_size == (0, 0) or other_bbox_size == (0, 0):
            return 0.0
        w1, h1 = self.last_bbox_size
        w2, h2 = other_bbox_size
        w_ratio = min(w1, w2) / max(w1, w2) if max(w1, w2) > 0 else 0
        h_ratio = min(h1, h2) / max(h1, h2) if max(h1, h2) > 0 else 0
        return (w_ratio + h_ratio) / 2
    
    def _update_velocity(self):
        if len(self.history) < 5: return
        recent = list(self.history)[-10:]
        if len(recent) < 2: return
        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        dt = recent[-1][2] - recent[0][2]
        if dt > 0:
            self.avg_velocity = (dx/dt, dy/dt)
    
    def update_zone_status(self, currently_in_zone: bool):
        if currently_in_zone and not self.was_in_zone:
            self.zone_entry_count += 1
        elif not currently_in_zone and self.was_in_zone:
            self.zone_exit_count += 1
        self.was_in_zone = self.in_zone
        self.in_zone = currently_in_zone
    
    def get_pax1_coverage(self) -> float:
        total = len(TRANSVERSALES1)
        return len(self.transversales_pax1_crossed) / total if total > 0 else 0.0
    
    def get_pax2_coverage(self) -> float:
        total = len(TRANSVERSALES2)
        return len(self.transversales_pax2_crossed) / total if total > 0 else 0.0
    
    def meets_passenger_criteria(self) -> bool:
        return self.in_pax1 or self.in_pax2
    
    @property
    def time_tracked(self) -> float:
        return self.last_seen - self.first_seen
    
    @property
    def last_position(self) -> Optional[Tuple[float, float]]:
        if self.history:
            return (self.history[-1][0], self.history[-1][1])
        return None

@dataclass  
class CabinEvent:
    """Evento de paso de cabina."""
    cabin_id: int
    track_id: Optional[int]
    cross_time: float
    persons_before: Set[int] = field(default_factory=set)
    persons_embarked: int = 0
    embarked_ids: List[int] = field(default_factory=list)


# ============================================================================
# UTILIDADES GEOMÉTRICAS
# ============================================================================

def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[int, int]]) -> bool:
    return cv2.pointPolygonTest(np.array(polygon, np.int32), point, False) >= 0

def line_side(point: Tuple[float, float], line: List[Tuple[int, int]]) -> float:
    x, y = point
    (x1, y1), (x2, y2) = line
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)

def crossed_line(prev_pos: Tuple[float, float], curr_pos: Tuple[float, float],
                 line: List[Tuple[int, int]]) -> Tuple[bool, str]:
    prev_side = line_side(prev_pos, line)
    curr_side = line_side(curr_pos, line)
    if prev_side * curr_side < 0:
        direction = 'left' if curr_side > 0 else 'right'
        return True, direction
    return False, ''

def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

def bbox_bottom(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2, bbox[3])

def check_transversal_crossings(prev_pos, curr_pos, zona, transversales) -> Set[int]:
    if not prev_pos: return set()
    crossed = set()
    for idx, (p1_idx, p2_idx) in enumerate(transversales):
        line = [zona[p1_idx], zona[p2_idx]]
        crossed_line_result, _ = crossed_line(prev_pos, curr_pos, line)
        if crossed_line_result:
            crossed.add(idx)
    return crossed


# ============================================================================
# CLASE PRINCIPAL
# ============================================================================

class TelefericoCounterV2:
    def __init__(self, model_path: str, rtsp_url: str, output_folder: str,
                 conf_threshold: float = 0.7, fps_video: float = 15.0):
        
        self.model_path = model_path
        self.rtsp_url = rtsp_url
        self.output_folder = output_folder
        self.conf_threshold = conf_threshold
        self.fps_video = fps_video
        
        logger.info(f"Cargando modelo: {model_path}")
        self.model = YOLO(model_path)
        
        self.persons: Dict[int, PersonTracker] = {}
        self.prev_cabin_positions: Dict[int, Tuple[int, int, int, int]] = {}
        
        self.cabin_count = 0
        self.cabin_events: List[CabinEvent] = []
        self.current_event: Optional[CabinEvent] = None
        
        self.persons_in_zone: Set[int] = set()
        self.already_counted: Set[int] = set()
        self.total_embarked_count: int = 0
        self.lost_persons: Dict[int, PersonTracker] = {}
        self.track_id_map: Dict[int, int] = {}  # Mapa de IDs raw -> IDs persistentes
        self.csv_path = None
        
        self.tracker_config_path = self._create_tracker_config()
        
        os.makedirs(output_folder, exist_ok=True)
        self.timestamp_str = datetime.now().strftime("%d-%m-%y_%H-%M-%S")
    
    def _create_tracker_config(self) -> str:
        config_path = os.path.join(tempfile.gettempdir(), "bytetrack_teleferico.yaml")
        with open(config_path, 'w') as f:
            f.write(BYTETRACK_CONFIG)
        return config_path
    
    def _reidentify_person(self, bbox: Tuple[int, int, int, int], pos: Tuple[float, float], 
                           current_time: float) -> Optional[int]:
        best_match_id = None
        best_score = 0.0
        bbox_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
        
        for lost_id, lost_person in list(self.lost_persons.items()):
            time_lost = current_time - lost_person.lost_timestamp
            if time_lost > TIEMPO_REIDENTIFICACION: continue
            
            predicted_pos = lost_person.predict_position(time_lost)
            distance = np.sqrt((pos[0] - predicted_pos[0])**2 + (pos[1] - predicted_pos[1])**2)
            size_similarity = lost_person.bbox_similarity(bbox_size)
            
            distance_score = max(0, 1 - (distance / DISTANCIA_MAX_REIDENT))
            combined_score = (distance_score * 0.6) + (size_similarity * 0.4)
            
            if distance < DISTANCIA_MAX_REIDENT and size_similarity > SIMILARIDAD_BBOX_MIN:
                if combined_score > best_score:
                    best_score = combined_score
                    best_match_id = lost_id
        
        if best_match_id:
            logger.info(f"Re-identificación exitosa: ID {best_match_id} (score: {best_score:.2f})")
        return best_match_id
    
    def _detect_operator(self, person: PersonTracker, current_time: float) -> bool:
        """
        Detecta operador ÚNICAMENTE por tiempo de permanencia.
        """
        if person.time_tracked > TIEMPO_OPERADOR:
            if not person.is_operator:
                logger.debug(f"Operador {person.track_id} detectado por tiempo ({person.time_tracked:.1f}s)")
            return True
        return False
    
    def _process_frame(self, frame: np.ndarray, current_time: float) -> Tuple[List, List]:
        results = self.model.track(
            frame, persist=True, tracker=self.tracker_config_path,
            conf=self.conf_threshold, verbose=False
        )
        heads = []
        cabins = []
        
        for result in results:
            if result.boxes is None: continue
            for box in result.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                bbox = tuple(map(int, box.xyxy[0].tolist()))
                track_id = int(box.id[0]) if hasattr(box, 'id') and box.id is not None else None
                detection = {'bbox': bbox, 'conf': conf, 'track_id': track_id}
                
                if cls == CLASE_CABEZA:
                    heads.append(detection)
                elif cls == CLASE_CABINA:
                    cabins.append(detection)
        return heads, cabins
    
    def _update_persons(self, heads: List, current_time: float) -> List[int]:
        """
        Actualiza tracking y realiza CONTEO POR DESAPARICIÓN.
        """
        embarked_this_frame = []
        current_ids = set()
        
        # [CORREGIDO] Iteración correcta sobre la lista heads
        for head in heads:
            raw_track_id = head['track_id']
            bbox = head['bbox']
            pos = bbox_bottom(bbox)
            
            track_id = raw_track_id

            # 1. Manejo de IDs nulos o nuevos (Re-identificación)
            if raw_track_id is None:
                # Caso: Detección sin ID del tracker
                reidentified_id = self._reidentify_person(bbox, pos, current_time)
                if reidentified_id is not None:
                    track_id = reidentified_id
                    if reidentified_id in self.lost_persons:
                        person = self.lost_persons.pop(reidentified_id)
                        person.is_lost = False
                        self.persons[track_id] = person
                else:
                    continue
            else:
                # Caso: ID válido del tracker
                # Verificar si este ID raw ya está mapeado a un ID antiguo
                if raw_track_id in self.track_id_map:
                    track_id = self.track_id_map[raw_track_id]
                
                # Si el ID (mapeado o no) es nuevo para nosotros, intentar re-identificar
                # Esto maneja el caso donde el tracker pierde el rastro y asigna un ID nuevo
                if track_id not in self.persons:
                    reidentified_id = self._reidentify_person(bbox, pos, current_time)
                    if reidentified_id is not None:
                        # ¡Encontramos que este "nuevo" ID es en realidad alguien perdido!
                        # Mapeamos el ID raw del tracker al ID antiguo
                        self.track_id_map[raw_track_id] = reidentified_id
                        track_id = reidentified_id
                        
                        if reidentified_id in self.lost_persons:
                            person = self.lost_persons.pop(reidentified_id)
                            person.is_lost = False
                            self.persons[track_id] = person
                            logger.info(f"Persistencia: ID {raw_track_id} unificado con ID {track_id}")

            current_ids.add(track_id)
            
            # Crear nuevo tracker si definitivamente es nuevo
            if track_id not in self.persons:
                in_zone = point_in_polygon(pos, ZONA_EMBARQUE)
                in_pax1 = point_in_polygon(pos, ZONA_PAX1)
                in_pax2 = point_in_polygon(pos, ZONA_PAX2)
                self.persons[track_id] = PersonTracker(
                    track_id=track_id, first_seen=current_time, last_seen=current_time,
                    in_zone=in_zone, in_pax1=in_pax1, in_pax2=in_pax2
                )
            
            person = self.persons[track_id]
            prev_pos = person.last_position
            person.add_position(pos[0], pos[1], current_time, bbox)
            
            in_zone = point_in_polygon(pos, ZONA_EMBARQUE)
            person.update_zone_status(in_zone)
            person.in_pax1 = point_in_polygon(pos, ZONA_PAX1)
            person.in_pax2 = point_in_polygon(pos, ZONA_PAX2)
            
            # Verificar transversales
            if prev_pos:
                if person.in_pax1:
                    crossed = check_transversal_crossings(prev_pos, pos, ZONA_PAX1, TRANSVERSALES1)
                    person.transversales_pax1_crossed.update(crossed)
                if person.in_pax2:
                    crossed = check_transversal_crossings(prev_pos, pos, ZONA_PAX2, TRANSVERSALES2)
                    person.transversales_pax2_crossed.update(crossed)
            
            if not person.is_passenger:
                person.is_passenger = person.meets_passenger_criteria()
            
            # Actualizar estado de operador (Sticky: si ya es True, no cambia)
            if not person.is_operator:
                person.is_operator = self._detect_operator(person, current_time)
            
            # Gestión de lista visual
            if in_zone and not person.is_operator:
                self.persons_in_zone.add(track_id)
            else:
                self.persons_in_zone.discard(track_id)
        
        # --- LÓGICA DE CONTEO ACUMULATIVO ---
        all_known_ids = set(self.persons.keys())
        disappeared_ids = all_known_ids - current_ids
        
        for track_id in disappeared_ids:
            if track_id in self.persons:
                person = self.persons[track_id]
                
                if person.is_lost: continue

                # Marcar como perdido
                person.is_lost = True
                person.lost_timestamp = current_time
                self.lost_persons[track_id] = person
                
                # REGLA DE CONTEO
                if person.in_zone and not person.is_operator and self.current_event:
                    
                    if track_id not in self.current_event.embarked_ids:
                        self.current_event.persons_embarked += 1
                        self.current_event.embarked_ids.append(track_id)
                        self.total_embarked_count += 1
                        embarked_this_frame.append(track_id)
                        self.already_counted.add(track_id)
                        
                        logger.info(f"--> Pasajero {track_id} EMBARCÓ. (Acumulado cabina #{self.current_event.cabin_id}: {self.current_event.persons_embarked})")

        # Limpiar tracks perdidos expirados
        expired_lost = []
        for lost_id, lost_person in self.lost_persons.items():
            if current_time - lost_person.lost_timestamp > TIEMPO_REIDENTIFICACION:
                expired_lost.append(lost_id)
                self.persons_in_zone.discard(lost_id)
        
        for lost_id in expired_lost:
            del self.lost_persons[lost_id]
        
        return embarked_this_frame
    
    def _check_cabin_crossing(self, cabins: List, current_time: float) -> bool:
        cabin_crossed = False
        for cabin in cabins:
            track_id = cabin['track_id']
            if track_id is None: continue
            
            curr_bbox = cabin['bbox']
            curr_center = bbox_center(curr_bbox)
            prev_bbox = self.prev_cabin_positions.get(track_id)
            
            if prev_bbox:
                prev_center = bbox_center(prev_bbox)
                crossed, _ = crossed_line(prev_center, curr_center, LINEA_TELEFERICO)
                if crossed:
                    cabin_crossed = True
                    self._handle_cabin_event(track_id, current_time)
            
            self.prev_cabin_positions[track_id] = curr_bbox
        return cabin_crossed
    
    def _handle_cabin_event(self, cabin_track_id: int, current_time: float):
        """
        Cierra el conteo de la cabina anterior e inicia uno nuevo.
        """
        # 1. CERRAR evento anterior
        if self.current_event:
            if self.csv_path:
                self._save_to_csv(self.csv_path, self.current_event)
            logger.info(f"Cabina #{self.current_event.cabin_id} completada. Total: {self.current_event.persons_embarked}")
        
        # 2. ABRIR nuevo evento
        self.cabin_count += 1
        self.current_event = CabinEvent(
            cabin_id=self.cabin_count,
            track_id=cabin_track_id,
            cross_time=current_time,
            persons_before=set(), # Solo referencial
            persons_embarked=0,   # Reset contador
            embarked_ids=[]
        )
        
    def _draw_annotations(self, frame: np.ndarray, heads: List, cabins: List) -> np.ndarray:
        h, w = frame.shape[:2]
        
        # --- CAPA TRANSPARENTE PARA ZONAS ---
        overlay = frame.copy()
        
        # Rellenar zonas con color en la capa overlay
        cv2.fillPoly(overlay, [np.array(ZONA_EMBARQUE)], (0, 255, 0))    # Verde
        cv2.fillPoly(overlay, [np.array(ZONA_PAX1)], (230, 150, 50))     # Azul
        cv2.fillPoly(overlay, [np.array(ZONA_PAX2)], (150, 50, 230))     # Magenta
        
        # Mezclar frame con overlay (Opacidad)
        cv2.addWeighted(overlay, OPACIDAD_ZONAS, frame, 1 - OPACIDAD_ZONAS, 0, frame)
        
        # --- LÍNEAS SÓLIDAS SOBRE LAS ZONAS ---
        cv2.polylines(frame, [np.array(ZONA_EMBARQUE)], True, (0, 200, 0), 2)
        cv2.line(frame, LINEA_TELEFERICO[0], LINEA_TELEFERICO[1], (0, 0, 255), 3)
        
        # Transversales
        for p1_idx, p2_idx in TRANSVERSALES1:
             cv2.line(frame, ZONA_PAX1[p1_idx], ZONA_PAX1[p2_idx], (255, 200, 0), 1)
        for p1_idx, p2_idx in TRANSVERSALES2:
             cv2.line(frame, ZONA_PAX2[p1_idx], ZONA_PAX2[p2_idx], (0, 255, 255), 1)

        # Dibujar Cabinas
        for cabin in cabins:
            bbox = cabin['bbox']
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2)
            cv2.putText(frame, f"CABINA", (bbox[0], bbox[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # --- DETECCIONES Y TEXTO ---
        # [CORREGIDO] Lógica de dibujo restaurada
        for head in heads:
            bbox = head['bbox']
            raw_tid = head['track_id']
            
            # Resolver ID real usando el mapa
            tid = raw_tid
            if raw_tid in self.track_id_map:
                tid = self.track_id_map[raw_tid]

            if tid and tid in self.persons:
                person = self.persons[tid]
                tiempo = person.time_tracked
                time_str = f"{tiempo:.1f}s"

                if person.is_operator:
                    color = (128, 128, 128)
                    label = f"OP:{tid} | {time_str}"
                elif person.is_passenger:
                    color = (0, 255, 0)
                    label = f"PAX:{tid} | {time_str}"
                else:
                    color = (255, 165, 0)
                    label = f"ID:{tid} | {time_str}"
                
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                cv2.putText(frame, label, (bbox[0], bbox[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Panel Info
        cv2.rectangle(frame, (10, 10), (320, 120), (30, 30, 30), -1)
        cv2.putText(frame, f"CABINA ACTUAL: #{self.cabin_count}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        embarcados_actual = self.current_event.persons_embarked if self.current_event else 0
        cv2.putText(frame, f"Embarcando ahora: {embarcados_actual}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Total acumulado: {self.total_embarked_count}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def _save_to_csv(self, csv_path: str, event: CabinEvent):
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            hora = datetime.fromtimestamp(event.cross_time).strftime("%H:%M:%S")
            ids_str = ','.join(map(str, event.embarked_ids)) if event.embarked_ids else "ninguno"
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                event.cabin_id,
                event.persons_embarked,
                hora,
                ids_str,
                0 
            ])
    
    def _save_minute_log(self, log_path: str, elapsed_time: float, frame_count: int):
        with open(log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            minuto = int(elapsed_time / 60)
            personas_totales = len(self.persons)
            operadores = len([p for p in self.persons.values() if p.is_operator])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                minuto, self.cabin_count, personas_totales,
                operadores, self.total_embarked_count, frame_count
            ])
    
    def _cleanup_old_tracks(self, current_time: float):
        to_remove = []
        for tid, person in self.persons.items():
            if current_time - person.last_seen > TIEMPO_REFRESCO_TRACKS:
                if not person.in_zone:
                    to_remove.append(tid)
        for tid in to_remove:
            del self.persons[tid]
            self.already_counted.discard(tid)
        
        lost_to_remove = []
        for tid, person in self.lost_persons.items():
            if current_time - person.last_seen > TIEMPO_REFRESCO_TRACKS:
                lost_to_remove.append(tid)
        for tid in lost_to_remove:
            del self.lost_persons[tid]
    
    def _connect_to_camera(self, max_retries=10, retry_delay=3.0):
        for attempt in range(1, max_retries + 1):
            logger.info(f"Intento de conexión {attempt}/{max_retries}...")
            cap = cv2.VideoCapture(self.rtsp_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Minimizar latencia
            if cap.isOpened():
                ret, _ = cap.read()
                if ret: return cap
            cap.release()
            time.sleep(retry_delay)
        raise RuntimeError("Fallo conexión cámara")
    
    def run(self, duration_minutes: float):
        print("\nSISTEMA DE CONTEO DE PASAJEROS v2.2 (FINAL)\n" + "="*50)
        
        cap = self._connect_to_camera()
        ret, frame = cap.read()
        if not ret: raise RuntimeError("No se pudo leer frame inicial")
        h, w = frame.shape[:2]
        
        video_path = os.path.join(self.output_folder, f"video_{self.timestamp_str}.mp4")
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps_video, (w, h))
        
        # Inicializar CSVs
        self.csv_path = os.path.join(self.output_folder, f"conteo_{self.timestamp_str}.csv")
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(['timestamp', 'cabina_numero', 'personas_embarcadas', 'hora_paso', 'ids_personas', 'ignore'])
        
        log_path = os.path.join(self.output_folder, f"log_minuto_{self.timestamp_str}.csv")
        with open(log_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(['timestamp', 'minuto', 'cabinas', 'personas_total', 'operadores', 'embarcados_total', 'frames'])
        
        start_time = time.time()
        last_log = start_time
        last_cleanup = start_time
        frame_count = 0
        consecutive_failures = 0
        
        try:
            while True:
                current_time = time.time()
                elapsed = current_time - start_time
                if elapsed >= duration_minutes * 60: break
                
                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                    logger.warning(f"Frame perdido ({consecutive_failures}), reintentando...")
                    if consecutive_failures > 30:
                        cap.release()
                        cap = self._connect_to_camera()
                        consecutive_failures = 0
                    continue
                
                consecutive_failures = 0
                frame_count += 1
                heads, cabins = self._process_frame(frame, current_time)
                self._update_persons(heads, current_time)
                self._check_cabin_crossing(cabins, current_time)
                
                annotated = self._draw_annotations(frame, heads, cabins)
                video_writer.write(annotated)
                
                if current_time - last_cleanup > 30:
                    self._cleanup_old_tracks(current_time)
                    last_cleanup = current_time
                
                if current_time - last_log >= 60:
                    self._save_minute_log(log_path, elapsed, frame_count)
                    last_log = current_time
                    logger.info(f"Minuto {int(elapsed/60)} - Embarcados total: {self.total_embarked_count}")
                    
        except KeyboardInterrupt:
            logger.info("Interrumpido por usuario")
        finally:
            if self.current_event and self.csv_path:
                self._save_to_csv(self.csv_path, self.current_event)
            
            cap.release()
            video_writer.release()
            print(f"Finalizado. Total embarcados: {self.total_embarked_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('minutos', type=float)
    parser.add_argument('--modelo', default=MODEL_PATH)
    parser.add_argument('--rtsp', default=RTSP_URL)
    parser.add_argument('--output', default=OUTPUT_FOLDER)
    parser.add_argument('--conf', type=float, default=CONF_THRESHOLD)
    args = parser.parse_args()
    
    TelefericoCounterV2(args.modelo, args.rtsp, args.output, args.conf, FPS_VIDEO).run(args.minutos)
