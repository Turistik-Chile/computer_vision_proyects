#!/usr/bin/env python3
"""
Sistema Avanzado de Conteo de Pasajeros en Teleférico v2
=========================================================
Implementa lógica de conteo basada en:
1. Detección de cabinas cruzando línea trigger
2. Tracking direccional de personas hacia la cabina
3. Filtrado inteligente de operadores por comportamiento
4. Detección de personas que "desaparecen" hacia la cabina

Uso: python teleferico_counter_v2.py <minutos>

Mejoras sobre v1:
- Tracking direccional (detecta dirección de movimiento)
- Zona de "salida" hacia cabina más precisa
- Mejor filtrado de operadores
- Soporte para cuando 0 personas embarcan
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
# CONFIGURACIÓN DE BYTETRACK (embebida para no necesitar archivo externo)
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
# CONFIGURACIÓN
# ============================================================================

RTSP_URL = "rtsp://usertf:Tfo.-2525@172.16.10.22:554/cam/realmonitor?channel=5&subtype=0"
MODEL_PATH = "/home/aaravenatk/yolo/best.pt"
CONF_THRESHOLD = 0.7
OUTPUT_FOLDER = "detecciones"
FPS_VIDEO = 15.0

# Geometría de la escena
# FLUJO: Cabinas van de IZQUIERDA a DERECHA
# - Cabina llega por la izquierda
# - Cruza la línea trigger
# - Personas embarcan moviéndose hacia la IZQUIERDA (hacia la cabina)
# - Cabina sale por la derecha
LINEA_TELEFERICO = [(561, 110), (570, 465)]

ZONA_EMBARQUE = [
    (570, 168), (809, 164), (897, 159),
    (984, 185), (997, 388), (575, 359)
]

# Línea de embarque: personas que cruzan esta línea hacia la IZQUIERDA embarcan
LINEA_EMBARQUE = [(572, 170), (574, 355)]

# Parámetros de comportamiento
TIEMPO_OPERADOR = 12.0          # Segundos en escena para considerar operador
MOVIMIENTO_MIN_OPERADOR = 25    # Píxeles/seg mínimo para no ser operador (si está quieto)
BUFFER_CONFIRMACION = 2.0       # Segundos para confirmar embarque
TIEMPO_REFRESCO_TRACKS = 60.0   # Segundos para limpiar tracks antiguos

# Parámetros adicionales para operadores que se mueven
CICLOS_ENTRADA_SALIDA = 2       # Si entra/sale de zona X veces, es operador
TIEMPO_ESCENA_LARGO = 45.0      # Si está en escena >45s total, es operador

# Parámetros de re-identificación (para reducir falsos positivos)
TIEMPO_REIDENTIFICACION = 2.0   # Segundos para mantener personas "perdidas" en buffer
DISTANCIA_MAX_REIDENT = 100     # Distancia máxima en píxeles para re-identificar
SIMILARIDAD_BBOX_MIN = 0.7      # Similitud mínima de tamaño de bbox (0-1)

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
    has_embarked: bool = False
    crossed_embarque_line: bool = False
    
    # Para análisis de dirección
    avg_velocity: Tuple[float, float] = (0.0, 0.0)
    
    # Nuevos: para detectar operadores que entran/salen de zona
    zone_entry_count: int = 0       # Veces que entró a la zona
    zone_exit_count: int = 0        # Veces que salió de la zona
    was_in_zone: bool = False       # Estado anterior (para detectar transiciones)
    
    # Para re-identificación
    last_bbox_size: Tuple[int, int] = (0, 0)  # (ancho, alto) del último bbox
    is_lost: bool = False           # Si está temporalmente perdido
    lost_timestamp: float = 0.0     # Cuándo se perdió
    
    def add_position(self, x: float, y: float, timestamp: float, bbox: Tuple[int, int, int, int] = None):
        """Añade una posición al historial."""
        self.history.append((x, y, timestamp))
        self.last_seen = timestamp
        if bbox:
            self.last_bbox_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
        self._update_velocity()
    
    def predict_position(self, delta_time: float) -> Tuple[float, float]:
        """Predice posición futura basándose en velocidad actual."""
        if not self.last_position:
            return (0, 0)
        
        pred_x = self.last_position[0] + self.avg_velocity[0] * delta_time
        pred_y = self.last_position[1] + self.avg_velocity[1] * delta_time
        return (pred_x, pred_y)
    
    def bbox_similarity(self, other_bbox_size: Tuple[int, int]) -> float:
        """Calcula similitud entre tamaños de bbox (0-1)."""
        if self.last_bbox_size == (0, 0) or other_bbox_size == (0, 0):
            return 0.0
        
        w1, h1 = self.last_bbox_size
        w2, h2 = other_bbox_size
        
        w_ratio = min(w1, w2) / max(w1, w2) if max(w1, w2) > 0 else 0
        h_ratio = min(h1, h2) / max(h1, h2) if max(h1, h2) > 0 else 0
        
        return (w_ratio + h_ratio) / 2
    
    def _update_velocity(self):
        """Calcula velocidad promedio."""
        if len(self.history) < 5:
            return
        
        recent = list(self.history)[-10:]
        if len(recent) < 2:
            return
        
        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        dt = recent[-1][2] - recent[0][2]
        
        if dt > 0:
            self.avg_velocity = (dx/dt, dy/dt)
    
    def get_movement_stats(self) -> Tuple[float, float]:
        """Retorna (distancia total, velocidad promedio)."""
        if len(self.history) < 2:
            return 0.0, 0.0
        
        total_dist = 0.0
        positions = list(self.history)
        for i in range(1, len(positions)):
            x1, y1, _ = positions[i-1]
            x2, y2, _ = positions[i]
            total_dist += np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        duration = self.last_seen - self.first_seen
        avg_speed = total_dist / duration if duration > 0 else 0
        
        return total_dist, avg_speed
    
    def is_moving_left(self) -> bool:
        """Verifica si la persona se mueve hacia la izquierda (hacia cabina)."""
        return self.avg_velocity[0] < -5  # Velocidad negativa en X
    
    def update_zone_status(self, currently_in_zone: bool):
        """Actualiza contadores de entrada/salida de zona."""
        if currently_in_zone and not self.was_in_zone:
            # Acaba de ENTRAR a la zona
            self.zone_entry_count += 1
        elif not currently_in_zone and self.was_in_zone:
            # Acaba de SALIR de la zona
            self.zone_exit_count += 1
        
        self.was_in_zone = self.in_zone
        self.in_zone = currently_in_zone
    
    def has_cyclic_pattern(self) -> bool:
        """Detecta si tiene patrón de entrada/salida repetida (operador)."""
        return self.zone_entry_count >= CICLOS_ENTRADA_SALIDA
    
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
    persons_before: Set[int] = field(default_factory=set)  # IDs en zona antes
    persons_embarked: int = 0
    embarked_ids: List[int] = field(default_factory=list)


# ============================================================================
# UTILIDADES GEOMÉTRICAS
# ============================================================================

def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[int, int]]) -> bool:
    """Verifica si un punto está dentro de un polígono."""
    return cv2.pointPolygonTest(np.array(polygon, np.int32), point, False) >= 0


def line_side(point: Tuple[float, float], line: List[Tuple[int, int]]) -> float:
    """
    Determina de qué lado de una línea está un punto.
    Retorna: >0 si está a la izquierda, <0 si está a la derecha, 0 si está en la línea.
    """
    x, y = point
    (x1, y1), (x2, y2) = line
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


def crossed_line(prev_pos: Tuple[float, float], curr_pos: Tuple[float, float],
                 line: List[Tuple[int, int]]) -> Tuple[bool, str]:
    """
    Verifica si hubo cruce de línea entre dos posiciones.
    Retorna: (cruzó, dirección) donde dirección es 'left' o 'right'
    """
    prev_side = line_side(prev_pos, line)
    curr_side = line_side(curr_pos, line)
    
    if prev_side * curr_side < 0:  # Cambio de signo = cruce
        direction = 'left' if curr_side > 0 else 'right'
        return True, direction
    
    return False, ''


def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """Centro de un bounding box."""
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def bbox_bottom(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """Centro inferior de un bounding box."""
    return ((bbox[0] + bbox[2]) / 2, bbox[3])


# ============================================================================
# CLASE PRINCIPAL
# ============================================================================

class TelefericoCounterV2:
    """Sistema de conteo de pasajeros versión 2."""
    
    def __init__(self, model_path: str, rtsp_url: str, output_folder: str,
                 conf_threshold: float = 0.7, fps_video: float = 15.0):
        
        self.model_path = model_path
        self.rtsp_url = rtsp_url
        self.output_folder = output_folder
        self.conf_threshold = conf_threshold
        self.fps_video = fps_video
        
        # Cargar modelo
        logger.info(f"Cargando modelo: {model_path}")
        self.model = YOLO(model_path)
        
        # Estado de tracking
        self.persons: Dict[int, PersonTracker] = {}
        self.prev_cabin_positions: Dict[int, Tuple[int, int, int, int]] = {}
        
        # Conteo de cabinas
        self.cabin_count = 0
        self.cabin_events: List[CabinEvent] = []
        self.current_event: Optional[CabinEvent] = None
        
        # IDs de personas actualmente en zona de embarque
        self.persons_in_zone: Set[int] = set()
        
        # IDs que ya fueron contados (para no contar doble)
        self.already_counted: Set[int] = set()
        
        # Buffer de re-identificación para personas perdidas temporalmente
        self.lost_persons: Dict[int, PersonTracker] = {}
        
        # Crear archivo de configuración de ByteTrack temporal
        self.tracker_config_path = self._create_tracker_config()
        
        # Crear carpeta de salida
        os.makedirs(output_folder, exist_ok=True)
        self.timestamp_str = datetime.now().strftime("%d-%m-%y_%H-%M-%S")
    
    def _create_tracker_config(self) -> str:
        """Crea archivo temporal de configuración de ByteTrack."""
        config_path = os.path.join(tempfile.gettempdir(), "bytetrack_teleferico.yaml")
        with open(config_path, 'w') as f:
            f.write(BYTETRACK_CONFIG)
        return config_path
    
    def _reidentify_person(self, bbox: Tuple[int, int, int, int], pos: Tuple[float, float], 
                          current_time: float) -> Optional[int]:
        """Intenta re-identificar una detección nueva con personas perdidas."""
        best_match_id = None
        best_score = 0.0
        
        bbox_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
        
        for lost_id, lost_person in list(self.lost_persons.items()):
            time_lost = current_time - lost_person.lost_timestamp
            
            # Verificar si aún está dentro del tiempo de re-identificación
            if time_lost > TIEMPO_REIDENTIFICACION:
                continue
            
            # Predecir dónde debería estar
            predicted_pos = lost_person.predict_position(time_lost)
            
            # Calcular distancia a la posición predicha
            distance = np.sqrt((pos[0] - predicted_pos[0])**2 + (pos[1] - predicted_pos[1])**2)
            
            # Calcular similitud de tamaño de bbox
            size_similarity = lost_person.bbox_similarity(bbox_size)
            
            # Score combinado (normalizado)
            distance_score = max(0, 1 - (distance / DISTANCIA_MAX_REIDENT))
            combined_score = (distance_score * 0.6) + (size_similarity * 0.4)
            
            # Verificar si cumple criterios mínimos
            if distance < DISTANCIA_MAX_REIDENT and size_similarity > SIMILARIDAD_BBOX_MIN:
                if combined_score > best_score:
                    best_score = combined_score
                    best_match_id = lost_id
        
        if best_match_id:
            logger.info(f"Re-identificación exitosa: ID {best_match_id} (score: {best_score:.2f})")
        
        return best_match_id
    
    def _detect_operator(self, person: PersonTracker, current_time: float) -> bool:
        """
        Determina si una persona es operador basándose en comportamiento.
        
        Criterio:
        1. Mucho tiempo en escena (>45s) = definitivamente operador
        """
        # Criterio 1: Tiempo muy largo en escena = operador seguro
        if person.time_tracked > TIEMPO_ESCENA_LARGO:
            logger.debug(f"Operador {person.track_id}: tiempo largo en escena ({person.time_tracked:.1f}s)")
            return True
        
        return False
    
    def _process_frame(self, frame: np.ndarray, current_time: float) -> Tuple[List, List]:
        """Procesa un frame y retorna detecciones de cabezas y cabinas."""
        
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_config_path,
            conf=self.conf_threshold,
            verbose=False
        )
        
        heads = []
        cabins = []
        
        for result in results:
            if result.boxes is None:
                continue
            
            for box in result.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                bbox = tuple(map(int, box.xyxy[0].tolist()))
                
                track_id = None
                if hasattr(box, 'id') and box.id is not None:
                    track_id = int(box.id[0])
                
                detection = {'bbox': bbox, 'conf': conf, 'track_id': track_id}
                
                if cls == CLASE_CABEZA:
                    heads.append(detection)
                elif cls == CLASE_CABINA:
                    cabins.append(detection)
        
        return heads, cabins
    
    def _update_persons(self, heads: List, current_time: float) -> List[int]:
        """
        Actualiza tracking de personas.
        Retorna lista de IDs que embarcaron en este frame.
        """
        embarked_this_frame = []
        current_ids = set()
        
        for head in heads:
            track_id = head['track_id']
            bbox = head['bbox']
            pos = bbox_bottom(bbox)
            
            # Si no tiene track_id, intentar re-identificar
            if track_id is None:
                reidentified_id = self._reidentify_person(bbox, pos, current_time)
                if reidentified_id is not None:
                    # Recuperar persona del buffer de perdidos
                    track_id = reidentified_id
                    person = self.lost_persons.pop(reidentified_id)
                    person.is_lost = False
                    self.persons[track_id] = person
                    logger.info(f"Re-identificado: Persona {track_id}")
                else:
                    # No se pudo re-identificar, saltar esta detección
                    continue
            
            current_ids.add(track_id)
            in_zone = point_in_polygon(pos, ZONA_EMBARQUE)
            
            # Crear o actualizar tracker
            if track_id not in self.persons:
                self.persons[track_id] = PersonTracker(
                    track_id=track_id,
                    first_seen=current_time,
                    last_seen=current_time,
                    in_zone=in_zone
                )
            
            person = self.persons[track_id]
            prev_pos = person.last_position
            person.add_position(pos[0], pos[1], current_time, bbox)
            
            # Actualizar estado de zona CON tracking de entradas/salidas
            person.update_zone_status(in_zone)
            
            # Verificar si es operador (ANTES de actualizar personas en zona)
            if not person.is_operator:
                person.is_operator = self._detect_operator(person, current_time)
            
            # Actualizar conjunto de personas en zona (solo no-operadores)
            if in_zone and not person.is_operator:
                self.persons_in_zone.add(track_id)
            else:
                self.persons_in_zone.discard(track_id)
            
            # Detectar cruce de línea de embarque
            if prev_pos and not person.has_embarked and not person.is_operator:
                crossed, direction = crossed_line(prev_pos, pos, LINEA_EMBARQUE)
                if crossed and direction == 'left':
                    person.crossed_embarque_line = True
                    logger.info(f"Persona {track_id} cruzó línea de embarque")
        
        # Gestionar personas que desaparecieron (mover a buffer temporal)
        all_known_ids = set(self.persons.keys())
        disappeared_ids = all_known_ids - current_ids
        
        for track_id in disappeared_ids:
            if track_id in self.persons:
                person = self.persons[track_id]
                
                # Si no está marcado como perdido, marcarlo ahora
                if not person.is_lost:
                    person.is_lost = True
                    person.lost_timestamp = current_time
                    # Mover a buffer de perdidos
                    self.lost_persons[track_id] = person
                    logger.debug(f"Persona {track_id} perdida temporalmente")
        
        # Limpiar buffer de perdidos (personas que llevan >2s perdidas)
        expired_lost = []
        for lost_id, lost_person in self.lost_persons.items():
            time_lost = current_time - lost_person.lost_timestamp
            if time_lost > TIEMPO_REIDENTIFICACION:
                # Verificar si embarcó antes de expirar
                if lost_person.in_zone and not lost_person.is_operator and not lost_person.has_embarked:
                    if lost_person.is_moving_left() or lost_person.crossed_embarque_line:
                        if lost_id not in self.already_counted:
                            lost_person.has_embarked = True
                            embarked_this_frame.append(lost_id)
                            self.already_counted.add(lost_id)
                            logger.info(f"Persona {lost_id} embarcó (desapareció de zona)")
                
                expired_lost.append(lost_id)
                self.persons_in_zone.discard(lost_id)
        
        # Remover personas expiradas del buffer
        for lost_id in expired_lost:
            del self.lost_persons[lost_id]
        
        return embarked_this_frame
    
    def _check_cabin_crossing(self, cabins: List, current_time: float) -> bool:
        """Verifica si alguna cabina cruzó la línea trigger."""
        cabin_crossed = False
        
        for cabin in cabins:
            track_id = cabin['track_id']
            if track_id is None:
                continue
            
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
        """Maneja el evento de una cabina cruzando la línea."""
        
        # Finalizar evento anterior
        if self.current_event:
            # Contar embarcados del ciclo anterior
            persons_now = self.persons_in_zone.copy()
            persons_before = self.current_event.persons_before
            
            # Personas que estaban antes y ya no están = embarcaron
            embarked_ids = []
            for pid in persons_before:
                if pid in self.persons and self.persons[pid].has_embarked:
                    if pid not in self.current_event.embarked_ids:
                        embarked_ids.append(pid)
            
            self.current_event.embarked_ids = embarked_ids
            self.current_event.persons_embarked = len(embarked_ids)
            self.cabin_events.append(self.current_event)
            
            logger.info(f"Cabina #{self.current_event.cabin_id} completada: "
                       f"{self.current_event.persons_embarked} personas")
        
        # Nuevo evento
        self.cabin_count += 1
        self.current_event = CabinEvent(
            cabin_id=self.cabin_count,
            track_id=cabin_track_id,
            cross_time=current_time,
            persons_before=self.persons_in_zone.copy()
        )
        
        # Resetear flags de embarque para nuevo ciclo
        for person in self.persons.values():
            if person.has_embarked:
                person.has_embarked = False
                person.crossed_embarque_line = False
        
        logger.info(f"Nueva cabina #{self.cabin_count} - "
                   f"{len(self.persons_in_zone)} personas en zona")
    
    def _draw_annotations(self, frame: np.ndarray, heads: List, cabins: List) -> np.ndarray:
        """Dibuja anotaciones en el frame."""
        h, w = frame.shape[:2]
        
        # Zona de embarque (verde semi-transparente)
        overlay = frame.copy()
        pts = np.array(ZONA_EMBARQUE, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (0, 200, 0))
        frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        
        # Línea del teleférico (roja)
        cv2.line(frame, LINEA_TELEFERICO[0], LINEA_TELEFERICO[1], (0, 0, 255), 3)
        cv2.putText(frame, "LINEA TRIGGER", (LINEA_TELEFERICO[0][0]-80, LINEA_TELEFERICO[0][1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Línea de embarque (azul)
        cv2.line(frame, LINEA_EMBARQUE[0], LINEA_EMBARQUE[1], (255, 0, 0), 2)
        
        # Dibujar cabezas
        for head in heads:
            bbox = head['bbox']
            tid = head['track_id']
            
            if tid and tid in self.persons:
                person = self.persons[tid]
                tiempo = person.time_tracked
                
                if person.is_operator:
                    color = (128, 128, 128)  # Gris = operador
                    label = f"OP:{tid} ({tiempo:.1f}s)"
                elif person.has_embarked:
                    color = (0, 255, 0)  # Verde = embarcó
                    label = f"EMB:{tid} ({tiempo:.1f}s)"
                elif person.in_zone:
                    color = (0, 255, 255)  # Amarillo = en zona
                    label = f"ZONA:{tid} ({tiempo:.1f}s)"
                else:
                    color = (255, 165, 0)  # Naranja = fuera de zona
                    label = f"ID:{tid} ({tiempo:.1f}s)"
            else:
                color = (200, 200, 200)
                label = f"?:{tid} (0.0s)"
            
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, label, (bbox[0], bbox[1]-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Dibujar cabinas
        for cabin in cabins:
            bbox = cabin['bbox']
            tid = cabin['track_id']
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 255), 3)
            cv2.putText(frame, f"CABINA:{tid}", (bbox[0], bbox[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Panel de información
        panel_h = 170
        cv2.rectangle(frame, (10, 10), (320, panel_h), (30, 30, 30), -1)
        cv2.rectangle(frame, (10, 10), (320, panel_h), (200, 200, 200), 2)
        
        y = 35
        cv2.putText(frame, f"CABINA ACTUAL: #{self.cabin_count}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y += 28
        personas_zona = len([p for p in self.persons.values() 
                           if p.in_zone and not p.is_operator])
        cv2.putText(frame, f"Personas en zona: {personas_zona}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        
        y += 25
        operadores = len([p for p in self.persons.values() if p.is_operator])
        cv2.putText(frame, f"Operadores: {operadores}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (128, 128, 128), 1)
        
        y += 25
        embarcados = len([p for p in self.persons.values() if p.has_embarked])
        cv2.putText(frame, f"Embarcados ciclo: {embarcados}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        
        y += 25
        cv2.putText(frame, f"Cabinas procesadas: {self.cabin_count}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 165, 0), 1)
        
        # Timestamp
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, ts, (w - 220, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def _save_to_csv(self, csv_path: str, event: CabinEvent):
        """Guarda un evento de cabina al CSV."""
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
                len(event.persons_before)
            ])
    
    def _save_minute_log(self, log_path: str, elapsed_time: float, frame_count: int):
        """Guarda log de inferencias cada minuto."""
        with open(log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            minuto = int(elapsed_time / 60)
            personas_totales = len(self.persons)
            personas_zona = len([p for p in self.persons.values() if p.in_zone and not p.is_operator])
            operadores = len([p for p in self.persons.values() if p.is_operator])
            embarcados_total = len([p for p in self.persons.values() if p.has_embarked or p.track_id in self.already_counted])
            personas_activas = len([p for p in self.persons.values() if elapsed_time - p.last_seen < 5.0])
            
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                minuto,
                self.cabin_count,
                personas_totales,
                personas_activas,
                personas_zona,
                operadores,
                embarcados_total,
                frame_count
            ])
    
    def _cleanup_old_tracks(self, current_time: float):
        """Limpia tracks antiguos para liberar memoria."""
        to_remove = []
        for tid, person in self.persons.items():
            if current_time - person.last_seen > TIEMPO_REFRESCO_TRACKS:
                if not person.in_zone:
                    to_remove.append(tid)
        
        for tid in to_remove:
            del self.persons[tid]
            self.already_counted.discard(tid)
        
        # También limpiar buffer de perdidos antiguos
        lost_to_remove = []
        for tid, person in self.lost_persons.items():
            if current_time - person.last_seen > TIEMPO_REFRESCO_TRACKS:
                lost_to_remove.append(tid)
        
        for tid in lost_to_remove:
            del self.lost_persons[tid]
        
        # También limpiar buffer de perdidos antiguos
        lost_to_remove = []
        for tid, person in self.lost_persons.items():
            if current_time - person.last_seen > TIEMPO_REFRESCO_TRACKS:
                lost_to_remove.append(tid)
        
        for tid in lost_to_remove:
            del self.lost_persons[tid]
    
    def _connect_to_camera(self, max_retries: int = 10, retry_delay: float = 3.0) -> cv2.VideoCapture:
        """
        Conecta a la cámara RTSP con reintentos.
        
        Args:
            max_retries: Número máximo de intentos de conexión
            retry_delay: Segundos de espera entre reintentos
            
        Returns:
            VideoCapture conectado y funcionando
            
        Raises:
            RuntimeError si no se puede conectar después de todos los reintentos
        """
        for attempt in range(1, max_retries + 1):
            logger.info(f"Intento de conexión {attempt}/{max_retries}...")
            
            try:
                cap = cv2.VideoCapture(self.rtsp_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                if not cap.isOpened():
                    logger.warning(f"No se pudo abrir stream (intento {attempt})")
                    cap.release()
                    if attempt < max_retries:
                        logger.info(f"Reintentando en {retry_delay} segundos...")
                        time.sleep(retry_delay)
                    continue
                
                # Intentar leer un frame para verificar conexión real
                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.warning(f"Stream abierto pero no se pudo leer frame (intento {attempt})")
                    cap.release()
                    if attempt < max_retries:
                        logger.info(f"Reintentando en {retry_delay} segundos...")
                        time.sleep(retry_delay)
                    continue
                
                # Conexión exitosa
                logger.info(f"✓ Conexión exitosa en intento {attempt}")
                return cap
                
            except Exception as e:
                logger.error(f"Error en conexión (intento {attempt}): {e}")
                if attempt < max_retries:
                    logger.info(f"Reintentando en {retry_delay} segundos...")
                    time.sleep(retry_delay)
        
        raise RuntimeError(f"No se pudo conectar al stream RTSP después de {max_retries} intentos: {self.rtsp_url}")
    
    def run(self, duration_minutes: float):
        """Ejecuta el sistema de conteo."""
        
        print("\n" + "="*60)
        print("SISTEMA DE CONTEO DE PASAJEROS - TELEFÉRICO v2")
        print("="*60)
        print(f"Duración: {duration_minutes} minutos")
        print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60 + "\n")
        
        # Conectar a RTSP con reintentos
        logger.info(f"Conectando a: {self.rtsp_url}")
        cap = self._connect_to_camera(max_retries=10, retry_delay=3.0)
        
        # Leer frame para obtener dimensiones
        ret, frame = cap.read()
        if not ret:
            # Si falla aquí, intentar reconectar una vez más
            logger.warning("Fallo al leer frame inicial, reconectando...")
            cap.release()
            cap = self._connect_to_camera(max_retries=5, retry_delay=2.0)
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("No se pudo leer frame inicial después de reconexión")
        
        h, w = frame.shape[:2]
        logger.info(f"Resolución: {w}x{h}")
        
        # Inicializar video writer
        video_path = os.path.join(self.output_folder, f"video_{self.timestamp_str}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, self.fps_video, (w, h))
        logger.info(f"Video: {video_path}")
        
        # Inicializar CSV de conteo
        csv_path = os.path.join(self.output_folder, f"conteo_{self.timestamp_str}.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                'timestamp', 'cabina_numero', 'personas_embarcadas',
                'hora_paso', 'ids_personas', 'personas_en_zona_al_inicio'
            ])
        logger.info(f"CSV conteo: {csv_path}")
        
        # Inicializar CSV de log minuto a minuto
        log_path = os.path.join(self.output_folder, f"log_minuto_{self.timestamp_str}.csv")
        with open(log_path, 'w', newline='', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                'timestamp', 'minuto', 'cabinas_procesadas', 'personas_detectadas_total',
                'personas_activas', 'personas_en_zona', 'operadores_detectados',
                'personas_embarcadas_total', 'frames_procesados'
            ])
        logger.info(f"CSV log: {log_path}")
        
        # Control de tiempo
        start_time = time.time()
        duration_sec = duration_minutes * 60
        frame_count = 0
        last_cleanup = start_time
        last_log = start_time
        consecutive_failures = 0
        max_consecutive_failures = 30  # Si falla 30 veces seguidas, reconectar completamente
        
        try:
            while True:
                current_time = time.time()
                elapsed = current_time - start_time
                
                if elapsed >= duration_sec:
                    logger.info("Tiempo completado")
                    break
                
                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                    logger.warning(f"Frame perdido ({consecutive_failures} consecutivos)")
                    
                    if consecutive_failures >= max_consecutive_failures:
                        # Reconexión completa con reintentos
                        logger.warning("Demasiados fallos, reconectando completamente...")
                        cap.release()
                        try:
                            cap = self._connect_to_camera(max_retries=5, retry_delay=2.0)
                            consecutive_failures = 0
                            logger.info("Reconexión exitosa")
                        except RuntimeError as e:
                            logger.error(f"Fallo en reconexión: {e}")
                            logger.info("Esperando 10 segundos antes de reintentar...")
                            time.sleep(10)
                            try:
                                cap = self._connect_to_camera(max_retries=10, retry_delay=3.0)
                                consecutive_failures = 0
                            except RuntimeError:
                                logger.error("No se pudo reconectar, terminando...")
                                break
                    else:
                        time.sleep(0.5)  # Pequeña espera antes de reintentar
                    continue
                
                # Frame leído correctamente
                consecutive_failures = 0
                frame_count += 1
                
                # Procesar frame
                heads, cabins = self._process_frame(frame, current_time)
                
                # Actualizar tracking
                self._update_persons(heads, current_time)
                self._check_cabin_crossing(cabins, current_time)
                
                # Dibujar
                annotated = self._draw_annotations(frame, heads, cabins)
                
                # Barra de progreso
                progress = elapsed / duration_sec
                bar_w = 200
                cv2.rectangle(annotated, (w-220, h-30), (w-20, h-10), (50,50,50), -1)
                cv2.rectangle(annotated, (w-220, h-30), 
                             (w-220+int(bar_w*progress), h-10), (0,255,0), -1)
                cv2.putText(annotated, f"{int(duration_sec-elapsed)}s", 
                           (w-220, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                
                video_writer.write(annotated)
                
                # Guardar eventos completados
                while self.cabin_events:
                    event = self.cabin_events.pop(0)
                    self._save_to_csv(csv_path, event)
                
                # Limpieza periódica
                if current_time - last_cleanup > 30:
                    self._cleanup_old_tracks(current_time)
                    last_cleanup = current_time
                
                # Log minuto a minuto
                if current_time - last_log >= 60:
                    self._save_minute_log(log_path, elapsed, frame_count)
                    last_log = current_time
                    logger.info(f"Log minuto {int(elapsed/60)} guardado")
                
                # Log de progreso
                if frame_count % 150 == 0:
                    logger.info(f"Frames: {frame_count} | Tiempo: {elapsed:.0f}s | "
                               f"Cabinas: {self.cabin_count}")
        
        except KeyboardInterrupt:
            logger.info("Interrumpido por usuario")
        
        finally:
            # Guardar último log del minuto actual (si no se guardó aún)
            elapsed_final = time.time() - start_time
            if elapsed_final - (last_log - start_time) >= 1.0:  # Si pasó al menos 1 segundo desde último log
                self._save_minute_log(log_path, elapsed_final, frame_count)
                logger.info(f"Log final guardado (minuto {int(elapsed_final/60)})")
            
            # Finalizar último evento
            if self.current_event:
                embarcados = [p.track_id for p in self.persons.values() 
                             if p.has_embarked]
                self.current_event.embarked_ids = embarcados
                self.current_event.persons_embarked = len(embarcados)
                self._save_to_csv(csv_path, self.current_event)
            
            # Agregar resumen al final del CSV
            total_embarked = sum(1 for p in self.persons.values() if p.has_embarked or p.track_id in self.already_counted)
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([])  # Línea en blanco
                writer.writerow(['RESUMEN'])
                writer.writerow(['Total cabinas procesadas', self.cabin_count])
                writer.writerow(['Total personas contadas', total_embarked])
            
            cap.release()
            video_writer.release()
            
            # Resumen
            print("\n" + "="*60)
            print("RESUMEN")
            print("="*60)
            print(f"Cabinas procesadas: {self.cabin_count}")
            print(f"Personas detectadas: {len(self.persons)}")
            print(f"Operadores identificados: {len([p for p in self.persons.values() if p.is_operator])}")
            total_embarked = sum(1 for p in self.persons.values() if p.has_embarked or p.track_id in self.already_counted)
            print(f"Total embarcados: {total_embarked}")
            print(f"\nArchivos en: {self.output_folder}/")
            print("="*60 + "\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Contador de pasajeros teleférico v2")
    parser.add_argument('minutos', type=float, help='Duración en minutos')
    parser.add_argument('--modelo', default=MODEL_PATH, help='Ruta al modelo YOLO')
    parser.add_argument('--rtsp', default=RTSP_URL, help='URL RTSP')
    parser.add_argument('--output', default=OUTPUT_FOLDER, help='Carpeta de salida')
    parser.add_argument('--conf', type=float, default=CONF_THRESHOLD, help='Umbral confianza')
    
    args = parser.parse_args()
    
    counter = TelefericoCounterV2(
        model_path=args.modelo,
        rtsp_url=args.rtsp,
        output_folder=args.output,
        conf_threshold=args.conf,
        fps_video=FPS_VIDEO
    )
    
    counter.run(args.minutos)


if __name__ == "__main__":
    main()

