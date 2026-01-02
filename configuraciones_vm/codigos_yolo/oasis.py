#!/usr/bin/env python3
"""
Sistema Avanzado de Conteo de Pasajeros v3.3
===========================================
- Respaldo en Azure Blob Storage (Gen2) con logs detallados.
- Estructura de CSV optimizada (Sin timestamp, fecha al inicio).
- Feedback en tiempo real (Inferencia y Conexión).
"""

import cv2
import numpy as np
import argparse
import time
import csv
import os
import tempfile
import logging
import pandas as pd
from datetime import datetime
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
from ultralytics import YOLO
from azure.storage.filedatalake import DataLakeServiceClient

# ============================================================================
# CONFIGURACIÓN DE LOGS (Para ver el rastro de Azure)
# ============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
# Habilitar logs específicos de la infraestructura de Azure
logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.INFO)
logging.getLogger('azure.storage').setLevel(logging.INFO)

# ============================================================================
# CONFIGURACIÓN DE AZURE Y RUTAS
# ============================================================================
# Cadena de conexión a Azure Storage Account (Debe ser segura)
AZURE_CONNECTION_STRING = "STRING CONNECTION DE BLOBSTORAGE"
# Nombre del contenedor en Azure Blob Storage donde se guardarán los datos
CONTAINER_NAME = "conteo-teleferico"

# URL del flujo de video RTSP de la cámara
RTSP_URL = "rtsp://usertf:Tfo.-2525@172.16.10.22:554/cam/realmonitor?channel=5&subtype=0"
# Ruta al modelo YOLO entrenado para detectar cabezas y cabinas
MODEL_PATH = "/home/aaravenatk/yolo/best.pt"
# Umbral de confianza para las detecciones (0.7 = 70%)
CONF_THRESHOLD = 0.7
# Carpeta local donde se guardarán temporalmente los CSV
OUTPUT_FOLDER = "detecciones"

# Geometría y Parámetros
# Coordenadas de la línea virtual que cruzan las cabinas para ser contadas
LINEA_TELEFERICO = [(561, 110), (570, 465)]
# Polígono que define la zona de embarque donde se cuentan las personas
ZONA_EMBARQUE = [(570, 168), (809, 164), (897, 159), (984, 185), (997, 388), (575, 359)]
# Tiempo en segundos para considerar a una persona como operador (no pasajero)
TIEMPO_OPERADOR = 90.0
# Tiempo máximo para re-identificar a una persona perdida
TIEMPO_REIDENTIFICACION = 10.0
DISTANCIA_MAX_REIDENT = 200
SIMILARIDAD_BBOX_MIN = 0.6
# IDs de las clases del modelo YOLO
CLASE_CABEZA = 0
CLASE_CABINA = 1

@dataclass
class PersonTracker:
    """
    Clase para rastrear el estado de cada persona detectada.
    Mantiene historial de posiciones, velocidad y estado (si es operador, si está en zona, etc.)
    """
    track_id: int
    first_seen: float
    last_seen: float
    history: deque = field(default_factory=lambda: deque(maxlen=100))
    in_zone: bool = False
    is_operator: bool = False
    avg_velocity: Tuple[float, float] = (0.0, 0.0)
    last_bbox_size: Tuple[int, int] = (0, 0)
    is_lost: bool = False
    lost_timestamp: float = 0.0

    def add_position(self, x, y, timestamp, bbox=None):
        """Actualiza la posición y calcula la velocidad promedio."""
        self.history.append((x, y, timestamp))
        self.last_seen = timestamp
        if bbox: self.last_bbox_size = (bbox[2]-bbox[0], bbox[3]-bbox[1])
        if len(self.history) >= 2:
            dt = self.history[-1][2] - self.history[0][2]
            if dt > 0:
                self.avg_velocity = ((self.history[-1][0]-self.history[0][0])/dt, (self.history[-1][1]-self.history[0][1])/dt)

    @property
    def time_tracked(self) -> float:
        """Retorna el tiempo total que la persona ha sido rastreada."""
        return self.last_seen - self.first_seen

    def predict_position(self, dt):
        """Predice la posición futura basada en la velocidad actual."""
        if not self.history: return (0,0)
        return (self.history[-1][0] + self.avg_velocity[0]*dt, self.history[-1][1] + self.avg_velocity[1]*dt)

    def bbox_similarity(self, other_size):
        """Calcula la similitud de tamaño entre bounding boxes."""
        if self.last_bbox_size == (0,0): return 0.0
        return (min(self.last_bbox_size[0], other_size[0])/max(self.last_bbox_size[0], other_size[0]) + 
                min(self.last_bbox_size[1], other_size[1])/max(self.last_bbox_size[1], other_size[1])) / 2

@dataclass
class CabinEvent:
    """
    Clase para registrar un evento de paso de cabina.
    Almacena qué cabina pasó, cuándo y cuántas personas embarcaron.
    """
    cabin_id: int
    track_id: Optional[int]
    cross_time: float
    persons_embarked: int = 0
    embarked_ids: List[int] = field(default_factory=list)

class TelefericoCounterV3:
    def __init__(self, model_path, rtsp_url, output_folder, conf):
        """Inicializa el contador, carga el modelo y configura el tracker."""
        logger.info(f"Cargando modelo YOLO: {model_path}")
        self.model = YOLO(model_path)
        self.rtsp_url = rtsp_url
        self.output_folder = output_folder
        self.conf = conf
        # Diccionarios para mantener el estado de personas y cabinas
        self.persons: Dict[int, PersonTracker] = {}
        self.lost_persons: Dict[int, PersonTracker] = {}
        self.prev_cabin_positions = {}
        self.cabin_count = 0
        self.current_event = None
        self.total_embarked_count = 0
        self.track_id_map = {}
        self.csv_path = None
        self.timestamp_str = datetime.now().strftime("%d-%m-%y_%H-%M-%S")
        
        # Configuración específica para el algoritmo de tracking ByteTrack
        self.tracker_config = os.path.join(tempfile.gettempdir(), "bytetrack_cfg.yaml")
        with open(self.tracker_config, 'w') as f:
            f.write("tracker_type: bytetrack\ntrack_high_thresh: 0.5\ntrack_low_thresh: 0.1\nnew_track_thresh: 0.6\ntrack_buffer: 45\nmatch_thresh: 0.8\nfuse_score: true")

    def _connect_to_camera(self, max_retries=10, retry_delay=3.0):
        """Intenta conectar a la cámara RTSP con reintentos automáticos."""
        for attempt in range(1, max_retries + 1):
            logger.info(f"Conectando a cámara RTSP... Intento {attempt}/{max_retries}")
            cap = cv2.VideoCapture(self.rtsp_url)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret: 
                    logger.info("✅ Conexión establecida con éxito.")
                    return cap
            cap.release()
            time.sleep(retry_delay)
        raise RuntimeError("❌ Fallo crítico de conexión RTSP.")

    def _subir_solo_a_blob_storage(self):
        """Reestructuración de columnas y subida con logs de red detallados."""
        if not self.csv_path or not os.path.exists(self.csv_path): return

        nombre_archivo = os.path.basename(self.csv_path)
        try:
            # 1. TRANSFORMACIÓN CON PANDAS
            # Se lee el CSV generado y se separan las fechas y horas en columnas individuales
            logger.info(f"Reestructurando columnas para {nombre_archivo}...")
            df = pd.read_csv(self.csv_path)
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['Fecha'] = df['timestamp'].dt.strftime('%d-%m-%Y')
            df['hora_minuto'] = df['timestamp'].dt.strftime('%H:%M')

            # Reemplazar timestamp por las columnas al inicio para facilitar consultas
            columnas = ['Fecha', 'hora_minuto', 'cabina_numero', 'personas_embarcadas', 'hora_paso', 'ids_personas', 'ignore']
            df = df[columnas]
            df.to_csv(self.csv_path, index=False)

            # 2. SUBIDA A AZURE (Logs de red activos)
            # Se conecta al servicio de Azure Data Lake Storage Gen2 y se sube el archivo
            logger.info(f"Iniciando transferencia a Azure...")
            service_client = DataLakeServiceClient.from_connection_string(AZURE_CONNECTION_STRING, logging_enable=True)
            fs_client = service_client.get_file_system_client(file_system=CONTAINER_NAME)
            
            directory_client = fs_client.get_directory_client("conteos-diarios")
            file_client = directory_client.create_file(nombre_archivo)
            
            with open(self.csv_path, "rb") as data:
                file_client.append_data(data, offset=0, length=os.path.getsize(self.csv_path))
                file_client.flush_data(os.path.getsize(self.csv_path))

            # 3. LIMPIAR
            # Se elimina el archivo local una vez subido exitosamente
            # file_client.rename_file(f"{CONTAINER_NAME}/procesados/{nombre_archivo}")
            os.remove(self.csv_path)
            logger.info(f"✅ Archivo finalizado en: {CONTAINER_NAME}/conteos-diarios/")

        except Exception as e:
            logger.error(f"Fallo en el pipeline de Azure: {e}")

    def run(self, minutes):
        """
        Bucle principal de procesamiento de video.
        Captura frames, ejecuta inferencia, rastrea objetos y cuenta eventos.
        """
        cap = self._connect_to_camera()
        self.csv_path = os.path.join(self.output_folder, f"conteo_{self.timestamp_str}.csv")
        
        # Inicializar CSV con cabeceras
        with open(self.csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(['timestamp', 'cabina_numero', 'personas_embarcadas', 'hora_paso', 'ids_personas', 'ignore'])

        start_t = time.time()
        frame_count = 0
        
        try:
            while (time.time() - start_t) < minutes * 60:
                ret, frame = cap.read()
                if not ret: break
                
                frame_count += 1
                # Ejecutar tracking con YOLO y ByteTrack
                results = self.model.track(frame, persist=True, tracker=self.tracker_config, conf=self.conf, verbose=False)
                curr_t = time.time()
                
                heads, cabins = [], []
                if results[0].boxes.id is not None:
                    for box in results[0].boxes:
                        bbox = tuple(map(int, box.xyxy[0]))
                        raw_id = int(box.id[0])
                        cls = int(box.cls[0])
                        # Separar detecciones en cabezas (personas) y cabinas
                        if cls == CLASE_CABEZA: heads.append({'bbox': bbox, 'track_id': raw_id})
                        else: cabins.append({'bbox': bbox, 'track_id': raw_id})

                # Feedback en tiempo real (Inferencia)
                if frame_count % 30 == 0:
                    print(f"FPS: {frame_count} | Cabezas: {len(heads)} | Cabinas: {len(cabins)} | Embarcados: {self.total_embarked_count}")

                # Lógica de conteo acumulativo
                curr_ids = set()
                for h in heads:
                    raw_id, bbox = h['track_id'], h['bbox']
                    pos = ((bbox[0]+bbox[2])/2, bbox[3])
                    tid = self.track_id_map.get(raw_id, raw_id)
                    
                    # Registrar nueva persona si no existe
                    if tid not in self.persons:
                        self.persons[tid] = PersonTracker(tid, curr_t, curr_t)
                    p = self.persons[tid]
                    p.add_position(pos[0], pos[1], curr_t, bbox)
                    
                    # Verificar si está en zona de embarque
                    p.in_zone = cv2.pointPolygonTest(np.array(ZONA_EMBARQUE, np.int32), pos, False) >= 0
                    # Detectar operadores por tiempo de permanencia
                    if not p.is_operator: p.is_operator = p.time_tracked > TIEMPO_OPERADOR
                    curr_ids.add(tid)

                # Procesar personas que desaparecieron del cuadro (posiblemente embarcaron)
                for tid in list(set(self.persons.keys()) - curr_ids):
                    p = self.persons[tid]
                    if p.is_lost: continue
                    p.is_lost, p.lost_timestamp = True, curr_t
                    self.lost_persons[tid] = p
                    
                    # Si desapareció en zona de embarque y no es operador, contar como embarcado
                    if p.in_zone and not p.is_operator and self.current_event:
                        if tid not in self.current_event.embarked_ids:
                            self.current_event.persons_embarked += 1
                            self.current_event.embarked_ids.append(tid)
                            self.total_embarked_count += 1

                # Cruce de cabinas
                for c in cabins:
                    tid, bbox = c['track_id'], c['bbox']
                    center = ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)
                    
                    # Detectar cruce de línea virtual
                    if tid in self.prev_cabin_positions:
                        l = LINEA_TELEFERICO
                        # Producto cruz para determinar si cruzó la línea
                        if ((l[1][0]-l[0][0])*(self.prev_cabin_positions[tid][1]-l[0][1])-(l[1][1]-l[0][1])*(self.prev_cabin_positions[tid][0]-l[0][0])) * \
                           ((l[1][0]-l[0][0])*(center[1]-l[0][1])-(l[1][1]-l[0][1])*(center[0]-l[0][0])) < 0:
                            
                            # Si ya había un evento activo, guardarlo en CSV
                            if self.current_event:
                                with open(self.csv_path, 'a', newline='') as f:
                                    ev = self.current_event
                                    csv.writer(f).writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ev.cabin_id, ev.persons_embarked, datetime.fromtimestamp(ev.cross_time).strftime("%H:%M:%S"), ",".join(map(str, ev.embarked_ids)), 0])
                            
                            # Iniciar nuevo evento de cabina
                            self.cabin_count += 1
                            self.current_event = CabinEvent(self.cabin_count, tid, curr_t)
                    self.prev_cabin_positions[tid] = center

        finally:
            # Guardar último evento pendiente y subir a Azure al finalizar
            if self.current_event and self.csv_path:
                with open(self.csv_path, 'a', newline='') as f:
                    ev = self.current_event
                    csv.writer(f).writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ev.cabin_id, ev.persons_embarked, datetime.fromtimestamp(ev.cross_time).strftime("%H:%M:%S"), ",".join(map(str, ev.embarked_ids)), 0])
            cap.release()
            print("\nIniciando respaldo en Azure Blob Storage...")
            self._subir_solo_a_blob_storage()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('minutos', type=float)
    args = parser.parse_args()
    TelefericoCounterV3(MODEL_PATH, RTSP_URL, OUTPUT_FOLDER, CONF_THRESHOLD).run(args.minutos)
