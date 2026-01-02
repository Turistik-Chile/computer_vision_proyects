"""
Aplicación Web Flask para Monitoreo en Tiempo Real de Teleférico
================================================================
Sistema de visualización web con streaming de video RTSP y API REST
para control remoto de grabación y conteo de detecciones.
"""

import cv2
import time
import datetime
import csv
import torch
import gc
import os
import threading
import queue
import numpy as np
from flask import Flask, Response, jsonify
from shapely.geometry import Point, Polygon
from ultralytics import solutions

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
RTSP_URL = "rtsp://usertf:Tfo.-2525@172.16.10.22:554/cam/realmonitor?channel=5&subtype=0"
MODEL_PATH = "/home/aaravenatk/yolo/best.pt"
CONF_THRESHOLD = 0.7
MIN_FRAMES = 15 # Bajamos un poco para que detecte más rápido
OUTPUT_FOLDER = "registros_web"

# Resolución WEB (Para visualizar)
WEB_RES = (1280, 720) 

# Zonas
polygon_personas = [(445, 299), (812, 341), (1061, 330), (1046, 153), (428, 177)]
polygon_teleferico = [(149, 136), (435, 114), (445, 417), (161, 443)]

region_points = { "personas": polygon_personas, "teleferico": polygon_teleferico }
person_poly = Polygon(polygon_personas)
tele_poly = Polygon(polygon_teleferico)
CLASS_MAP = {0: "person", 1: "teleferico"}

if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)

app = Flask(__name__)

# ==========================================
# 2. CLASE PARA CÁMARA SIN LAG (THREADED)
# ==========================================
# Implementa captura de frames en hilo separado para reducir latencia
# y evitar buffers llenos en streams RTSP
class ThreadedCamera:
    def __init__(self, src=0):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        # Configurar buffer pequeño (ayuda en algunos backends)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            print("⚠️ La cámara ya está corriendo en un hilo.")
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True # Se muere si cierras el programa principal
        self.thread.start()
        print("⚡ Cámara iniciada en hilo paralelo (Modo Baja Latencia)")
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
            # Pequeña pausa para no quemar CPU innecesariamente si la cámara es lenta
            time.sleep(0.005) 

    def read(self):
        with self.read_lock:
            # Retornamos una copia para evitar conflictos de escritura/lectura
            if self.frame is not None:
                return self.grabbed, self.frame.copy()
            return self.grabbed, None

    def stop(self):
        self.started = False
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()

# ==========================================
# 3. ESTADO GLOBAL
# ==========================================
# Clase que mantiene el estado compartido de la aplicación:
# contadores, historial de tracking, archivos CSV activos
class SystemState:
    def __init__(self):
        self.is_recording = False
        self.camera_active = False
        self.counts = {"person": 0, "teleferico": 0}
        self.track_history = {}
        self.unique_ids = set()
        self.csv_file = None
        self.csv_writer = None
        self.cam_thread = None # Aquí guardaremos nuestro objeto ThreadedCamera

    def start_recording(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(OUTPUT_FOLDER, f"session_{timestamp}.csv")
        self.csv_file = open(filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["Timestamp", "Clase", "ID", "Confianza"])
        self.is_recording = True
        print(f"✅ Grabación iniciada: {filename}")

    def stop_recording(self):
        self.is_recording = False
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
        print("🛑 Grabación detenida.")

    def reset_counters(self):
        print("🔄 Reiniciando contadores...")
        self.counts = {"person": 0, "teleferico": 0}
        self.unique_ids = set()
        self.track_history = {}

state = SystemState()

# ==========================================
# Carga del modelo YOLO y configuración del contador por regiones
# 4. LÓGICA DE VISIÓN
# ==========================================
def load_model():
    try:
        print("Cargando modelo YOLO...")
        return solutions.RegionCounter(
            model=MODEL_PATH, region=region_points, show=False,
            tracker="bytetrack.yaml", conf=CONF_THRESHOLD
        )
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        return None
# Generador de frames para streaming MJPEG
# Procesa inferencia, dibuja anotaciones y codifica en JPEG

rc = load_model()

def generar_frames():
    while True:
        # 1. GESTIÓN DE CONEXIÓN
        if not state.camera_active:
            # Si se desactivó, apagamos el hilo para liberar recursos
            if state.cam_thread is not None:
                state.cam_thread.stop()
                state.cam_thread = None
            
            # Placeholder NEGRO
            blank = np.zeros((720, 1280, 3), np.uint8)
            cv2.putText(blank, "SISTEMA EN ESPERA", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (50, 50, 50), 2)
            ret, buffer = cv2.imencode('.jpg', blank)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.5)
            continue

        # Iniciamos la cámara si no existe
        if state.cam_thread is None:
            try:
                state.cam_thread = ThreadedCamera(RTSP_URL).start()
                # Damos un segundo para que llene el primer frame
                time.sleep(1)
            except Exception as e:
                print(f"Error conectando: {e}")
                state.camera_active = False
                continue

        # 2. OBTENER ÚLTIMO FRAME (Sin Buffer)
        success, frame = state.cam_thread.read()
        
        if not success or frame is None:
            # Si falla la lectura, enviamos espera pero NO detenemos el hilo inmediatamente
            time.sleep(0.1)
            continue

        # --- OPTIMIZACIÓN CRÍTICA: REDIMENSIONAR ANTES DE PROCESAR ---
        # Si tienes lag, es probable que la inferencia en 1080p sea lenta.
        # Procesaremos sobre la imagen ya redimensionada a 720p (WEB_RES)
        # Esto aumenta muchisimo los FPS.
        frame_process = cv2.resize(frame, WEB_RES)

        # 3. INFERENCIA
        if rc: _ = rc(frame_process)
        
        # 4. EXTRACCIÓN Y CONTEO
        # (Nota: Como redimensionamos, las coordenadas de YOLO ya estarán en escala 1280x720)
        # Asegurate de que tus polígonos estén pensados para 1280x720 o ajusta la escala
        
        boxes = getattr(rc, "boxes", [])
        track_ids = getattr(rc, "track_ids", [])
        clss = getattr(rc, "clss", [])

        if track_ids is not None and len(track_ids) > 0:
             # Ajuste para evitar error de zip si longitudes difieren
            limit = min(len(boxes), len(track_ids), len(clss))
            timestamp_now = datetime.datetime.now().strftime("%H:%M:%S")

            for i in range(limit):
                try:
                    box = boxes[i]
                    cls_idx = int(clss[i])
                    tid = int(track_ids[i])
                    cls_name = CLASS_MAP.get(cls_idx, "unknown")
                    
                    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                    pt = Point(cx, cy)
                    
                    is_inside = False
                    if cls_name == "person" and person_poly.contains(pt): is_inside = True
                    elif cls_name == "teleferico" and tele_poly.contains(pt): is_inside = True
                    
                    if is_inside:
                        if tid not in state.track_history: state.track_history[tid] = 0
                        state.track_history[tid] += 1
                        
                        if state.track_history[tid] >= MIN_FRAMES:
                            if state.is_recording and tid not in state.unique_ids:
                                state.unique_ids.add(tid)
                                state.counts[cls_name] += 1
                                if state.csv_writer:
                                    state.csv_writer.writerow([timestamp_now, cls_name, tid])
                except: pass

        # 5. DIBUJAR (Sobre frame_process que ya tiene tamaño correcto)
        cv2.polylines(frame_process, [np.array(polygon_personas, np.int32)], True, (0, 255, 0), 2)
        cv2.polylines(frame_process, [np.array(polygon_teleferico, np.int32)], True, (0, 0, 255), 2)

        if state.is_recording:
            cv2.circle(frame_process, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame_process, "REC", (50, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame_process)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# Definición de endpoints HTTP para interfaz y control
# ==========================================
# 5. RUTAS WEB Y API
# ==========================================
@app.route('/')
def index():
    # Usando el diseño DARK que te gustó (app_v2 style)
    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>Panel PonLab</title>
        <style>
            body { font-family: 'Segoe UI', sans-serif; background: #121212; color: #e0e0e0; margin: 0; padding: 20px; text-align: center; }
            .main-wrapper { max-width: 1100px; margin: 0 auto; display: flex; gap: 20px; align-items: flex-start;}
            
            .video-container { flex: 3; background: #000; border: 2px solid #333; border-radius: 8px; overflow: hidden; aspect-ratio: 16/9; }
            .video-container img { width: 100%; height: 100%; object-fit: contain; }

            .sidebar { flex: 1; display: flex; flex-direction: column; gap: 20px; }
            
            .card { background: #1e1e1e; padding: 20px; border-radius: 12px; border-top: 4px solid #00d2ff; }
            .card.red { border-top-color: #dc3545; }
            .card.green { border-top-color: #28a745; }
            .card h2 { font-size: 3em; margin: 0; color: #fff; }
            .card p { margin: 5px 0 0; color: #aaa; font-size: 0.9em; text-transform: uppercase; }

            button { padding: 15px; width: 100%; font-size: 15px; border: none; border-radius: 6px; cursor: pointer; font-weight: 600; color: white; margin-bottom: 10px; }
            .btn-cam { background: #444; }
            .btn-cam.active { background: #dc3545; }
            .btn-rec { background: #28a745; }
            .btn-rec:disabled { background: #333; color: #666; }
            .btn-rec.recording { background: #ffc107; color: black; animation: pulse 1.5s infinite; }
            .btn-reset { background: #17a2b8; }
            
            @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.6; } 100% { opacity: 1; } }
            h1 { width: 100%; margin-bottom: 30px;}
        </style>
    </head>
    <body>
        <h1>📡 Monitor PonLab - Tiempo Real</h1>
        <div class="main-wrapper">
            <div class="video-container">
                <img id="video-stream" src="" alt="Cámara Detenida">
            </div>

            <div class="sidebar">
                <div class="card green">
                    <h2 id="count-person">0</h2>
                    <p>Personas</p>
                </div>
                <div class="card red">
                    <h2 id="count-tele">0</h2>
                    <p>Teleférico</p>
                </div>
                
                <div style="margin-top: 20px;">
                    <button id="btn-cam" class="btn-cam" onclick="toggleCamera()">📷 Activar Cámara</button>
                    <button id="btn-rec" class="btn-rec" onclick="toggleRecording()" disabled>▶ Iniciar Recolección</button>
                    <button id="btn-reset" class="btn-reset" onclick="resetStats()">🗑 Reiniciar Contadores</button>
                </div>
            </div>
        </div>

        <script>
            let cameraActive = false;
            let recording = false;

            async function toggleCamera() {
                const btn = document.getElementById('btn-cam');
                const btnRec = document.getElementById('btn-rec');
                const img = document.getElementById('video-stream');
                
                const res = await fetch('/api/toggle_camera', {method: 'POST'});
                const data = await res.json();
                cameraActive = data.active;

                if (cameraActive) {
                    img.src = "/video_feed?" + new Date().getTime();
                    btn.textContent = "⏹ Detener Cámara";
                    btn.classList.add("active");
                    btnRec.disabled = false;
                } else {
                    img.src = "";
                    btn.textContent = "📷 Activar Cámara";
                    btn.classList.remove("active");
                    btnRec.disabled = true;
                    if(recording) toggleRecording();
                }
            }

            async function toggleRecording() {
                const btn = document.getElementById('btn-rec');
                const endpoint = recording ? '/api/stop_rec' : '/api/start_rec';
                const res = await fetch(endpoint, {method: 'POST'});
                const data = await res.json();
                recording = data.recording;

                if (recording) {
                    btn.textContent = "⏸ Pausar";
                    btn.classList.add("recording");
                } else {
                    btn.textContent = "▶ Iniciar Recolección";
                    btn.classList.remove("recording");
                }
            }

            async function resetStats() {
                if(!confirm("¿Resetear a 0?")) return;
                await fetch('/api/reset', {method: 'POST'});
                document.getElementById('count-person').innerText = "0";
                document.getElementById('count-tele').innerText = "0";
            }

            setInterval(async () => {
                if (cameraActive) {
                    try {
                        const res = await fetch('/api/stats');
                        const stats = await res.json();
                        document.getElementById('count-person').innerText = stats.person;
                        document.getElementById('count-tele').innerText = stats.teleferico;
                    } catch(e) {}
                }
            }, 1000);
        </script>
    </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generar_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- API ---
@app.route('/api/toggle_camera', methods=['POST'])
def api_toggle_camera():
    state.camera_active = not state.camera_active
    return jsonify({"active": state.camera_active})

@app.route('/api/start_rec', methods=['POST'])
def api_start_rec():
    if not state.is_recording: state.start_recording()
    return jsonify({"recording": True})

@app.route('/api/stop_rec', methods=['POST'])
def api_stop_rec():
    if state.is_recording: state.stop_recording()
    return jsonify({"recording": False})

@app.route('/api/reset', methods=['POST'])
def api_reset():
    state.reset_counters()
    return jsonify({"status": "ok"})

@app.route('/api/stats')
def api_stats():
    return jsonify(state.counts)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
