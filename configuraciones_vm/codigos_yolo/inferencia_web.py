import cv2
import torch
import numpy as np
from flask import Flask, Response, render_template_string
from shapely.geometry import Polygon, Point
from ultralytics import solutions

# ==========================================
# 1. TUS PARÁMETROS
# ==========================================
RTSP_URL = "rtsp://usertf:Tfo.-2525@172.16.10.22:554/cam/realmonitor?channel=5&subtype=0"
MODEL_PATH = "/home/aaravenatk/yolo/best.pt"
CONF_THRESHOLD = 0.7

# Zonas
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

# Configuración de Flask
app = Flask(__name__)

# ==========================================
# 2. INICIALIZACIÓN
# ==========================================
try:
    print("Cargando modelo...")
    rc = solutions.RegionCounter(
        model=MODEL_PATH,
        region=region_points,
        show=False,
        tracker="bytetrack.yaml",
        conf=CONF_THRESHOLD,
    )
except Exception as e:
    print(f"Error modelo: {e}")
    exit(1)

# ==========================================
# 3. GENERADOR DE VIDEO
# ==========================================
def generar_frames():
    # Conectar cámara dentro del generador para poder reconectar si se cae la página
    cap = cv2.VideoCapture(RTSP_URL)
    
    # Reducir resolución para que la transmisión sea fluida por red
    # (Opcional, pero recomendado para web)
    
    with torch.no_grad():
        while True:
            success, frame = cap.read()
            if not success:
                # Si falla la cámara, enviamos una imagen negra o reintentamos
                cap.release()
                cap = cv2.VideoCapture(RTSP_URL)
                continue

            # 1. Inferencia
            _ = rc(frame)
            
            # 2. Recuperar imagen anotada
            # Intentamos obtener la imagen dibujada por YOLO. Si falla, usamos la original.
            frame_visual = getattr(rc, "im0", frame.copy())
            
            # 3. DIBUJAR POLÍGONOS (Visualización Forzada)
            # Dibujamos las zonas en colores brillantes para verlas bien en la web
            pts_p = np.array(polygon_personas, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame_visual, [pts_p], True, (0, 255, 0), 2) # Verde
            
            pts_t = np.array(polygon_teleferico, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame_visual, [pts_t], True, (0, 0, 255), 2) # Rojo

            # 4. Convertir a formato Web (JPG)
            ret, buffer = cv2.imencode('.jpg', frame_visual)
            frame_bytes = buffer.tobytes()

            # 5. Enviar el frame al navegador
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ==========================================
# 4. RUTAS WEB
# ==========================================
@app.route('/')
def index():
    # Página HTML simple
    return """
    <html>
        <head><title>Monitor de Inferencia YOLO</title></head>
        <body style="background:black; color:white; text-align:center;">
            <h1>Vista en Tiempo Real</h1>
            <h3>Verde: Personas | Rojo: Teleférico</h3>
            <img src="/video_feed" width="800" style="border: 2px solid white;">
        </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generar_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ==========================================
# 5. EJECUCIÓN
# ==========================================
if __name__ == '__main__':
    print("--- SERVIDOR WEB INICIADO ---")
    print("Abre tu navegador (Chrome/Edge) y entra a:")
    print("http://<IP_DE_TU_UBUNTU>:5000")
    print("-----------------------------")
    # host='0.0.0.0' permite que entres desde otra PC en la red
    app.run(host='0.0.0.0', port=5000, debug=False)
