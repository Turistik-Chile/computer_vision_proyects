import cv2
import os

def extraer_fotogramas(video_path, output_folder):
    # Crear carpeta de salida si no existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Obtener nombre base del archivo sin extensión
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    # Cargar el video
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duracion = total_frames / fps if fps > 0 else 0

    print(f"\n🎬 Archivo: {base_name}")
    print(f"📹 FPS detectados: {fps:.2f}")
    print(f"⏱️ Duración estimada: {duracion:.2f} segundos ({duracion/60:.2f} min)\n")

    # Pedir al usuario el intervalo
    while True:
        try:
            intervalo_segundos = float(input("⏳ Ingrese cada cuántos segundos desea extraer un fotograma: "))
            if intervalo_segundos <= 0:
                print("⚠️ El intervalo debe ser mayor que 0.")
                continue
            break
        except ValueError:
            print("⚠️ Ingrese un número válido, por ejemplo 2.5 o 3.")

    # Calcular cada cuántos fotogramas guardar uno
    salto_frames = int(fps * intervalo_segundos)
    count = 0
    saved = 0

    print("\n⏺️ Extrayendo fotogramas...\n")

    while True:
        ret, frame = video.read()
        if not ret:
            break

        if count % salto_frames == 0:
            filename = os.path.join(output_folder, f"{base_name}_frame_{saved:04d}.jpg")
            cv2.imwrite(filename, frame)
            saved += 1

        count += 1

    video.release()
    print(f"✅ {saved} fotogramas guardados en '{output_folder}' con prefijo '{base_name}_'")

# --- Ejemplo de uso ---
video_path = "Cumbre1_1819.mp4"              # Ruta del video
output_folder = "frames_extraidos"    # Carpeta de salida

extraer_fotogramas(video_path, output_folder)
