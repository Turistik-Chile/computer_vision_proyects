# Configurar Drivers

1. Actualiza los repositorios

sudo apt update && sudo apt upgrade -y

2. Instala la herramienta de detección

sudo apt install -y ubuntu-drivers-common

3. Instala el controlador recomendado automáticamente

sudo ubuntu-drivers install

4. Reinicia la máquina virtual

sudo reboot

5. Revisar GPU

nvidia-smi

---

# Instalar Python y ambiente virtual

1. Instalar python y virtual environment

sudo apt install python3 python3-pip python3-venv

2. Crear ambiente virtual y activarlo

python3 -m venv yolo
source /yolo/bin/activate

3. Instalar librerias necesarias para ejecución de scripts en python.

(Antes de instalar se requiere revisar la versión de cuda del controlador de la tarjeta gráfica)

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129

pip install flask ultralytics opencv-python opencv-python-headless shapely numpy

# Configuración de Modelo y Yolov.

Se tiene que traspasar el archivo de peso ".pt" dentro de la carpeta /yolo/

El script principal es "oasis.py" el cual tiene las configuraciones y parametros de la inferencia. Para ello se tiene que ejecutar con el comando

** python oasis.py 60**

El parametro de entrada es la cantidad de minutos a ejecutarse "60 minutos"

La salida es un archivo csv que indica la cantidad de personas que ingresan por cabina.

# Script Automatico.

Se crea el script en .sh "ejecutar.sh" el cual ejecuta el servicio de manera automatica para los servicios de powerautomate, se ejecuta ese script para generar el proceso de inferencia. Dentro del script se especifica un parámetro para la cantidad de minutos.
