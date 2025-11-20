# Computer Vision Projects

Repositorio enfocado en el desarrollo de herramientas con computer vision utilizando YOLO para detección de objetos.

## Descripción del Proyecto

Este proyecto utiliza **YOLOv11** de [Ultralytics](https://github.com/ultralytics/ultralytics) como modelo de detección de objetos.

## Entrenamiento

El entrenamiento se realizó utilizando la plataforma **Google Colab** con una instancia **A100** de alta capacidad de RAM para optimizar los tiempos de procesamiento.

## Pipeline de Trabajo

### 1. Pre-procesamiento de Videos

Los clips de video se extraen desde el software **SmartPSS Lite** en formato `.mp4`. Para extraer fotogramas a lo largo de clips largos, se utiliza el script `videocutter.py`, que permite especificar intervalos de tiempo personalizados para la extracción de frames y los guarda con nombres secuenciales.

### 2. Etiquetamiento con CVAT

Para el proceso de etiquetamiento se utiliza **CVAT** mediante Docker:

1. Clonar el repositorio de CVAT:

   ```bash
   git clone https://github.com/cvat-ai/cvat
   ```

2. Ejecutar los contenedores dentro del repositorio clonado según las instrucciones del repositorio.

**Recomendaciones:**

- Para tareas con muchas imágenes, se recomienda conectar el repositorio de forma local con el servidor, ya que CVAT tiene una limitación de 100 archivos al arrastrar y soltar.
- Para etiquetado de videos, la carga directa no presenta mayores inconvenientes.

### 3. Exportación de Etiquetas

Una vez completadas las tareas de etiquetamiento:

- Exportar las etiquetas en formato **YOLO detection**
- **Se recomienda guardar las imágenes etiquetadas** para futuras referencias

### 4. Preparación del Dataset

Utilizar el notebook `reorganizar_dataset.ipynb` para:

- Organizar las carpetas del dataset
- Generar el archivo `.yaml` necesario para el entrenamiento
- Estructurar correctamente el dataset de entrenamiento

### 5. Entrenamiento Final

Una vez preparado el dataset:

1. Comprimir el dataset completo
2. Subir a la nube (Google Drive u otro servicio)
3. Procesarlo en Google Colab para realizar los entrenamientos

## Estructura del Proyecto

- `model_cabinas/`: Modelos entrenados para detección de cabinas
- `model_personas/`: Modelos entrenados para detección de personas
- `pesos_bases/`: Pesos pre-entrenados de YOLOv11 (l, m, s)
- `videocutter.py`: Script para extracción de fotogramas
- `reorganizar_dataset.ipynb`: Notebook para preparación del dataset
