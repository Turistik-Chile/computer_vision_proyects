#!/bin/bash

# --- CONFIGURACIÓN ---
HOME_DIR="/home/aaravenatk"
PROJECT_DIR="$HOME_DIR/yolo"
PYTHON_VENV="$PROJECT_DIR/bin/python"
SCRIPT_PYTHON="oasis.py"
LOGS_DIR="$HOME_DIR/logs"   # Nueva carpeta exclusiva para logs

# Generamos una marca de tiempo única (Ej: log_2025-11-28_18-30-05.txt)
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="$LOGS_DIR/log_$TIMESTAMP.txt"

MINUTOS=180

# --- PASO 1: Crear carpetas necesarias ---
# Creamos la carpeta de logs si no existe (el -p evita errores si ya existe)
mkdir -p $LOGS_DIR

# --- PASO 2: Moverse al directorio del proyecto ---
cd $PROJECT_DIR

# --- PASO 3: Iniciar Log ---
echo "========================================" > $LOG_FILE
echo "Iniciando ejecución: $(date)" >> $LOG_FILE
echo "Directorio de trabajo: $(pwd)" >> $LOG_FILE
echo "Log guardado en: $LOG_FILE" >> $LOG_FILE

# --- PASO 4: Ejecutar Python ---
# Todo lo que salga por pantalla (errores o prints) va al archivo con fecha única
$PYTHON_VENV -u $SCRIPT_PYTHON $MINUTOS >> $LOG_FILE 2>&1

# --- PASO 5: Corregir Permisos ---
# Devolvemos la propiedad de la carpeta logs y del archivo actual a tu usuario
chown -R aaravenatk:aaravenatk $LOGS_DIR
chown -R aaravenatk:aaravenatk $PROJECT_DIR/detecciones

echo "Fin de la ejecución: $(date)" >> $LOG_FILE
echo "========================================" >> $LOG_FILE
