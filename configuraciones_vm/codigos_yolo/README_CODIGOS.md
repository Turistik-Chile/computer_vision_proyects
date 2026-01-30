# Documentación de Códigos YOLO - Sistema de Conteo de Teleférico

Este documento describe la funcionalidad y diferencias entre los distintos scripts de detección y conteo desarrollados para el sistema de teleférico.

> **⚠️ IMPORTANTE:** El código principal en producción es **`oasis.py`**, que se ejecuta mediante el script `ejecutar.sh`. Este es el sistema activo con integración a Azure Blob Storage y todas las funcionalidades de tracking avanzado.

---

## 📋 Índice de Códigos

### 1. **app.py**

**Funcionalidad:** Aplicación web con Flask para monitoreo en tiempo real  
**Características principales:**

- Interfaz web con visualización en vivo del stream RTSP
- Sistema de threading para reducir latencia de cámara
- Control de inicio/parada de grabación mediante API REST
- Panel de estadísticas en tiempo real (personas y cabinas detectadas)
- Conteo automático con filtrado por zonas poligonales
- Grabación de datos en CSV durante sesiones activas

**Uso:** Ideal para monitoreo remoto con múltiples usuarios conectados simultáneamente.

---

### 2. **blob.py** (alias: oasis.py en versión antigua)

**Funcionalidad:** Sistema completo de conteo con respaldo automático en Azure Blob Storage  
**Características principales:**

- Integración con Azure Data Lake Storage Gen2
- Transformación automática de CSV con columnas de fecha reorganizadas
- Sistema de tracking avanzado con re-identificación de personas
- Filtrado de operadores por tiempo de permanencia (>90s)
- Detección de eventos de cabina con conteo acumulativo
- Logs detallados de red para debugging de conexión Azure

**Uso:** Sistema de producción con respaldo en la nube para análisis posterior.

---

### 3. **claudio.py**, **claudio2.py**, **claudio3.py**

**Serie de evolución del sistema de conteo**

#### **claudio.py** (Versión 1)

- Sistema básico de conteo con filtrado por tiempo en escena
- Detección de operadores por permanencia (>45s)
- Conteo basado en desaparición de personas de la zona
- Procesamiento de videos pregrabados
- Uso de tiempo del video (no tiempo real) para lógica de tracking

#### **claudio2.py** (Versión 2)

- Agrega detección direccional de movimiento
- Implementa línea de embarque para validación
- Sistema de re-identificación de personas perdidas temporalmente
- Buffer de tracking con predicción de posición
- Similitud de bounding boxes para reducir falsos positivos

#### **claudio3.py** (Versión 2.2 - Final)

- Incorpora zonas PAX1 y PAX2 con líneas transversales
- Sistema de cobertura porcentual para clasificación de pasajeros
- Tracking persistente con mapeo de IDs
- Visualización con transparencias configurables
- Conteo acumulativo mejorado

**Diferencias clave:** Cada versión incrementa la sofisticación del tracking y la precisión del conteo mediante validaciones adicionales y métricas más complejas.

---

### 4. **gemini.py**

**Funcionalidad:** Aplicación web minimalista para inferencia visual  
**Características principales:**

- Servidor Flask simple sin persistencia de datos
- Visualización en tiempo real con overlays de zonas
- No guarda logs ni archivos de salida
- Diseñado para pruebas rápidas de inferencia del modelo
- Streaming directo sin almacenamiento intermedio

**Uso:** Debugging visual y verificación rápida de zonas de detección.

---

### 5. **inferencia_web.py**

**Funcionalidad:** Sistema de monitoreo con grabación de video de evidencia  
**Características principales:**

- Grabación simultánea de video MP4 con anotaciones
- Dibujo de polígonos de zonas sobre frames procesados
- Sistema de filtrado por permanencia mínima (MIN_FRAMES)
- Tracking con historial de detecciones
- Logs CSV detallados por frame
- Resumen de sesión con totales

**Uso:** Generación de evidencia visual con video anotado para auditorías.

---

### 6. **main.py** y **main2.py**

**Versiones sucesivas de procesamiento headless**

#### **main.py** (Versión básica)

- Ejecución en servidor sin interfaz gráfica
- Recibe duración en minutos como argumento CLI
- Validación estricta de zonas poligonales
- Prevención de duplicados entre clases
- Logs detallados y resumen final

#### **main2.py** (Con filtro anti-ruido)

- Añade filtrado por permanencia mínima (30 frames ≈ 2s)
- Reduce falsos positivos por detecciones momentáneas
- Historial de frames por ID para confirmación
- Optimizado para ambientes con ruido de detección

**Diferencias:** main2.py es más robusto ante oclusiones temporales y false positives.

---

### 7. **oasis.py**, **oasis2.py**, **oasis3.py**

**Serie de producción con Azure (versión actualmente en uso)**

#### **oasis.py** ⭐ (Versión actual - CÓDIGO PRINCIPAL DEL PROYECTO)

**Este es el código en producción ejecutado por `ejecutar.sh`**

**Características completas:**

- Sistema de conteo acumulativo con tracking persistente
- Re-identificación avanzada de personas perdidas
- Filtrado inteligente de operadores (>90s en escena)
- Integración completa con Azure Data Lake Storage Gen2
- Transformación automática de CSV con columnas optimizadas (Fecha, hora_minuto, etc.)
- Logs detallados de conexión y transferencia Azure
- Detección de eventos de cabina con registro temporal
- Manejo robusto de reconexión RTSP
- Respaldo automático en la nube al finalizar sesión

**Ejecución:**

```bash
./ejecutar.sh  # Ejecuta oasis.py en modo daemon con logging
python3 oasis.py <minutos>  # Ejecución manual directa
```

Ver archivo principal para documentación completa del código.

#### **oasis2.py** y **oasis3.py**

Versiones de desarrollo previas al sistema actual. Mantienen estructura similar pero con ajustes en:

- Parámetros de filtrado de operadores
- Configuración de zonas de detección
- Formato de salida CSV
- Lógica de eventos de cabina

**Nota:** Se recomienda usar **SOLO `oasis.py`** (versión actual) en producción. Las versiones 2 y 3 son históricas.

---

### 8. **prueba.py**

**Funcionalidad:** Script de validación de zonas y calibración  
**Características principales:**

- Captura frames estáticos para análisis
- Visualización de polígonos sin lógica de conteo
- Guardado de imágenes con zonas dibujadas
- Útil para ajuste de coordenadas de polígonos
- Testing de conexión RTSP y carga de modelo

**Uso:** Configuración inicial y debugging de zonas de detección.

---

### 9. **test_visual.py**

**Funcionalidad:** Sistema de conteo con visualización de tracking en vivo  
**Características principales:**

- Feedback visual completo en tiempo real
- Dibujo de IDs y estados de personas (Pasajero/Operador)
- Historial de cabinas procesadas en panel lateral
- Detección de personas "esperando" filtradas por umbral
- Grabación de video con anotaciones overlay
- Logs por minuto para análisis temporal

**Uso:** Monitoreo en vivo con supervisión humana y generación de métricas.

---

### 10. **videorecorder.py**

**Funcionalidad:** Grabador de video con inferencia y anotaciones  
**Características principales:**

- Captura video mientras ejecuta inferencia YOLO
- Dibuja cajas y labels sobre detecciones confirmadas
- Sistema de filtrado por permanencia (MIN_FRAMES = 20)
- Guardado sincronizado de video MP4 y CSV
- Panel de progreso en tiempo real

**Uso:** Grabación de sesiones con evidencia visual y datos tabulares.

---

## 🔄 Comparación de Versiones por Familia

### **Familia "claudio"** (claudio.py → claudio2.py → claudio3.py)

- **Evolución:** Tracking básico → Direccional → Multi-zona con persistencia
- **Complejidad:** Creciente
- **Precisión:** ⭐⭐ → ⭐⭐⭐ → ⭐⭐⭐⭐

### **Familia "main"** (main.py → main2.py)

- **Evolución:** Validación estricta → Filtrado anti-ruido
- **Complejidad:** Media
- **Precisión:** ⭐⭐⭐ → ⭐⭐⭐⭐

### **Familia "oasis"** (oasis.py, oasis2.py, oasis3.py)

- **Evolución:** Prototipo → Iteraciones → Producción actual
- **Complejidad:** Alta
- **Precisión:** ⭐⭐⭐⭐⭐
- **Extra:** Integración Azure, Logs avanzados, Re-identificación

---

## 📊 Tabla Comparativa de Características

| Característica        | app.py | blob.py | claudio3 | gemini | main2 | oasis.py | test_visual |
| --------------------- | ------ | ------- | -------- | ------ | ----- | -------- | ----------- |
| **Interfaz Web**      | ✅     | ❌      | ❌       | ✅     | ❌    | ❌       | ❌          |
| **Azure Backup**      | ❌     | ✅      | ❌       | ❌     | ❌    | ✅       | ❌          |
| **Re-identificación** | ❌     | ✅      | ✅       | ❌     | ❌    | ✅       | ❌          |
| **Grabación Video**   | ❌     | ❌      | ✅       | ❌     | ❌    | ❌       | ✅          |
| **Multi-zona PAX**    | ❌     | ❌      | ✅       | ❌     | ❌    | ❌       | ❌          |
| **Filtro Operadores** | ✅     | ✅      | ✅       | ❌     | ❌    | ✅       | ✅          |
| **Logs CSV**          | ✅     | ✅      | ✅       | ❌     | ✅    | ✅       | ✅          |
| **Headless Mode**     | ❌     | ✅      | ✅       | ❌     | ✅    | ✅       | ❌          |

---

## 🎯 Recomendaciones de Uso

- **⭐ Producción (PRINCIPAL):** `oasis.py` + `ejecutar.sh` (sistema completo con respaldo en nube - EN USO ACTIVO)
- **Monitoreo web:** `app.py` (interfaz remota para múltiples usuarios)
- **Debugging:** `gemini.py` o `prueba.py` (validación rápida)
- **Evidencia legal:** `test_visual.py` o `videorecorder.py` (video + datos)
- **Desarrollo:** `claudio3.py` (testing de nuevas características)

> **Nota:** El sistema en producción utiliza exclusivamente `oasis.py`, el cual es ejecutado automáticamente mediante el script `ejecutar.sh` que gestiona el proceso en segundo plano con registro de logs.

---

## 📝 Notas Técnicas

### Configuraciones Comunes

- **RTSP URL:** `rtsp://usertf:Tfo.-2525@172.16.10.22:554/cam/realmonitor?channel=5&subtype=0` (OASIS EMBARQUE)
- **RTSP URL:** `rtsp://usertf:Tfo.-2525@172.16.10.22:554/cam/realmonitor?channel=20&subtype=0` (OASIS CENITAL)
- **Modelo:** `/home/aaravenatk/yolo/best.pt`
- **Confianza:** 0.6-0.7 (60-70%)
- **Clases:** 0=person, 1=teleferico

### Parámetros Críticos

- **UMBRAL_OPERADOR:** 45-90 segundos (ajustar según flujo de trabajo)
- **MIN_FRAMES:** 15-30 frames (1-2 segundos de persistencia)
- **TIEMPO_REIDENTIFICACION:** 2-10 segundos (buffer de recuperación)

---

**Última actualización:** 02/01/2026  
