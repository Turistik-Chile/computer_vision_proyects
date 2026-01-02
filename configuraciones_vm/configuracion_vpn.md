Guía de Configuración: Azure VM como Gateway VPN L2TP/IPsec para RTSP
Esta guía detalla paso a paso cómo configurar una Máquina Virtual en Azure (Ubuntu 20.04/22.04/24.04) para conectarse como cliente a una VPN corporativa y permitir el acceso a una cámara RTSP interna.

# Fase 0: Configurar en Azure las politicas de entrada del Grupo de seguridad de red.

Crear reglas de entrada siguientes:
Origen: Any, Destino: Any Servicio: Custom, Protocolo: UDP, Acción: Permitir.
Intervalos de puertos (crear una regla por cada puerto): 500, 4500 y 1701
Origen: Any, Destino: Any Servicio: Custom, Protocolo: TCP, Acción: Permitir.
RTSP: puerto destino 554

#Fase 1: Actualizar y preparar módulos

1. Ejecuta los siguientes comandos como root (sudo su) o con sudo:

```bash
# 1. Actualizar repositorios
sudo apt-get update

# 2. Instalar módulos extra del kernel (CRÍTICO: Soluciona el error "L2TP kernel support not detected")
sudo apt-get install -y linux-modules-extra-$(uname -r)

# 3. Cargar el módulo manualmente para verificar
sudo modprobe l2tp_ppp

# 4. Instalar paquetes de software necesarios
sudo apt-get install -y strongswan xl2tpd ppp libstrongswan-extra-plugins ffmpeg net-tools

```

2. Configurar Firewall Interno

```Bash
sudo ufw allow 500/udp
sudo ufw allow 4500/udp
sudo ufw allow 1701/udp
sudo ufw allow 554/tcp
```

# Fase 2: Configuración de los Archivos de VPN

Debes editar 4 archivos clave. Reemplaza los valores en MAYÚSCULAS con tus datos reales.

1. /etc/ipsec.conf (Capa de Seguridad)

`sudo nano /etc/ipsec.conf`

```Bash
config setup
    charondebug="ike 1, knl 1, cfg 0"
    uniqueids=no

conn vpn-empresa
    auto=add
    keyexchange=ikev1
    authby=secret
    type=transport
    left=%defaultroute
    leftprotoport=17/1701
    right=IP_PUBLICA_VPN_EMPRESA
    rightprotoport=17/1701
    ike=aes256-sha1-modp1024,3des-sha1-modp1024!
    esp=aes256-sha1,3des-sha1!
```

2. /etc/ipsec.secrets (Clave Precompartida)

`sudo nano /etc/ipsec.secrets`

```bash
IP_PRIVADA_VM IP_PUBLICA_VPN_EMPRESA : PSK "TU_CLAVE_PRECOMPARTIDA"
```

3. /etc/xl2tpd/xl2tpd.conf (Túnel L2TP)

```bash
sudo nano /etc/xl2tpd/xl2tpd.conf
```

```bash
[lac vpn-connection]
lns = IP_PUBLICA_VPN_EMPRESA
ppp debug = yes
pppoptfile = /etc/ppp/options.l2tpd.client
length bit = yes
# CRÍTICO: Conecta automáticamente al iniciar el servicio
autodial = yes
```

4. /etc/ppp/options.l2tpd.client (Credenciales PPP)

```bash
sudo nano /etc/ppp/options.l2tpd.client
```

```bash
ipcp-accept-local
ipcp-accept-remote
refuse-eap
require-chap
noccp
noauth
mtu 1280
mru 1280
noipdefault
defaultroute
usepeerdns
connect-delay 5000
name "TU_USUARIO_VPN"
password "TU_CONTRASEÑA_VPN"
```

# Fase 3: Enrutamiento Automático (Split Tunneling)

Para que la VM sepa enviar el tráfico de la cámara por la VPN (y no por internet), creamos un script que se ejecuta al conectar. Este script es dinámico y funciona sin importar si la interfaz es ppp0 o ppp1.

1. Crear el script

```bash
sudo nano /etc/ppp/ip-up.d/rutas-vpn
```

2. Contenido del Script

Copia esto exactamente (incluyendo la primera línea #!/bin/sh):

```bash
#!/bin/sh

# Verifica que exista una interfaz PPP activa
if [ -n "$PPP_IFACE" ]; then
    # Ruta a la red de la Cámara (172.16.10.x)
    route add -net 172.16.10.0 netmask 255.255.255.0 dev $PPP_IFACE

    # Ruta a la red de infraestructura VPN (10.10.20.x)
    route add -net 10.10.20.0 netmask 255.255.255.0 dev $PPP_IFACE
fi
```

3. Permisos de Ejecución (OBLIGATORIO)

```Bash
sudo chmod +x /etc/ppp/ip-up.d/rutas-vpn
```

# Fase 4: Secuencia de Inicio

Para asegurar una conexión limpia, sigue este orden.

```Bash
# 1. Limpiar procesos antiguos (por seguridad)
sudo killall pppd
sudo killall xl2tpd

# 2. Iniciar IPsec (Capa 1)
sudo ipsec restart
sleep 3
sudo ipsec up vpn-empresa
# Esperar mensaje: "established successfully"

# 3. Iniciar L2TP (Capa 2 - Con autodial)
sudo systemctl restart xl2tpd
```

Verificación
Espera 10 segundos y ejecuta:

```Bash
ifconfig      # Debe aparecer la interfaz ppp0 (o ppp1)
ip route      # Debe aparecer la línea: 172.16.10.0/24 dev ppp...
```

# Fase 5: Pruebas de Conexión y Video

1. Prueba de Ping

```Bash
ping -c 4 172.16.10.22
```

2. Prueba de Video (FFprobe)
   Verifica que el stream llega correctamente usando TCP.

```Bash
ffprobe -v error -show_streams -rtsp_transport tcp "rtsp://usertf:Tfo.-2525@172.16.10.22:554/cam/realmonitor?channel=5&subtype=0"
```

(Si ves metadatos del video como h264, width, height, la conexión interna es un éxito).

# Fase 5: Automatización de inicio.

Para evitar tener que levantar la VPN manualmente cada vez que la VM se reinicia, configuramos un servicio de sistema que maneja la secuencia de arranque (Limpieza -> IPsec -> L2TP).

1. Crear el Script de Arranque
   Este script se encarga de la lógica de conexión.

```bash
sudo bash -c 'cat > /usr/local/bin/vpn-boot.sh <<EOF
#!/bin/bash
LOG_FILE="/var/log/vpn-boot.log"

echo ">>> [\$(date)] Iniciando arranque VPN..." > \$LOG_FILE

# 1. Limpieza de procesos antiguos
killall -q pppd
killall -q xl2tpd

# 2. Reiniciar IPsec
ipsec restart
sleep 5

# 3. Levantar conexión IPsec
OUTPUT=\$(ipsec up vpn-empresa 2>&1)
echo "\$OUTPUT" >> \$LOG_FILE

if echo "\$OUTPUT" | grep -q "established successfully"; then
    echo ">>> EXITO: IPsec conectado." >> \$LOG_FILE

    # 4. Iniciar L2TP (El script de rutas se ejecutará solo por el hook de pppd)
    systemctl restart xl2tpd
    sleep 5

    # 5. Verificación final
    if ip addr | grep -q "ppp"; then
         echo ">>> VICTORIA: Interfaz PPP activa." >> \$LOG_FILE
         exit 0
    else
         echo ">>> ERROR: IPsec ok, pero no hay interfaz PPP." >> \$LOG_FILE
         exit 1
    fi
else
    echo ">>> ERROR FATAL: IPsec no conecto." >> \$LOG_FILE
    exit 1
fi
EOF'
```

Dar permisos de ejecución:

```bash
sudo chmod +x /usr/local/bin/vpn-boot.sh
```

2. Crear el Servicio de Systemd
   Este archivo le dice a Linux que ejecute nuestro script al inicio, pero solo después de que tenga internet.

```bash
sudo bash -c 'cat > /etc/systemd/system/vpn-start.service <<EOF
[Unit]
Description=Script de Auto-Conexion VPN (Bash Version)
# Esperar a que la red esté totalmente lista
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
# Inyectamos el PATH completo para evitar errores de "command not found"
Environment="PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
# Ejecutamos con Bash explícitamente
ExecStart=/bin/bash /usr/local/bin/vpn-boot.sh
RemainAfterExit=yes
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF'
```

3. Activar la Automatización
   Finalmente, registramos el servicio y lo probamos.

```bash
# Recargar la base de datos de servicios
sudo systemctl daemon-reload

# Habilitar para que arranque en el próximo reinicio
sudo systemctl enable vpn-start.service

# Iniciar ahora mismo para probar
sudo systemctl start vpn-start.service
```

4.  Verificación
    Para confirmar que todo está automatizado:

        Reinicia la VM: sudo reboot

        Espera 1 minuto.

        Entra y ejecuta: ifconfig (Debería aparecer ppp0 automáticamente).

        Si algo falla, revisa el log: cat /var/log/vpn-boot.log.
