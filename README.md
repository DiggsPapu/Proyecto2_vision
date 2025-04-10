# Proyecto2_vision
Proyecto 2 de visión por computadora

Este proyecto utiliza OpenCV, MediaPipe y YOLOv8 (Ultralytics) para realizar detección de poses humanas en video, y genera dos salidas:
- Un video con la imagen original + esqueleto.
- Un video con fondo negro que muestra únicamente el esqueleto.

## 📦 Requisitos

Instala las dependencias necesarias con:

```bash
pip install -r requirements.txt
```

### `requirements.txt` incluye:
- `opencv-python`
- `mediapipe`
- `numpy`
- `ultralytics`
- `torch`
- `torchvision`

## ▶️ Cómo ejecutar

1. Asegúrate de tener Python 3.8 o superior.
2. Coloca tu video en la ruta correspondiente o actualiza la variable `video_path` dentro del script.
3. Ejecuta el script principal:

```bash
python Parte2.py
```

4. Se generarán dos archivos de salida:
   - `video_doble.mp4`: Video con el original y el esqueleto lado a lado.
   - `video_skeleton.mp4`: Video con solo el esqueleto sobre fondo negro.

## 🧠 Modelos
Este proyecto usa el modelo YOLOv8 (`yolov8n.pt`) de [Ultralytics](https://github.com/ultralytics/ultralytics). Se descargará automáticamente la primera vez que se use.

## 💻 Recomendaciones
- Si cuentas con GPU, puedes instalar una versión optimizada de `torch` desde [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) para mejorar el rendimiento.
- Asegúrate de tener acceso a una cámara o un archivo de video válido.
