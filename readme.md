# Dashcam YOLO Home Lab
### Intel Mac Mini · Unraid · Label Studio · YOLOv8n · CPU-only

---

## Table of Contents
1. [Project Structure](#1-project-structure)
2. [Prerequisites](#2-prerequisites)
3. [First-Time Setup](#3-first-time-setup)
4. [Understanding the Docker Compose File](#4-understanding-the-docker-compose-file)
5. [Understanding the Dockerfile](#5-understanding-the-dockerfile)
6. [Starting the Stack](#6-starting-the-stack)
7. [Daily Dashcam Workflow](#7-daily-dashcam-workflow)
8. [Accessing Services](#8-accessing-services)
9. [Running Scripts On-Demand](#9-running-scripts-on-demand)
10. [Label Studio ML Backend](#10-label-studio-ml-backend)
11. [Troubleshooting](#11-troubleshooting)
12. [RAM & Resource Management](#12-ram--resource-management)

---

## 1. Project Structure

```
/mnt/user/appdata/yolo/          ← this repo lives here
├── docker-compose.yml
├── yolo/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app/
│   │   └── main.py              ← FastAPI YOLO server
│   └── scripts/
│       ├── extract_frames.sh
│       ├── highlight_detector.py
│       └── select_diverse_frames.py
└── nginx/
    ├── nginx.conf
    └── html/
        └── index.html           ← simple dashboard landing page

/mnt/cache/
├── labelstudio/                 ← Label Studio database & media
├── yolo/
│   ├── models/                  ← trained model weights
│   └── scripts/                 ← shared scripts
├── dashcam/
│   ├── raw/                     ← paste new footage files here
│   ├── frames/                  ← extracted JPGs land here
│   └── clips/                   ← tagged highlight clips land here
└── datasets/                    ← training data lives here
    └── dashcam/
        ├── images/
        │   ├── train/
        │   └── val/
        ├── labels/
        │   ├── train/
        │   └── val/
        └── data.yaml
```

> 💡 Keep footage and code in **separate Unraid shares**. This makes it easy to expand storage later without touching the appdata share.

---

## 2. Prerequisites

Before running anything, confirm these are in place on your Unraid server:

- [ ] Unraid 6.11+ installed and booted
- [ ] Docker enabled: **Settings → Docker → Enable Docker → Yes**
- [ ] The following cache directories created:
  - `/mnt/cache/labelstudio`
  - `/mnt/cache/yolo/models`
  - `/mnt/cache/yolo/scripts`
  - `/mnt/cache/dashcam/raw`, `frames`, `clips`
  - `/mnt/cache/datasets`
- [ ] Ports available: `8081` (nginx), `8082` (Label Studio), `8000` (YOLO API)
  - Port `8080` and `443` are reserved for Unraid
- [ ] Community Applications plugin installed (optional but recommended)
- [ ] Compose Manager plugin installed (lets you manage this stack from the Unraid GUI)

---

## 3. First-Time Setup

### Step 1 — Create the directory structure

SSH into your Unraid server and run:

```bash
mkdir -p /mnt/cache/labelstudio
mkdir -p /mnt/cache/yolo/models
mkdir -p /mnt/cache/yolo/scripts
mkdir -p /mnt/cache/dashcam/raw
mkdir -p /mnt/cache/dashcam/frames
mkdir -p /mnt/cache/dashcam/clips
mkdir -p /mnt/cache/datasets
mkdir -p /mnt/user/appdata/yolo/nginx/html
mkdir -p /mnt/user/appdata/yolo/yolo/app
mkdir -p /mnt/user/appdata/yolo/yolo/scripts
```

### Step 2 — Clone or copy this project

```bash
cd /mnt/user/appdata/yolo
git clone <your-repo-url> .
```

### Step 3 — requirements.txt

```
ultralytics==8.3.0
fastapi==0.115.0
uvicorn==0.30.0
pillow==10.4.0
python-multipart==0.0.12
opencv-python-headless==4.10.0.84
label-studio-ml==1.0.9
numpy==1.26.4
groundingdino-py==0.4.0
supervision==0.22.0
```

> ⚠️ **Why pin versions?** Ultralytics releases frequently and occasionally introduces breaking API changes. Pinned versions ensure your stack doesn't silently break after a `docker compose pull`.

### Step 4 — docker-compose.yml

```yaml
volumes:
  labelstudio_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /mnt/cache/labelstudio
  yolo_models:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /mnt/cache/yolo/models
  yolo_scripts:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /mnt/cache/yolo/scripts
  dashcam_raw:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /mnt/cache/dashcam/raw
  dashcam_frames:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /mnt/cache/dashcam/frames
  dashcam_clips:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /mnt/cache/dashcam/clips
  datasets:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /mnt/cache/datasets

networks:
  homelab:
    driver: bridge

services:
  label-studio:
    image: heartexlabs/label-studio:latest
    container_name: label-studio
    networks:
      - homelab
    ports:
      - "8082:8080"        # 8080 reserved for Unraid
    volumes:
      - labelstudio_data:/label-studio/data
      - datasets:/ls-files
    environment:
      - LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
      - LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/ls-files
      - LABEL_STUDIO_HOST=http://10.10.80.21:8082
      - LABEL_STUDIO_BASE_URL=/
    mem_limit: 3g
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 60s

  yolo-api:
    build:
      context: ./yolo
      dockerfile: Dockerfile
      shm_size: "512mb"
    container_name: yolo-api
    networks:
      - homelab
    ports:
      - "8000:8000"
    volumes:
      - yolo_models:/app/models
      - yolo_scripts:/app/scripts
      - datasets:/app/datasets
      - dashcam_frames:/app/frames
      - dashcam_clips:/app/clips
    environment:
      - OMP_NUM_THREADS=2
      - MKL_NUM_THREADS=2
      - PYTHONUNBUFFERED=1
    mem_limit: 6g
    restart: unless-stopped
    depends_on:
      label-studio:
        condition: service_healthy

  ffmpeg-worker:
    image: jrottenberg/ffmpeg:4.4-alpine
    container_name: ffmpeg-worker
    networks:
      - homelab
    volumes:
      - dashcam_raw:/dashcam/raw
      - dashcam_frames:/dashcam/frames
      - dashcam_clips:/dashcam/clips
      - yolo_scripts:/scripts
    entrypoint: ["tail", "-f", "/dev/null"]
    mem_limit: 1g
    restart: unless-stopped

  highlight-worker:
    build:
      context: ./yolo
      dockerfile: Dockerfile
    container_name: highlight-worker
    networks:
      - homelab
    volumes:
      - dashcam_raw:/app/raw
      - dashcam_frames:/app/frames
      - dashcam_clips:/app/clips
      - yolo_models:/app/models
      - yolo_scripts:/app/scripts
    environment:
      - OMP_NUM_THREADS=2
      - PYTHONUNBUFFERED=1
    command: ["tail", "-f", "/dev/null"]
    mem_limit: 4g
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    container_name: nginx-proxy
    networks:
      - homelab
    ports:
      - "8081:8081"        # 80/443 reserved for Unraid
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/html:/usr/share/nginx/html:ro
    mem_limit: 256m
    restart: unless-stopped
    depends_on:
      - yolo-api
      - label-studio
```

### Step 5 — nginx.conf

```nginx
worker_processes 1;

events {
    worker_connections 1024;
}

http {
    resolver 127.0.0.11 valid=30s;

    server {
        listen 8081;

        location /label-studio/ {
            proxy_pass http://label-studio:8080/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_read_timeout 300;
            proxy_connect_timeout 300;
        }

        location /yolo/ {
            set $yolo_upstream http://yolo-api:8000;
            proxy_pass $yolo_upstream/;
            proxy_set_header Host $host;
        }

        location / {
            root /usr/share/nginx/html;
            index index.html;
        }
    }
}
```

---

## 4. Understanding the Docker Compose File

### Port Allocation

| Port | Service | Reason |
|---|---|---|
| `8080` | Unraid OS | Reserved — do not use |
| `443` | Unraid OS | Reserved — do not use |
| `8081` | nginx proxy | Dashboard & reverse proxy |
| `8082` | Label Studio | Direct access |
| `8000` | YOLO API | FastAPI + Swagger |

### Named Volumes vs Bind Mounts

All volumes are bind mounts to `/mnt/cache/`. This means:
- Data physically lives on your Unraid SSD
- Files are visible in Unraid's file manager
- Rebuilding containers never deletes your data

### Memory Limits

```yaml
mem_limit: 6g   # yolo-api
mem_limit: 3g   # label-studio
mem_limit: 1g   # ffmpeg-worker
mem_limit: 4g   # highlight-worker
mem_limit: 256m # nginx
```

> ⚠️ Never run YOLO training and highlight detection simultaneously. Together they can peak at 8–9GB leaving insufficient headroom for Label Studio and the OS.

### The `shm_size` Setting

PyTorch's DataLoader uses shared memory (`/dev/shm`) to pass data between worker processes. The default on Linux is only 64MB. With a dataset of even a few hundred images, PyTorch will crash with a `Bus error`. This setting raises the limit to 512MB, sufficient for `batch=8` training on CPU.

---

## 5. Understanding the Dockerfile

### Multi-Stage Build

```dockerfile
FROM python:3.11-slim AS builder
# install build tools, compile packages

FROM python:3.11-slim
COPY --from=builder /root/.local /root/.local
# no build tools in final image
```

gcc and g++ add ~400MB to an image. Multi-stage builds let you use them and discard them. The final image is roughly half the size.

### Layer Caching

```dockerfile
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt
COPY app/ ./app/
```

Copy `requirements.txt` before app code so Docker skips the pip install on most rebuilds.

### Pre-downloading Model Weights

```dockerfile
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

Bakes `yolov8n.pt` into the image at build time so the first inference call is instant and works in air-gapped environments.

### Non-Root User

```dockerfile
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    chown -R appuser:appuser /root/.local
USER appuser
CMD ["python3.11", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

> ⚠️ Use `python3.11 -m uvicorn` instead of calling the uvicorn binary directly to avoid permission errors in multi-stage builds.

---

## 6. Starting the Stack

```bash
cd /mnt/user/appdata/yolo

# Build images and start all services
docker compose up -d --build --remove-orphans

# Watch the logs during first boot
docker compose logs -f

# Check all containers are healthy
docker compose ps
```

Expected output after ~60 seconds:
```
NAME               STATUS              PORTS
label-studio       Up (healthy)        0.0.0.0:8082->8080/tcp
yolo-api           Up                  0.0.0.0:8000->8000/tcp
ffmpeg-worker      Up
highlight-worker   Up
nginx-proxy        Up                  0.0.0.0:8081->8081/tcp
```

---

## 7. Daily Dashcam Workflow

### Step 1 — Copy footage from your dashcam SD card
```bash
cp /path/to/sd/card/*.MP4 /mnt/cache/dashcam/raw/
```

### Step 2 — Extract frames
```bash
docker exec ffmpeg-worker sh /scripts/extract_frames.sh footage.avi run_$(date +%Y%m%d) 1
```

### Step 3 — Run highlight detection
```bash
docker exec highlight-worker python /app/scripts/highlight_detector.py \
  --input /app/raw/footage.avi \
  --output /app/clips
```

### Step 4 — Select diverse frames for annotation
```bash
docker exec highlight-worker python /app/scripts/select_diverse_frames.py \
  --input /app/frames/run_$(date +%Y%m%d) \
  --output /app/datasets/dashcam/images/train \
  --count 500
```

### Step 5 — Annotate in Label Studio
1. Open `http://10.10.80.21:8082`
2. Open your project → Import → Local Files → point to `/ls-files/dashcam/images/train`
3. Annotate with bounding boxes
4. Export → YOLO format → save to `/mnt/cache/datasets/dashcam/labels/train`

### Step 6 — Train
```bash
docker exec yolo-api python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='/app/datasets/dashcam/data.yaml', epochs=50, imgsz=640, batch=8, device='cpu', workers=2)
"
```

### Step 7 — Clean up raw footage
```bash
rm /mnt/cache/dashcam/raw/*.avi
rm -rf /mnt/cache/dashcam/frames/run_<date>
```

---

## 8. Accessing Services

| Service | Direct URL | Via Nginx |
|---|---|---|
| Label Studio | `http://10.10.80.21:8082` | `http://10.10.80.21:8081/label-studio/` |
| YOLO API | `http://10.10.80.21:8000` | `http://10.10.80.21:8081/yolo/` |
| YOLO Swagger Docs | `http://10.10.80.21:8000/docs` | `http://10.10.80.21:8081/yolo/docs` |
| Dashboard | — | `http://10.10.80.21:8081/` |

> 💡 Set a static IP for your server in your router's DHCP settings so the address never changes.

---

## 9. Running Scripts On-Demand

```bash
# Extract frames with ffmpeg
docker exec ffmpeg-worker sh /scripts/extract_frames.sh footage.avi run_$(date +%Y%m%d) 1

# Run highlight detector
docker exec highlight-worker python /app/scripts/highlight_detector.py [args]

# Open an interactive shell for debugging
docker exec -it highlight-worker sh
docker exec -it yolo-api sh
```

---

## 10. Label Studio ML Backend

This project uses the [HumanSignal Label Studio ML Backend](https://github.com/HumanSignal/label-studio-ml-backend) with the YOLO example backend.

### ML Backend Structure

```
label_studio_ml/
└── examples/
    └── yolo/           ← YOLO auto-annotation backend
        ├── model.py    ← YOLOv8n inference logic
        └── ...
```

### How it works

1. Label Studio sends an image to the ML backend via HTTP
2. The backend runs `yolov8n.pt` inference on the image
3. Predicted bounding boxes are returned as pre-annotations
4. You review, correct, and confirm annotations in Label Studio
5. Confirmed annotations are exported as YOLO training data

### Connecting the ML Backend to Label Studio

1. Open Label Studio at `http://10.10.80.21:8082`
2. Go to **Settings → Machine Learning → Add Model**
3. Enter the ML backend URL: `http://yolo-api:8000`
4. Click **Validate and Save**

### Model: YOLOv8n (COCO)

`yolov8n.pt` is pretrained on the COCO dataset with 80 classes including:
- `person`, `car`, `truck`, `bus`, `motorcycle`, `bicycle`
- `traffic light`, `stop sign`
- Animals, common objects

This makes it ideal as a **starting point** for dashcam annotation — it will pre-label cars and people out of the box, and you fine-tune it on your specific dashcam footage over time.

---

## 11. Troubleshooting

**Container won't start / exits immediately**
```bash
docker compose logs <service-name>
```
Check for missing directories — all bind mount paths must exist before compose starts.

**Label Studio shows blank page**
- Wait 30 seconds and refresh — first boot initializes a SQLite database
- Check `LABEL_STUDIO_HOST` matches your server IP and port
- Check `/mnt/cache/labelstudio/data/.env` is not overriding your compose environment variables

**YOLO API keeps restarting**
```bash
docker logs yolo-api
```
- If you see `Permission denied` on uvicorn, ensure your Dockerfile uses `python3.11 -m uvicorn` not the binary directly
- If you see missing model weights, rebuild with `docker compose build --no-cache yolo-api`

**Port conflicts**
- Port `8080` and `443` are used by Unraid — never assign these to containers
- nginx uses `8081`, Label Studio uses `8082`, YOLO API uses `8000`

**Nginx returns 502 Bad Gateway**
The upstream service isn't ready. Run `docker compose ps` to check health status.

**ffmpeg `bash not found` error**
The ffmpeg alpine image uses `sh` not `bash`:
```bash
docker exec ffmpeg-worker sh /scripts/extract_frames.sh [args]
```

**Training crashes with "Bus error"**
Reduce `workers` to 0 in the training call or reduce `batch` size from 8 to 4.

**Out of memory during training**
Reduce `batch` size from 8 to 4 or 2. Each batch loads multiple images into RAM simultaneously.

---

## 12. RAM & Resource Management

| Service | Limit | Typical Usage |
|---|---|---|
| Unraid OS | ~1GB | ~800MB |
| label-studio | 3GB | 500MB–1.5GB |
| yolo-api (idle) | 6GB | ~800MB |
| yolo-api (training) | 6GB | 3–5GB |
| ffmpeg-worker | 1GB | ~200MB |
| highlight-worker | 4GB | 1–3GB |
| nginx | 256MB | ~20MB |
| **Total** | **~15.25GB** | **varies** |

> ⚠️ **Never run training and highlight detection at the same time.** Use a lock file:
> ```bash
> # Before training
> touch /mnt/user/appdata/yolo/.training_lock
> # After training
> rm /mnt/user/appdata/yolo/.training_lock
> ```
