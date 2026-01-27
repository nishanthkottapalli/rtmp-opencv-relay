# rtmp-opencv-relay

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)](#requirements)

Headless RTMP video relay:
>  RTMP input → OpenCV processing → RTMP output (FFmpeg pipes)
 
**RTMP input → ffmpeg rawvideo pipe → NumPy/OpenCV frame processing → ffmpeg RTMP output**

Includes:
- headless operation (no GUI / no `cv2.waitKey`)
- ffmpeg `stderr` draining in a background thread (prevents blocking)
- rotating logs in `./logs/`

## Requirements

- Python 3.9+
- `ffmpeg` available in PATH (or provide `--ffmpeg /path/to/ffmpeg`)

Install Python deps:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python run_relay.py \
  --in-base  "rtmp://sourcedomain.com/appname/" \
  --in-key   "streamkey" \
  --out-base "rtmp://destinationdomain.com/appname/" \
  --out-key  "streamkey" \
  --width 1280 --height 720 \
  --audiodelay 2.0 \
  --loglevel info
```

### Audio source

By default, audio is taken from the full input URL (`--in-base + --in-key`).
Override it with:

```bash
--audio-url "rtmp://sourcedomain.com/appname/streamkey"
```

### Logs

* `logs/process.log` – main loop status
* `logs/ingestor.log` – FPS/RPS counters
* `logs/ffmpeg.log` – ffmpeg ingest + broadcast logs

## Notes

* The output video is encoded using `libx264` with `ultrafast` + `zerolatency`.
* Audio is mapped as optional (`1:a:0?`) so the relay still runs if the input has no audio.
* Watermark text can be changed via `--watermark`.

---

## Quick start command

```bash
pip install -r requirements.txt

python run_relay.py \
  --in-base  "rtmp://sourcedomain.com/appname/" \
  --in-key   "streamkey" \
  --out-base "rtmp://destinationdomain.com/appname/" \
  --out-key  "streamkey" \
  --width 1280 --height 720 \
  --audiodelay 2.0 \
  --loglevel info
```

### TO-DO
[ ] `Dockerfile` + `docker-compose.yml` for a clean "runs anywhere" deployment (ffmpeg + Python + headless OpenCV).
