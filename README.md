# rtmp-opencv-relay

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)](#requirements)

Headless RTMP video relay:

> RTMP input → OpenCV frame processing → RTMP output (FFmpeg pipes)

**RTMP input → ffmpeg rawvideo pipe → NumPy/OpenCV processing → ffmpeg RTMP output**

Includes:
- headless operation (no GUI / no `cv2.waitKey`)
- ffmpeg `stderr` draining in a background thread (prevents process blocking)
- rotating logs in `./logs/`
- optional watermark overlay
- optional **deterministic 7-segment timecode overlay** (epoch_ms) for latency measurement (**no tesseract / no OCR**)
- `examples/video_analysis.py` for **input vs output** stream analysis + **broadcaster stats** + **timecode latency** reporting

---

## Requirements

- Python 3.9+
- `ffmpeg` available in PATH (or provide `--ffmpeg /path/to/ffmpeg`)
- `ffprobe` available in PATH (usually installed with ffmpeg) for `examples/video_analysis.py`

---

## Install

This repo uses a **src/** layout.

### Option A: editable install (recommended)

```bash
pip install -e .
```

### Option B: normal install

```bash
pip install .
```

---

## Repo layout

```text
.
├── src/
│   └── rtmp_opencv/
│       ├── __init__.py
│       └── relay.py              # Ingestor + Broadcastor (FFmpeg pipes)
├── examples/
│   ├── run_relay.py              # headless relay runner (CLI)
│   └── video_analysis.py         # stream analysis + broadcaster stats + latency
├── logs/
│   └── .keep
├── LICENSE
├── NOTICE
├── pyproject.toml
└── README.md
```

---

## Usage: run the relay

Use `examples/run_relay.py`:

```bash
python examples/run_relay.py \
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

### Watermark

```bash
--watermark "Copyright 2026, Your Company."
```

### Deterministic timecode overlay (no OCR)

Enable a **7-segment epoch_ms overlay** on the output video frames:

```bash
python examples/run_relay.py \
  --in-base  "rtmp://sourcedomain.com/appname/" \
  --in-key   "streamkey" \
  --out-base "rtmp://destinationdomain.com/appname/" \
  --out-key  "streamkey" \
  --width 1280 --height 720 \
  --loglevel info \
  --timecode
```

Tuning knobs if you need a more readable overlay (recommended if the stream is heavily compressed):

```bash
--timecode-seg-len 22 \
--timecode-seg-th  5  \
--timecode-spacing 6  \
--timecode-margin  10 \
--timecode-position bl
```

Notes:

* The overlay is designed to be decoded by pixel sampling (not OCR).
* It is drawn in **white segments** on a **black background** for better survivability through H.264 compression.

---

## Logs

Rotating logs live under `./logs/`:

* `logs/process.log` – main loop status (runner script)
* `logs/ingestor.log` – ingest/read counters (FPS/RPS)
* `logs/ffmpeg.log` – ffmpeg ingest + broadcast stderr logs (drained in a background thread)

---

## Stream analysis + latency reporting

`examples/video_analysis.py` provides:

1. **Stream metadata** via `ffprobe` (input + output)
2. **Quality comparison** of sampled input/output frames:

   * MAE, PSNR (grayscale), histogram distance, brightness/contrast
   * saves sample artifacts: `in_*.png`, `out_*.png`, `diff_*.png`
3. **Broadcaster analysis** (when running in controlled test mode)
4. **Deterministic latency measurement** by decoding the 7-segment overlay from output frames (**no OCR**)

### A) Analyze live streams (quality only)

```bash
python examples/video_analysis.py \
  --in-url  "rtmp://sourcedomain.com/appname/streamkey" \
  --out-url "rtmp://destinationdomain.com/appname/streamkey" \
  --width 640 --height 360 \
  --sample-seconds 10 --sample-fps 10
```

### B) Controlled relay test (inject timecode + measure latency)

This runs a short relay with **7-segment overlay ON**, then samples output frames and computes latency stats:

```bash
python examples/video_analysis.py \
  --in-url  "rtmp://sourcedomain.com/appname/streamkey" \
  --out-url "rtmp://destinationdomain.com/appname/streamkey" \
  --test-relay --test-duration 10 \
  --expect-7seg-timecode \
  --width 640 --height 360 \
  --sample-seconds 12 --sample-fps 10
```

Optional tuning if the decoder fails to parse timecodes reliably:

```bash
python examples/video_analysis.py \
  --in-url  "rtmp://..." \
  --out-url "rtmp://..." \
  --test-relay --test-duration 10 \
  --expect-7seg-timecode \
  --timecode-seg-len 22 \
  --timecode-seg-th  5  \
  --timecode-spacing 6  \
  --video-bitrate 4M
```

### What the latency means

Latency is computed as:

> `latency_ms = now_epoch_ms - embedded_epoch_ms`

So it includes:

* network + buffering
* encoder/decoder latency
* any RTMP server/CDN buffering

If the relay and analysis run on different machines with clock skew, the computed latency will include that offset. For best results:

* run analysis on the same machine as the relay, or
* ensure stable NTP on both machines.

---

## Encoding notes

* Output video is encoded using `libx264` with `ultrafast` + `zerolatency`.
* Audio is mapped as optional (`1:a:0?`) so the relay continues even if input audio is missing.
* The ingestor forces scaling to `--width/--height` to keep raw frame size stable.

---

## TO-DO

* [ ] `Dockerfile` + `docker-compose.yml` for a clean "runs anywhere" deployment (ffmpeg + Python + headless OpenCV).
* [ ] Add a "health" CLI to validate ffmpeg/ffprobe availability and RTMP URL connectivity.
* [ ] Add CI smoke checks + basic unit tests for 7-seg decode and ffprobe parsing.
