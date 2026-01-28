# rtmp-opencv-relay

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)](#requirements)

Headless RTMP video relay:

> RTMP input → OpenCV processing → RTMP output (FFmpeg pipes)

**RTMP input → ffmpeg rawvideo pipe → NumPy/OpenCV frame processing → ffmpeg RTMP output**

Includes:
- headless operation (no GUI / no `cv2.waitKey`)
- ffmpeg `stderr` draining in a background thread (prevents pipe blocking)
- rotating logs in `./logs/`
- optional **7-segment timecode overlay** for deterministic latency measurement (**no tesseract / no OCR**)
- `examples/video_analysis.py` to compare input/output stream quality + relay/broadcaster stats + timecode-based latency

---

## Requirements

- Python 3.9+
- `ffmpeg` available in PATH (or provide `--ffmpeg /path/to/ffmpeg`)
- `ffprobe` available in PATH (usually comes with ffmpeg) for stream metadata in analysis

---

## Install

This repo uses a **src/** layout.

### Option A: editable install (recommended for development)

```bash
pip install -e .
```

### Option B: normal install

```bash
pip install .
```

Dependencies are:

* `numpy`
* `opencv-python-headless`

---

## Repo layout

```text
.
├── src/
│   └── rtmp_opencv/
│       └── relay.py              # Ingestor + Broadcastor (FFmpeg pipes)
├── examples/
│   ├── run_relay.py              # main relay runner (headless)
│   └── video_analysis.py         # stream analysis + deterministic latency
└── logs/
    ├── process.log
    ├── ingestor.log
    └── ffmpeg.log
```

---

## Usage: Run the relay

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

Override with:

```bash
--audio-url "rtmp://sourcedomain.com/appname/streamkey"
```

---

## Logs

Rotating logs live under `./logs/`:

* `logs/process.log` – main loop status (runner script)
* `logs/ingestor.log` – frame ingest/read counters (FPS/RPS)
* `logs/ffmpeg.log` – ffmpeg ingest + broadcast logs (stderr drained in a background thread)

---

## Video encoding notes

* Output video encoded using `libx264` with `ultrafast` + `zerolatency`.
* Audio mapping is optional (`1:a:0?`) so the relay still runs if the input has no audio.
* Watermark text can be changed via `--watermark`.

---

## Stream analysis + deterministic latency (no OCR)

`examples/video_analysis.py` provides:

1. **Stream metadata** via `ffprobe` (input + output)
2. **Quality comparison** between input/output frames (MAE, PSNR, histogram distance, brightness/contrast)
3. **Broadcaster/relay stats** when run in controlled mode
4. **Deterministic latency** using a **7-segment timecode overlay** decoded from output frames
   (no tesseract, no OCR)

### Recommended workflow

#### A) Run a controlled relay test (injects 7-seg epoch_ms overlay) + decode latency

```bash
python examples/video_analysis.py \
  --in-url  "rtmp://sourcedomain.com/appname/streamkey" \
  --out-url "rtmp://destinationdomain.com/appname/streamkey" \
  --test-relay --test-duration 10 \
  --expect-7seg-timecode \
  --width 640 --height 360 \
  --sample-seconds 12 --sample-fps 10
```

This will:

* run a relay for `--test-duration` seconds **with 7-seg overlay ON**
* sample frames from both input/output
* decode 13-digit epoch_ms from output overlay
* report latency stats (`mean`, `p50`, `p95`, etc.)
* write artifacts + `report.json` under `artifacts/video_analysis/<timestamp>/`

#### B) If timecode parsing is unstable

Increase overlay legibility (bigger segments):

```bash
python examples/video_analysis.py \
  --in-url  "rtmp://..." \
  --out-url "rtmp://..." \
  --test-relay --test-duration 10 \
  --expect-7seg-timecode \
  --timecode-seg-len 22 \
  --timecode-seg-th 5 \
  --video-bitrate 4M
```

### About the latency number

The deterministic latency uses:

> `latency_ms = now_epoch_ms - embedded_epoch_ms`

So it includes:

* network + buffering
* encoder/decoder latency
* any RTMP server/CDN buffering

If your relay and analysis run on different hosts with clock skew, the latency will include that offset. For best results:

* run analysis on the same host as the relay, or
* ensure stable NTP on both.

---

## Quick start

```bash
pip install -e .

python examples/run_relay.py \
  --in-base  "rtmp://sourcedomain.com/appname/" \
  --in-key   "streamkey" \
  --out-base "rtmp://destinationdomain.com/appname/" \
  --out-key  "streamkey" \
  --width 1280 --height 720 \
  --audiodelay 2.0 \
  --loglevel info
```

---

## TO-DO

* [ ] Add optional monotonic timecode mode + offset calibration (for multi-host analysis).
* [ ] Add “health” CLI that checks ffmpeg/ffprobe availability and validates stream URLs.
* [ ] `Dockerfile` + `docker-compose.yml` for a clean "runs anywhere" deployment (ffmpeg + Python + headless OpenCV).
