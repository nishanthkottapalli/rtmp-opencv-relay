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
