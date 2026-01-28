import subprocess as sp
import numpy as np

import datetime
import time
import os
import cv2
import logging
import threading
from logging.handlers import TimedRotatingFileHandler as trfh

# --------------------
# Logging
# --------------------
os.makedirs("logs", exist_ok=True)

ingestor_logger = logging.getLogger("ingestor")
ingestor_logger.setLevel(logging.INFO)

if not ingestor_logger.handlers:
    ingestor_formatter = logging.Formatter(
        "%(asctime)s:%(levelname)s:%(name)s:%(funcName)s:%(message)s"
    )
    ingestor_handler = trfh("logs/ingestor.log", when="m", interval=1, backupCount=10)
    ingestor_handler.setFormatter(ingestor_formatter)
    ingestor_logger.addHandler(ingestor_handler)

ffmpeg_logger = logging.getLogger("ffmpeg")
ffmpeg_logger.setLevel(logging.INFO)

if not ffmpeg_logger.handlers:
    ffmpeg_formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(name)s:%(message)s")
    ffmpeg_handler = trfh("logs/ffmpeg.log", when="m", interval=1, backupCount=10)
    ffmpeg_handler.setFormatter(ffmpeg_formatter)
    ffmpeg_logger.addHandler(ffmpeg_handler)


def _start_ffmpeg_stderr_logger(proc, logger, prefix: str):
    """
    Drain ffmpeg stderr in a background thread so the process cannot block
    when stderr pipe buffers fill.

    NOTE: stderr is read as bytes (since we don't use text=True).
    """
    if proc is None or proc.stderr is None:
        return None

    def _run():
        try:
            while True:
                line = proc.stderr.readline()
                if not line:
                    break
                try:
                    s = line.decode("utf-8", errors="replace").rstrip()
                except Exception:
                    s = repr(line)
                logger.info("%s %s", prefix, s)
        except Exception as e:
            logger.exception("%s stderr logger crashed: %s", prefix, e)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def _epoch_ms() -> int:
    return int(time.time() * 1000)


# --------------------
# 7-seg overlay helpers
# --------------------
# Segment order: a b c d e f g
# a: top, b: upper-right, c: lower-right, d: bottom, e: lower-left, f: upper-left, g: middle
_DIGIT_SEGMENTS = {
    0: (1, 1, 1, 1, 1, 1, 0),
    1: (0, 1, 1, 0, 0, 0, 0),
    2: (1, 1, 0, 1, 1, 0, 1),
    3: (1, 1, 1, 1, 0, 0, 1),
    4: (0, 1, 1, 0, 0, 1, 1),
    5: (1, 0, 1, 1, 0, 1, 1),
    6: (1, 0, 1, 1, 1, 1, 1),
    7: (1, 1, 1, 0, 0, 0, 0),
    8: (1, 1, 1, 1, 1, 1, 1),
    9: (1, 1, 1, 1, 0, 1, 1),
}


def _draw_rect(img: np.ndarray, x: int, y: int, w: int, h: int, color=(255, 255, 255)):
    cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), color, thickness=-1)


def _draw_7seg_digit(
    img: np.ndarray,
    digit: int,
    x: int,
    y: int,
    seg_len: int,
    seg_th: int,
    gap: int,
    color=(255, 255, 255),
):
    """
    Draw one digit inside a fixed bounding box:
      width  = seg_len + 2*seg_th
      height = 2*seg_len + 3*seg_th
    """
    d = int(digit) % 10
    a, b, c, dseg, e, f, g = _DIGIT_SEGMENTS[d]

    # a (top)
    if a:
        _draw_rect(img, x + seg_th, y, seg_len, seg_th, color)
    # g (middle)
    if g:
        _draw_rect(img, x + seg_th, y + seg_len + seg_th, seg_len, seg_th, color)
    # d (bottom)
    if dseg:
        _draw_rect(img, x + seg_th, y + 2 * seg_len + 2 * seg_th, seg_len, seg_th, color)

    # f (upper-left)
    if f:
        _draw_rect(img, x, y + seg_th, seg_th, seg_len, color)
    # b (upper-right)
    if b:
        _draw_rect(img, x + seg_th + seg_len, y + seg_th, seg_th, seg_len, color)

    # e (lower-left)
    if e:
        _draw_rect(img, x, y + 2 * seg_th + seg_len, seg_th, seg_len, color)
    # c (lower-right)
    if c:
        _draw_rect(img, x + seg_th + seg_len, y + 2 * seg_th + seg_len, seg_th, seg_len, color)

    w = seg_len + 2 * seg_th
    h = 2 * seg_len + 3 * seg_th
    return w, h


def _draw_7seg_number(
    img: np.ndarray,
    number_str: str,
    x: int,
    y: int,
    seg_len: int,
    seg_th: int,
    digit_spacing: int,
    color=(255, 255, 255),
):
    digit_w = seg_len + 2 * seg_th
    digit_h = 2 * seg_len + 3 * seg_th

    cursor = x
    for ch in number_str:
        if ch < "0" or ch > "9":
            continue
        _draw_7seg_digit(img, int(ch), cursor, y, seg_len, seg_th, gap=0, color=color)
        cursor += digit_w + digit_spacing

    total_w = cursor - x - digit_spacing if number_str else 0
    return total_w, digit_w, digit_h


class Ingestor:
    """
    Ingestor: rtmp://domain/app/streamkey -> raw bgr24 frames via stdout pipe.

    Enforces scale to (width,height) so read size is stable.

    Optional overlays:
      - watermark_text
      - timecode_7seg_overlay: draws epoch_ms as 13 digits in a fixed ROI.
    """

    def __init__(
        self,
        source,
        streamkey,
        width,
        height,
        ffmpeg_bin="ffmpeg",
        loglevel="info",
        watermark_text="",
        scale_algo="bilinear",
        timecode_7seg_overlay: bool = False,
        timecode_position: str = "bl",  # "bl" bottom-left or "tl" top-left
        timecode_seg_len: int = 18,
        timecode_seg_th: int = 4,
        timecode_digit_spacing: int = 6,
        timecode_margin_px: int = 10,
        timecode_bg: bool = True,
        timecode_bg_padding: int = 8,
    ):
        self.source = source
        self.streamkey = streamkey
        self.width = int(width)
        self.height = int(height)
        self.bin = ffmpeg_bin
        self.loglevel = loglevel
        self.watermark_text = watermark_text
        self.scale_algo = scale_algo

        self.timecode_7seg_overlay = bool(timecode_7seg_overlay)
        self.timecode_position = timecode_position
        self.timecode_seg_len = int(timecode_seg_len)
        self.timecode_seg_th = int(timecode_seg_th)
        self.timecode_digit_spacing = int(timecode_digit_spacing)
        self.timecode_margin_px = int(timecode_margin_px)
        self.timecode_bg = bool(timecode_bg)
        self.timecode_bg_padding = int(timecode_bg_padding)

        self._address = self.source + self.streamkey
        self.frame_size = self.width * self.height * 3

        vf = f"scale={self.width}:{self.height}:flags={self.scale_algo}"

        self._cmdx = [
            self.bin,
            "-loglevel",
            self.loglevel,
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-analyzeduration",
            "0",
            "-probesize",
            "32",
            "-i",
            self._address,
            "-an",
            "-vf",
            vf,
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "-vsync",
            "0",
            "-",
        ]

        self._vpipe = None
        self._stderr_thread = None

        self.stopped = False
        self._grabbed = False
        self._frame = b""

        self.start_time = 0.0
        self.total_frames = 0
        self.fps = 0.0

        self.frames_reads = 0
        self.last_frame_readat = 0.0
        self.rfps = 0.0
        self.first_readat = 0.0

    def initialize(self):
        self.start_time = time.time()
        ingestor_logger.info("STARTED_AT=%s", str(datetime.datetime.now()))
        self._vpipe = sp.Popen(
            self._cmdx,
            stdout=sp.PIPE,  # bytes
            stderr=sp.PIPE,  # bytes
            bufsize=0,
        )
        self._stderr_thread = _start_ffmpeg_stderr_logger(
            self._vpipe, ffmpeg_logger, "[ffmpeg-ingest]"
        )
        return self

    def _read_exact(self, nbytes) -> bytes:
        if self._vpipe is None or self._vpipe.stdout is None:
            return b""

        data = b""
        while len(data) < nbytes:
            chunk = self._vpipe.stdout.read(nbytes - len(data))
            if not chunk:
                break
            data += chunk
        return data

    def grab(self) -> bool:
        if self.stopped or self._vpipe is None:
            self._grabbed = False
            return False

        if self._vpipe.poll() is not None:
            self._grabbed = False
            return False

        raw = self._read_exact(self.frame_size)
        if len(raw) != self.frame_size:
            self._grabbed = False
            return False

        self._frame = raw
        self._grabbed = True

        self.total_frames += 1
        total_time = max(1e-6, time.time() - self.start_time)
        self.fps = self.total_frames / total_time
        ingestor_logger.info(
            "FPS=%s FRAMES=%s ELAPSED=%s",
            str(self.fps),
            str(self.total_frames),
            str(total_time),
        )
        return True

    def _overlay_timecode_7seg(self, vframe: np.ndarray):
        ms = _epoch_ms()
        s = f"{ms:013d}"  # stable width

        seg_len = self.timecode_seg_len
        seg_th = self.timecode_seg_th
        spacing = self.timecode_digit_spacing
        margin = self.timecode_margin_px

        digit_w = seg_len + 2 * seg_th
        digit_h = 2 * seg_len + 3 * seg_th
        total_w = 13 * digit_w + 12 * spacing
        total_h = digit_h

        if self.timecode_position == "tl":
            x = margin
            y = margin
        else:
            x = margin
            y = self.height - margin - total_h

        if self.timecode_bg:
            pad = self.timecode_bg_padding
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(self.width, x + total_w + pad)
            y2 = min(self.height, y + total_h + pad)
            cv2.rectangle(vframe, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)

        _draw_7seg_number(
            vframe,
            s,
            x=x,
            y=y,
            seg_len=seg_len,
            seg_th=seg_th,
            digit_spacing=spacing,
            color=(255, 255, 255),
        )

    def read(self):
        self.frames_reads += 1
        now = time.time()

        if self.frames_reads == 1:
            self.first_readat = now
            self.last_frame_readat = now

        vframe = np.frombuffer(self._frame, dtype=np.uint8).reshape(
            (self.height, self.width, 3)
        )

        if self.watermark_text:
            cv2.putText(
                vframe,
                self.watermark_text,
                (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        if self.timecode_7seg_overlay:
            self._overlay_timecode_7seg(vframe)

        elapsed_last_read = max(1e-6, now - self.last_frame_readat)
        rps = 1.0 / elapsed_last_read
        self.rfps = self.frames_reads / max(1e-6, now - self.first_readat)
        self.last_frame_readat = now

        ingestor_logger.info(
            "RPS=%s FPS=%s READS=%s DT=%s",
            str(rps),
            str(self.rfps),
            str(self.frames_reads),
            str(elapsed_last_read),
        )
        return vframe

    def stop(self):
        self.stopped = True
        self._cleanup()

    def _cleanup(self):
        if self._vpipe is None:
            return

        try:
            self._vpipe.terminate()
        except Exception:
            pass

        try:
            self._vpipe.wait(timeout=2)
        except Exception:
            try:
                self._vpipe.kill()
            except Exception:
                pass

        try:
            if self._vpipe.stdout:
                self._vpipe.stdout.close()
        except Exception:
            pass

        try:
            if self._vpipe.stderr:
                self._vpipe.stderr.close()
        except Exception:
            pass

        self._vpipe = None
        ingestor_logger.info("STOPPED_AT=%s", str(datetime.datetime.now()))


class Broadcastor:
    """
    Broadcastor: ndarray (H,W,3) BGR -> rtmp://destination/app/streamkey
    Optionally maps audio from sourceurl (optional map prevents failure if no audio).
    """

    def __init__(
        self,
        destination,
        streamkey,
        width,
        height,
        sourceurl,
        ffmpeg_bin="ffmpeg",
        loglevel="info",
        audiodelay=2.00,
        video_bitrate="2M",
        audio_copy=True,
        gop=30,
    ):
        self.destination = destination
        self.streamkey = streamkey
        self.width = int(width)
        self.height = int(height)
        self.bin = ffmpeg_bin
        self.loglevel = loglevel

        self.address = self.destination + self.streamkey
        self.videosize = f"{self.width}x{self.height}"
        self.audiodelay = float(audiodelay)
        self.sourceurl = sourceurl

        self._vpipe = None
        self._stderr_thread = None
        self.stopped = False

        self.cmdx = [
            self.bin,
            "-loglevel",
            self.loglevel,
            "-y",
            "-thread_queue_size",
            "4096",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-video_size",
            self.videosize,
            "-i",
            "-",
            "-itsoffset",
            str(self.audiodelay),
            "-i",
            self.sourceurl,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(int(gop)),
            "-keyint_min",
            "2",
            "-sc_threshold",
            "0",
            "-b:v",
            str(video_bitrate),
            "-maxrate",
            str(video_bitrate),
            "-bufsize",
            "1M",
        ]

        if audio_copy:
            self.cmdx += ["-c:a", "copy"]
        else:
            self.cmdx += ["-c:a", "aac", "-b:a", "128k"]

        self.cmdx += [
            "-f",
            "flv",
            "-use_wallclock_as_timestamps",
            "1",
            self.address,
        ]

        self.frames_pushed = 0
        self.write_failures = 0
        self.first_push_at = None
        self.last_push_at = None

    def initialize(self):
        self._vpipe = sp.Popen(
            self.cmdx,
            stdin=sp.PIPE,   # bytes
            stderr=sp.PIPE,  # bytes
            bufsize=0,
        )
        self._stderr_thread = _start_ffmpeg_stderr_logger(
            self._vpipe, ffmpeg_logger, "[ffmpeg-broadcast]"
        )

        sframe = np.zeros((self.height, self.width, 3), np.uint8)
        self.write(sframe)
        return self

    def is_alive(self) -> bool:
        return self._vpipe is not None and self._vpipe.poll() is None

    def write(self, frame) -> bool:
        if self.stopped or self._vpipe is None or self._vpipe.stdin is None:
            self.write_failures += 1
            return False
        if frame is None or frame.size == 0:
            self.write_failures += 1
            return False

        try:
            self._vpipe.stdin.write(frame.tobytes())
            now = time.time()
            self.frames_pushed += 1
            if self.first_push_at is None:
                self.first_push_at = now
            self.last_push_at = now
            return True
        except BrokenPipeError:
            self.write_failures += 1
            self.stop()
            return False
        except Exception:
            self.write_failures += 1
            self.stop()
            return False

    def pushed_fps(self) -> float:
        if self.first_push_at is None or self.last_push_at is None:
            return 0.0
        dt = max(1e-6, self.last_push_at - self.first_push_at)
        return float(self.frames_pushed / dt)

    def stop(self):
        self.stopped = True
        self._cleanup()

    def _cleanup(self):
        if self._vpipe is None:
            return

        try:
            if self._vpipe.stdin:
                self._vpipe.stdin.close()
        except Exception:
            pass

        try:
            self._vpipe.terminate()
        except Exception:
            pass

        try:
            self._vpipe.wait(timeout=2)
        except Exception:
            try:
                self._vpipe.kill()
            except Exception:
                pass

        try:
            if self._vpipe.stderr:
                self._vpipe.stderr.close()
        except Exception:
            pass

        self._vpipe = None
