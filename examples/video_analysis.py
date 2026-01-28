#!/usr/bin/env python3
import argparse
import json
import os
import subprocess as sp
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from rtmp_opencv.relay import Ingestor, Broadcastor


def _epoch_ms() -> int:
    return int(time.time() * 1000)


# -----------------------------
# ffprobe
# -----------------------------
def _run(cmd: List[str], timeout: int = 20) -> Tuple[int, str, str]:
    p = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE, text=True)
    try:
        out, err = p.communicate(timeout=timeout)
    except sp.TimeoutExpired:
        p.kill()
        out, err = p.communicate()
        return 124, out, err
    return p.returncode, out, err


def _safe_int(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def ffprobe_summary(url: str) -> Dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        url,
    ]
    rc, out, err = _run(cmd, timeout=20)
    if rc != 0 or not out.strip():
        return {"ok": False, "error": err.strip() or f"ffprobe rc={rc}", "url": url}

    try:
        data = json.loads(out)
    except Exception as e:
        return {"ok": False, "error": f"ffprobe json parse error: {e}", "url": url}

    streams = data.get("streams", [])
    fmt = data.get("format", {})

    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    def _fps(s):
        for k in ("avg_frame_rate", "r_frame_rate"):
            v = (s or {}).get(k)
            if isinstance(v, str) and "/" in v:
                a, b = v.split("/", 1)
                try:
                    a = float(a); b = float(b)
                    if b != 0:
                        return a / b
                except Exception:
                    pass
        return None

    summary = {
        "ok": True,
        "url": url,
        "format": {
            "format_name": fmt.get("format_name"),
            "duration_s": _safe_float(fmt.get("duration")),
            "bitrate_bps": _safe_int(fmt.get("bit_rate")),
        },
        "video": None,
        "audio": None,
    }

    if video:
        summary["video"] = {
            "codec": video.get("codec_name"),
            "profile": video.get("profile"),
            "width": video.get("width"),
            "height": video.get("height"),
            "pix_fmt": video.get("pix_fmt"),
            "fps": _fps(video),
            "bitrate_bps": _safe_int(video.get("bit_rate")),
        }

    if audio:
        summary["audio"] = {
            "codec": audio.get("codec_name"),
            "sample_rate": _safe_int(audio.get("sample_rate")),
            "channels": audio.get("channels"),
            "bitrate_bps": _safe_int(audio.get("bit_rate")),
        }

    return summary


# -----------------------------
# Sampling config
# -----------------------------
@dataclass
class SampleConfig:
    width: int
    height: int
    sample_seconds: float
    sample_fps: float
    warmup_seconds: float
    loglevel: str
    ffmpeg_bin: str
    watermark_text: str


def split_rtmp_url(url: str) -> Tuple[str, str]:
    url = url.rstrip("/")
    base, key = url.rsplit("/", 1)
    return base + "/", key


def sample_frames(url: str, cfg: SampleConfig, *, disable_overlays: bool = True) -> List[np.ndarray]:
    base, key = split_rtmp_url(url)
    ing = Ingestor(
        base,
        key,
        cfg.width,
        cfg.height,
        ffmpeg_bin=cfg.ffmpeg_bin,
        loglevel=cfg.loglevel,
        watermark_text=cfg.watermark_text if not disable_overlays else "",
        timecode_7seg_overlay=False,
    ).initialize()

    frames: List[np.ndarray] = []
    wanted = max(1, int(cfg.sample_seconds * cfg.sample_fps))
    period = 1.0 / max(1e-6, cfg.sample_fps)

    try:
        t0 = time.monotonic()
        while time.monotonic() - t0 < cfg.warmup_seconds:
            if not ing.grab():
                break

        for _ in range(wanted):
            t_start = time.monotonic()
            if not ing.grab():
                break
            frames.append(ing.read().copy())

            elapsed = time.monotonic() - t_start
            sleep_for = period - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        try:
            ing.stop()
        except Exception:
            pass

    return frames


# -----------------------------
# Pairwise quality metrics
# -----------------------------
def to_gray_u8(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float32)
    bf = b.astype(np.float32)
    mse = np.mean((af - bf) ** 2)
    if mse <= 1e-9:
        return 99.0
    PIX_MAX = 255.0
    return float(20.0 * np.log10(PIX_MAX) - 10.0 * np.log10(mse))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def brightness_contrast(gray: np.ndarray) -> Tuple[float, float]:
    return float(np.mean(gray)), float(np.std(gray))


def hist_distance(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    ha = cv2.calcHist([gray_a], [0], None, [256], [0, 256])
    hb = cv2.calcHist([gray_b], [0], None, [256], [0, 256])
    cv2.normalize(ha, ha)
    cv2.normalize(hb, hb)
    d = cv2.compareHist(ha, hb, cv2.HISTCMP_BHATTACHARYYA)
    return float(d)


def watermark_roi_delta(a: np.ndarray, b: np.ndarray, roi=(0, 0, 360, 40)) -> float:
    x, y, w, h = roi
    ra = a[y:y+h, x:x+w]
    rb = b[y:y+h, x:x+w]
    if ra.size == 0 or rb.size == 0:
        return 0.0
    return mae(ra, rb)


def analyze_pairwise(in_frames: List[np.ndarray], out_frames: List[np.ndarray]) -> Dict[str, Any]:
    n = min(len(in_frames), len(out_frames))
    if n == 0:
        return {"ok": False, "error": "No frames captured from one or both streams."}

    maes: List[float] = []
    psnrs: List[float] = []
    hds: List[float] = []
    in_b: List[float] = []
    out_b: List[float] = []
    in_c: List[float] = []
    out_c: List[float] = []
    wm_deltas: List[float] = []

    for i in range(n):
        a = in_frames[i]
        b = out_frames[i]
        g1 = to_gray_u8(a)
        g2 = to_gray_u8(b)

        maes.append(mae(a, b))
        psnrs.append(psnr(g1, g2))
        hds.append(hist_distance(g1, g2))

        bm1, ct1 = brightness_contrast(g1)
        bm2, ct2 = brightness_contrast(g2)
        in_b.append(bm1); out_b.append(bm2)
        in_c.append(ct1); out_c.append(ct2)

        wm_deltas.append(watermark_roi_delta(a, b))

    return {
        "ok": True,
        "frames_compared": n,
        "metrics": {
            "mae_mean": float(np.mean(maes)),
            "mae_p95": float(np.percentile(maes, 95)),
            "psnr_gray_mean": float(np.mean(psnrs)),
            "psnr_gray_p05": float(np.percentile(psnrs, 5)),
            "hist_bhattacharyya_mean": float(np.mean(hds)),
            "in_brightness_mean": float(np.mean(in_b)),
            "out_brightness_mean": float(np.mean(out_b)),
            "in_contrast_std_mean": float(np.mean(in_c)),
            "out_contrast_std_mean": float(np.mean(out_c)),
            "watermark_roi_mae_mean": float(np.mean(wm_deltas)),
        },
    }


# -----------------------------
# 7-seg decode (deterministic latency)
# -----------------------------
def _segments_for_digit(d: int) -> Tuple[int, int, int, int, int, int, int]:
    table = {
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
    return table[d]


_SEG_TO_DIGIT = {_segments_for_digit(d): d for d in range(10)}


def decode_7seg_epoch_ms(
    frame: np.ndarray,
    *,
    position: str = "bl",
    seg_len: int = 18,
    seg_th: int = 4,
    spacing: int = 6,
    margin: int = 10,
    digits: int = 13,
    bg_padding: int = 8,
) -> Optional[int]:
    h, w = frame.shape[:2]
    digit_w = seg_len + 2 * seg_th
    digit_h = 2 * seg_len + 3 * seg_th
    total_w = digits * digit_w + (digits - 1) * spacing
    total_h = digit_h

    if position == "tl":
        x0 = margin
        y0 = margin
    else:
        x0 = margin
        y0 = h - margin - total_h

    x1 = max(0, x0 - bg_padding)
    y1 = max(0, y0 - bg_padding)
    x2 = min(w, x0 + total_w + bg_padding)
    y2 = min(h, y0 + total_h + bg_padding)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    def sample(px: int, py: int) -> int:
        px = max(0, min(bw.shape[1] - 1, px))
        py = max(0, min(bw.shape[0] - 1, py))
        return int(bw[py, px] > 0)

    dx0 = x0 - x1
    dy0 = y0 - y1

    out_digits = []
    for i in range(digits):
        ox = dx0 + i * (digit_w + spacing)
        oy = dy0

        a = sample(ox + seg_th + seg_len // 2, oy + seg_th // 2)
        g = sample(ox + seg_th + seg_len // 2, oy + seg_len + seg_th + seg_th // 2)
        dseg = sample(ox + seg_th + seg_len // 2, oy + 2 * seg_len + 2 * seg_th + seg_th // 2)

        f = sample(ox + seg_th // 2, oy + seg_th + seg_len // 2)
        b = sample(ox + seg_th + seg_len + seg_th // 2, oy + seg_th + seg_len // 2)

        e = sample(ox + seg_th // 2, oy + 2 * seg_th + seg_len + seg_len // 2)
        c = sample(ox + seg_th + seg_len + seg_th // 2, oy + 2 * seg_th + seg_len + seg_len // 2)

        seg_tuple = (a, b, c, dseg, e, f, g)
        if seg_tuple not in _SEG_TO_DIGIT:
            return None
        out_digits.append(str(_SEG_TO_DIGIT[seg_tuple]))

    try:
        return int("".join(out_digits))
    except Exception:
        return None


def deterministic_latency_from_output_frames(
    out_frames: List[np.ndarray],
    *,
    position: str,
    seg_len: int,
    seg_th: int,
    spacing: int,
    margin: int,
) -> Dict[str, Any]:
    latencies = []
    parsed = 0
    total = 0
    examples = []

    for f in out_frames:
        total += 1
        embedded = decode_7seg_epoch_ms(
            f,
            position=position,
            seg_len=seg_len,
            seg_th=seg_th,
            spacing=spacing,
            margin=margin,
        )
        if embedded is None:
            continue
        parsed += 1
        now = _epoch_ms()
        lat = now - embedded
        latencies.append(int(lat))
        if len(examples) < 5:
            examples.append({"embedded_ms": embedded, "now_ms": now, "latency_ms": int(lat)})

    if parsed == 0:
        return {"ok": False, "error": "No parseable 7-seg timecodes found in output frames.", "frames_total": total}

    arr = np.array(latencies, dtype=np.int64)
    return {
        "ok": True,
        "frames_total": total,
        "frames_parsed": parsed,
        "parse_rate": float(parsed / max(1, total)),
        "latency_ms": {
            "mean": float(np.mean(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "min": int(np.min(arr)),
            "max": int(np.max(arr)),
        },
        "examples": examples,
    }


# -----------------------------
# Broadcaster test relay (with 7-seg overlay ON)
# -----------------------------
def run_test_relay_with_7seg_timecode(
    in_url: str,
    out_url: str,
    cfg: SampleConfig,
    duration_s: float,
    audiodelay: float,
    video_bitrate: str,
    gop: int,
    audio_copy: bool,
    timecode_position: str,
    seg_len: int,
    seg_th: int,
    spacing: int,
    margin: int,
) -> Dict[str, Any]:
    in_base, in_key = split_rtmp_url(in_url)
    out_base, out_key = split_rtmp_url(out_url)

    ing = Ingestor(
        in_base,
        in_key,
        cfg.width,
        cfg.height,
        ffmpeg_bin=cfg.ffmpeg_bin,
        loglevel=cfg.loglevel,
        watermark_text="",
        timecode_7seg_overlay=True,
        timecode_position=timecode_position,
        timecode_seg_len=seg_len,
        timecode_seg_th=seg_th,
        timecode_digit_spacing=spacing,
        timecode_margin_px=margin,
        timecode_bg=True,
    ).initialize()

    bc = Broadcastor(
        out_base,
        out_key,
        cfg.width,
        cfg.height,
        sourceurl=in_url,
        ffmpeg_bin=cfg.ffmpeg_bin,
        loglevel=cfg.loglevel,
        audiodelay=audiodelay,
        video_bitrate=video_bitrate,
        audio_copy=audio_copy,
        gop=gop,
    ).initialize()

    pushed = 0
    failures = 0
    grabbed = 0
    started = time.monotonic()

    try:
        while time.monotonic() - started < duration_s:
            if not ing.grab():
                break
            frame = ing.read()
            grabbed += 1
            if bc.write(frame):
                pushed += 1
            else:
                failures += 1
                if not bc.is_alive():
                    break
    finally:
        try:
            ing.stop()
        except Exception:
            pass
        try:
            bc.stop()
        except Exception:
            pass

    elapsed = max(1e-6, time.monotonic() - started)
    return {
        "duration_s": float(elapsed),
        "frames_grabbed": grabbed,
        "frames_pushed": pushed,
        "write_failures": failures + getattr(bc, "write_failures", 0),
        "pushed_fps": float(pushed / elapsed),
        "broadcaster_alive_end": bool(bc.is_alive()),
    }


def main():
    ap = argparse.ArgumentParser(description="Analyze RTMP relay quality + broadcaster + deterministic latency (no OCR).")
    ap.add_argument("--in-url", required=True)
    ap.add_argument("--out-url", required=True)

    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)

    ap.add_argument("--sample-seconds", type=float, default=10.0)
    ap.add_argument("--sample-fps", type=float, default=10.0)
    ap.add_argument("--warmup-seconds", type=float, default=2.0)

    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--loglevel", default="error")

    ap.add_argument("--artifacts", default="artifacts/video_analysis")
    ap.add_argument("--pairs", type=int, default=5)

    ap.add_argument("--test-relay", action="store_true", help="Run controlled relay test to out-url with 7-seg overlay.")
    ap.add_argument("--test-duration", type=float, default=8.0)
    ap.add_argument("--audiodelay", type=float, default=2.0)
    ap.add_argument("--video-bitrate", default="2M")
    ap.add_argument("--gop", type=int, default=30)
    ap.add_argument("--audio-copy", action="store_true")
    ap.add_argument("--audio-aac", action="store_true")

    ap.add_argument("--expect-7seg-timecode", action="store_true", help="Decode deterministic latency from output overlay.")
    ap.add_argument("--timecode-position", default="bl", choices=["bl", "tl"])
    ap.add_argument("--timecode-seg-len", type=int, default=18)
    ap.add_argument("--timecode-seg-th", type=int, default=4)
    ap.add_argument("--timecode-spacing", type=int, default=6)
    ap.add_argument("--timecode-margin", type=int, default=10)

    ap.add_argument("--watermark", default="", help="Sampling watermark (avoid for similarity).")
    args = ap.parse_args()

    audio_copy = True
    if args.audio_aac:
        audio_copy = False
    if args.audio_copy:
        audio_copy = True

    os.makedirs(args.artifacts, exist_ok=True)
    ts_label = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.artifacts, ts_label)
    os.makedirs(out_dir, exist_ok=True)

    cfg = SampleConfig(
        width=args.width,
        height=args.height,
        sample_seconds=args.sample_seconds,
        sample_fps=args.sample_fps,
        warmup_seconds=args.warmup_seconds,
        loglevel=args.loglevel,
        ffmpeg_bin=args.ffmpeg,
        watermark_text=args.watermark,
    )

    report: Dict[str, Any] = {
        "timestamp": ts_label,
        "config": asdict(cfg),
        "input_probe_before": ffprobe_summary(args.in_url),
        "output_probe_before": ffprobe_summary(args.out_url),
        "broadcaster_test": None,
        "quality": None,
        "deterministic_latency": None,
        "output_probe_after": None,
    }

    print("=== ffprobe (input/before) ===")
    print(json.dumps(report["input_probe_before"], indent=2))
    print("\n=== ffprobe (output/before) ===")
    print(json.dumps(report["output_probe_before"], indent=2))

    if args.test_relay:
        print("\n=== controlled relay test (7-seg overlay ON) ===")
        bstats = run_test_relay_with_7seg_timecode(
            args.in_url,
            args.out_url,
            cfg,
            duration_s=args.test_duration,
            audiodelay=args.audiodelay,
            video_bitrate=args.video_bitrate,
            gop=args.gop,
            audio_copy=audio_copy,
            timecode_position=args.timecode_position,
            seg_len=args.timecode_seg_len,
            seg_th=args.timecode_seg_th,
            spacing=args.timecode_spacing,
            margin=args.timecode_margin,
        )
        report["broadcaster_test"] = bstats
        print(json.dumps(bstats, indent=2))
        time.sleep(1.0)

    print("\n=== sampling frames (quality) ===")
    in_frames = sample_frames(args.in_url, cfg, disable_overlays=True)
    out_frames = sample_frames(args.out_url, cfg, disable_overlays=True)

    print(f"Captured input frames:  {len(in_frames)}")
    print(f"Captured output frames: {len(out_frames)}")

    report["quality"] = analyze_pairwise(in_frames, out_frames)

    def save_pairs(pairs: int = 5):
        n = min(len(in_frames), len(out_frames), pairs)
        for i in range(n):
            a = in_frames[i]
            b = out_frames[i]
            diff = cv2.absdiff(a, b)
            cv2.imwrite(os.path.join(out_dir, f"in_{i:03d}.png"), a)
            cv2.imwrite(os.path.join(out_dir, f"out_{i:03d}.png"), b)
            cv2.imwrite(os.path.join(out_dir, f"diff_{i:03d}.png"), diff)

    save_pairs(args.pairs)

    if args.expect_7seg_timecode:
        print("\n=== deterministic latency (7-seg decode) ===")
        dl = deterministic_latency_from_output_frames(
            out_frames,
            position=args.timecode_position,
            seg_len=args.timecode_seg_len,
            seg_th=args.timecode_seg_th,
            spacing=args.timecode_spacing,
            margin=args.timecode_margin,
        )
        report["deterministic_latency"] = dl
        print(json.dumps(dl, indent=2))

    report["output_probe_after"] = ffprobe_summary(args.out_url)

    report_path = os.path.join(out_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== ffprobe (output/after) ===")
    print(json.dumps(report["output_probe_after"], indent=2))

    print(f"\nArtifacts written to: {out_dir}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
