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


# -----------------------------
# Utilities: ffprobe
# -----------------------------
def _run(cmd: List[str], timeout: int = 15) -> Tuple[int, str, str]:
    p = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE, text=True)
    try:
        out, err = p.communicate(timeout=timeout)
    except sp.TimeoutExpired:
        p.kill()
        out, err = p.communicate()
        return 124, out, err
    return p.returncode, out, err


def ffprobe_summary(url: str) -> Dict[str, Any]:
    """
    Return a compact summary of video+audio streams via ffprobe.
    """
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
        # try avg_frame_rate, fallback r_frame_rate
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


# -----------------------------
# Frame sampling
# -----------------------------
@dataclass
class SampleConfig:
    width: int
    height: int
    sample_seconds: float
    sample_fps: float
    warmup_seconds: float
    loglevel: str


def _ffmpeg_raw_bgr_reader(url: str, cfg: SampleConfig) -> sp.Popen:
    """
    Start ffmpeg to read an RTMP URL and output raw BGR frames at a controlled fps/size.
    """
    # We intentionally scale & fps-filter so comparison is apples-to-apples.
    vf = f"scale={cfg.width}:{cfg.height},fps={cfg.sample_fps}"
    cmd = [
        "ffmpeg",
        "-loglevel", cfg.loglevel,
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-analyzeduration", "0",
        "-probesize", "32",
        "-i", url,
        "-an",
        "-vf", vf,
        "-pix_fmt", "bgr24",
        "-f", "rawvideo",
        "-vsync", "0",
        "-",
    ]
    return sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE)


def _read_exact(pipe, nbytes: int) -> bytes:
    data = b""
    while len(data) < nbytes:
        chunk = pipe.read(nbytes - len(data))
        if not chunk:
            break
        data += chunk
    return data


def sample_frames(url: str, cfg: SampleConfig) -> Tuple[List[np.ndarray], List[float], str]:
    """
    Returns (frames, capture_times_monotonic, ffmpeg_stderr_tail).
    """
    frame_size = cfg.width * cfg.height * 3
    p = _ffmpeg_raw_bgr_reader(url, cfg)

    # Warmup: allow stream decoder to stabilize a bit
    t0 = time.monotonic()
    while time.monotonic() - t0 < cfg.warmup_seconds:
        raw = _read_exact(p.stdout, frame_size) if p.stdout else b""
        if len(raw) != frame_size:
            break

    frames: List[np.ndarray] = []
    ts: List[float] = []

    wanted = max(1, int(cfg.sample_seconds * cfg.sample_fps))
    for _ in range(wanted):
        raw = _read_exact(p.stdout, frame_size) if p.stdout else b""
        if len(raw) != frame_size:
            break
        arr = np.frombuffer(raw, dtype=np.uint8).reshape((cfg.height, cfg.width, 3))
        frames.append(arr.copy())  # copy so buffer isn't reused
        ts.append(time.monotonic())

    # Stop process
    try:
        p.terminate()
    except Exception:
        pass
    try:
        p.wait(timeout=2)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass

    # Capture some stderr (can be large)
    stderr_tail = b""
    try:
        if p.stderr:
            stderr_tail = p.stderr.read()[-4000:]
    except Exception:
        pass
    return frames, ts, stderr_tail.decode(errors="ignore")


# -----------------------------
# Metrics
# -----------------------------
def to_gray_u8(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    # expects uint8 or float; convert to float for mse
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
    # mean and stddev
    return float(np.mean(gray)), float(np.std(gray))


def hist_distance(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    # Bhattacharyya distance on 256-bin histograms
    ha = cv2.calcHist([gray_a], [0], None, [256], [0, 256])
    hb = cv2.calcHist([gray_b], [0], None, [256], [0, 256])
    cv2.normalize(ha, ha)
    cv2.normalize(hb, hb)
    d = cv2.compareHist(ha, hb, cv2.HISTCMP_BHATTACHARYYA)
    return float(d)


def watermark_roi_delta(a: np.ndarray, b: np.ndarray, roi=(0, 0, 360, 40)) -> float:
    """
    Simple heuristic: compare top-left ROI average absolute difference.
    Useful to confirm watermark overlay changed pixels in the expected region.
    roi = (x, y, w, h)
    """
    x, y, w, h = roi
    ra = a[y:y+h, x:x+w]
    rb = b[y:y+h, x:x+w]
    if ra.size == 0 or rb.size == 0:
        return 0.0
    return mae(ra, rb)


def estimate_latency_luma_xcorr(
    in_frames: List[np.ndarray],
    out_frames: List[np.ndarray],
    sample_fps: float,
    max_shift_frames: int = 30,
) -> Optional[float]:
    """
    Rough latency estimate by cross-correlating mean luma time series.
    Works best if the scene has motion or brightness changes.
    Returns latency seconds (out lags in) if detectable.
    """
    n = min(len(in_frames), len(out_frames))
    if n < 10:
        return None

    in_l = np.array([np.mean(to_gray_u8(f)) for f in in_frames[:n]], dtype=np.float32)
    out_l = np.array([np.mean(to_gray_u8(f)) for f in out_frames[:n]], dtype=np.float32)

    # normalize
    in_l = (in_l - in_l.mean()) / (in_l.std() + 1e-6)
    out_l = (out_l - out_l.mean()) / (out_l.std() + 1e-6)

    best_shift = None
    best_score = -1e9

    # shift > 0 means out is delayed relative to in
    for shift in range(-max_shift_frames, max_shift_frames + 1):
        if shift >= 0:
            a = in_l[: n - shift]
            b = out_l[shift:n]
        else:
            a = in_l[-shift:n]
            b = out_l[: n + shift]
        if len(a) < 5:
            continue
        score = float(np.dot(a, b) / len(a))
        if score > best_score:
            best_score = score
            best_shift = shift

    if best_shift is None:
        return None

    # If correlation is weak, don't claim a latency
    if best_score < 0.3:
        return None

    latency_s = best_shift / float(sample_fps)
    # positive latency means output lags input
    return float(latency_s)


# -----------------------------
# Reporting / artifacts
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_artifacts(
    out_dir: str,
    in_frames: List[np.ndarray],
    out_frames: List[np.ndarray],
    pairs: int = 5,
) -> None:
    ensure_dir(out_dir)
    n = min(len(in_frames), len(out_frames), pairs)
    for i in range(n):
        a = in_frames[i]
        b = out_frames[i]
        diff = cv2.absdiff(a, b)
        cv2.imwrite(os.path.join(out_dir, f"in_{i:03d}.png"), a)
        cv2.imwrite(os.path.join(out_dir, f"out_{i:03d}.png"), b)
        cv2.imwrite(os.path.join(out_dir, f"diff_{i:03d}.png"), diff)


def analyze_pairwise(
    in_frames: List[np.ndarray], out_frames: List[np.ndarray], sample_fps: float
) -> Dict[str, Any]:
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

    latency = estimate_latency_luma_xcorr(in_frames[:n], out_frames[:n], sample_fps=sample_fps)

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
        "latency_estimate_seconds": latency,
        "notes": [
            "MAE is pixel-level mean absolute error (lower is better).",
            "PSNR is on grayscale frames (higher is better). Values > 35 dB typically indicate close similarity.",
            "Histogram Bhattacharyya distance (lower is better).",
            "Latency estimate is best-effort via luma cross-correlation; requires visible motion/changes.",
            "Watermark ROI delta is a heuristic that checks top-left region differences.",
        ],
    }


def main():
    ap = argparse.ArgumentParser(
        description="Compare input and output RTMP streams (quality + metadata + latency)."
    )
    ap.add_argument("--in-url", required=True, help="Full input RTMP URL")
    ap.add_argument("--out-url", required=True, help="Full output RTMP URL")
    ap.add_argument("--width", type=int, default=640, help="Sampling width (scaled)")
    ap.add_argument("--height", type=int, default=360, help="Sampling height (scaled)")
    ap.add_argument("--sample-seconds", type=float, default=10.0, help="How long to sample each stream")
    ap.add_argument("--sample-fps", type=float, default=10.0, help="Sampling fps for comparison")
    ap.add_argument("--warmup-seconds", type=float, default=2.0, help="Warmup decode before sampling")
    ap.add_argument("--loglevel", default="error", help="ffmpeg loglevel for samplers (error|warning|info)")
    ap.add_argument("--artifacts", default="artifacts/video_analysis", help="Output artifacts dir")
    ap.add_argument("--pairs", type=int, default=5, help="How many frame pairs to save as PNG")
    args = ap.parse_args()

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
    )

    print("=== ffprobe (input) ===")
    in_probe = ffprobe_summary(args.in_url)
    print(json.dumps(in_probe, indent=2))

    print("\n=== ffprobe (output) ===")
    out_probe = ffprobe_summary(args.out_url)
    print(json.dumps(out_probe, indent=2))

    print("\n=== sampling frames ===")
    in_frames, in_ts, in_stderr = sample_frames(args.in_url, cfg)
    out_frames, out_ts, out_stderr = sample_frames(args.out_url, cfg)

    print(f"Captured input frames:  {len(in_frames)}")
    print(f"Captured output frames: {len(out_frames)}")

    if in_stderr.strip():
        with open(os.path.join(out_dir, "ffmpeg_in_stderr_tail.txt"), "w", encoding="utf-8") as f:
            f.write(in_stderr)
    if out_stderr.strip():
        with open(os.path.join(out_dir, "ffmpeg_out_stderr_tail.txt"), "w", encoding="utf-8") as f:
            f.write(out_stderr)

    report: Dict[str, Any] = {
        "timestamp": ts_label,
        "config": asdict(cfg),
        "input_probe": in_probe,
        "output_probe": out_probe,
        "capture": {
            "input_frames": len(in_frames),
            "output_frames": len(out_frames),
        },
        "analysis": analyze_pairwise(in_frames, out_frames, sample_fps=cfg.sample_fps),
    }

    # Save artifacts (paired frames + diffs)
    save_artifacts(out_dir, in_frames, out_frames, pairs=args.pairs)

    # Write report JSON
    report_path = os.path.join(out_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== analysis summary ===")
    if report["analysis"].get("ok"):
        m = report["analysis"]["metrics"]
        print(f"Frames compared: {report['analysis']['frames_compared']}")
        print(f"MAE mean:       {m['mae_mean']:.3f} (p95={m['mae_p95']:.3f})")
        print(f"PSNR mean:      {m['psnr_gray_mean']:.2f} dB (p05={m['psnr_gray_p05']:.2f} dB)")
        print(f"Hist dist mean: {m['hist_bhattacharyya_mean']:.4f}")
        print(f"Brightness in/out: {m['in_brightness_mean']:.2f} / {m['out_brightness_mean']:.2f}")
        print(f"Contrast in/out:   {m['in_contrast_std_mean']:.2f} / {m['out_contrast_std_mean']:.2f}")
        print(f"Watermark ROI Δ:   {m['watermark_roi_mae_mean']:.3f}")
        print(f"Latency estimate:  {report['analysis']['latency_estimate_seconds']} (seconds, best-effort)")
    else:
        print("Analysis failed:", report["analysis"].get("error"))

    print(f"\nArtifacts written to: {out_dir}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
