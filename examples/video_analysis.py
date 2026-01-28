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
    if "/" not in url:
        raise ValueError(f"Invalid RTMP URL: {url}")
    base, key = url.rsplit("/", 1)
    return base + "/", key


def sample_frames(url: str, cfg: SampleConfig) -> Tuple[List[np.ndarray], List[float]]:
    """
    Sample frames using our Ingestor (which now enforces scaling).
    """
    base, key = split_rtmp_url(url)
    ing = Ingestor(
        base,
        key,
        cfg.width,
        cfg.height,
        ffmpeg_bin=cfg.ffmpeg_bin,
        loglevel=cfg.loglevel,
        watermark_text=cfg.watermark_text,
    ).initialize()

    frames: List[np.ndarray] = []
    ts: List[float] = []

    wanted = max(1, int(cfg.sample_seconds * cfg.sample_fps))
    period = 1.0 / max(1e-6, cfg.sample_fps)

    try:
        # warmup
        t0 = time.monotonic()
        while time.monotonic() - t0 < cfg.warmup_seconds:
            if not ing.grab():
                break

        for _ in range(wanted):
            t_start = time.monotonic()
            if not ing.grab():
                break
            frame = ing.read()
            frames.append(frame.copy())
            ts.append(time.monotonic())

            elapsed = time.monotonic() - t_start
            sleep_for = period - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        try:
            ing.stop()
        except Exception:
            pass

    return frames, ts


# -----------------------------
# Metrics
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


def estimate_latency_luma_xcorr(
    in_frames: List[np.ndarray],
    out_frames: List[np.ndarray],
    sample_fps: float,
    max_shift_frames: int = 30,
) -> Optional[float]:
    n = min(len(in_frames), len(out_frames))
    if n < 10:
        return None

    in_l = np.array([np.mean(to_gray_u8(f)) for f in in_frames[:n]], dtype=np.float32)
    out_l = np.array([np.mean(to_gray_u8(f)) for f in out_frames[:n]], dtype=np.float32)

    in_l = (in_l - in_l.mean()) / (in_l.std() + 1e-6)
    out_l = (out_l - out_l.mean()) / (out_l.std() + 1e-6)

    best_shift = None
    best_score = -1e9

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

    if best_shift is None or best_score < 0.3:
        return None

    return float(best_shift / float(sample_fps))


def analyze_pairwise(in_frames: List[np.ndarray], out_frames: List[np.ndarray], sample_fps: float) -> Dict[str, Any]:
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
    }


# -----------------------------
# Artifacts
# -----------------------------
def save_artifacts(out_dir: str, in_frames: List[np.ndarray], out_frames: List[np.ndarray], pairs: int = 5) -> None:
    os.makedirs(out_dir, exist_ok=True)
    n = min(len(in_frames), len(out_frames), pairs)
    for i in range(n):
        a = in_frames[i]
        b = out_frames[i]
        diff = cv2.absdiff(a, b)
        cv2.imwrite(os.path.join(out_dir, f"in_{i:03d}.png"), a)
        cv2.imwrite(os.path.join(out_dir, f"out_{i:03d}.png"), b)
        cv2.imwrite(os.path.join(out_dir, f"diff_{i:03d}.png"), diff)


# -----------------------------
# Broadcaster analysis: test relay
# -----------------------------
def run_test_relay(
    in_url: str,
    out_url: str,
    cfg: SampleConfig,
    duration_s: float,
    audiodelay: float,
    video_bitrate: str,
    gop: int,
    audio_copy: bool,
) -> Dict[str, Any]:
    """
    Runs a controlled relay: input -> Broadcastor -> out_url for duration_s.
    Returns broadcaster stats (push fps, failures, alive, etc.).
    """
    in_base, in_key = split_rtmp_url(in_url)
    out_base, out_key = split_rtmp_url(out_url)

    # Ingest frames WITHOUT watermark (we want analysis on raw relay behavior).
    ing = Ingestor(
        in_base,
        in_key,
        cfg.width,
        cfg.height,
        ffmpeg_bin=cfg.ffmpeg_bin,
        loglevel=cfg.loglevel,
        watermark_text="",  # raw frames in this test
    ).initialize()

    bc = Broadcastor(
        out_base,
        out_key,
        cfg.width,
        cfg.height,
        sourceurl=in_url,  # use input url for audio
        ffmpeg_bin=cfg.ffmpeg_bin,
        loglevel=cfg.loglevel,
        audiodelay=audiodelay,
        video_bitrate=video_bitrate,
        audio_copy=audio_copy,
        gop=gop,
    ).initialize()

    pushed = 0
    write_failures = 0
    grabbed = 0
    started = time.monotonic()

    try:
        while time.monotonic() - started < duration_s:
            if not ing.grab():
                break
            frame = ing.read()
            grabbed += 1
            ok = bc.write(frame)
            if ok:
                pushed += 1
            else:
                write_failures += 1
                # if broadcaster died, stop early
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
    stats = {
        "duration_s": float(elapsed),
        "frames_grabbed": grabbed,
        "frames_pushed": pushed,
        "write_failures": write_failures + getattr(bc, "write_failures", 0),
        "pushed_fps": float(pushed / elapsed),
        "broadcaster_alive_end": bool(bc.is_alive()) if hasattr(bc, "is_alive") else None,
        "broadcaster_internal": {
            "frames_pushed": getattr(bc, "frames_pushed", None),
            "write_failures": getattr(bc, "write_failures", None),
            "pushed_fps": getattr(bc, "pushed_fps", lambda: None)(),
        },
    }
    return stats


def main():
    ap = argparse.ArgumentParser(
        description="Analyze RTMP relay quality + broadcaster performance using rtmp_opencv.relay."
    )
    ap.add_argument("--in-url", required=True, help="Full input RTMP URL")
    ap.add_argument("--out-url", required=True, help="Full output RTMP URL")

    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)

    ap.add_argument("--sample-seconds", type=float, default=10.0, help="Sampling duration for quality comparison")
    ap.add_argument("--sample-fps", type=float, default=10.0)
    ap.add_argument("--warmup-seconds", type=float, default=2.0)

    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--loglevel", default="error", help="ffmpeg loglevel for analysis and relay test")

    ap.add_argument("--artifacts", default="artifacts/video_analysis")
    ap.add_argument("--pairs", type=int, default=5)

    # broadcaster test controls
    ap.add_argument("--test-relay", action="store_true", help="Run a controlled relay test before sampling output")
    ap.add_argument("--test-duration", type=float, default=8.0, help="Seconds to run controlled relay test")
    ap.add_argument("--audiodelay", type=float, default=2.0)
    ap.add_argument("--video-bitrate", default="2M")
    ap.add_argument("--gop", type=int, default=30)
    ap.add_argument("--audio-copy", action="store_true", help="Use -c:a copy in broadcaster (default)")
    ap.add_argument("--audio-aac", action="store_true", help="Force AAC encode instead of copy")

    # watermark option for sampling (off by default)
    ap.add_argument("--watermark", default="", help="If set, applies watermark during sampling")

    args = ap.parse_args()

    # resolve audio mode
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
        "output_probe_after": None,
    }

    print("=== ffprobe (input/before) ===")
    print(json.dumps(report["input_probe_before"], indent=2))
    print("\n=== ffprobe (output/before) ===")
    print(json.dumps(report["output_probe_before"], indent=2))

    # Optional: run a controlled relay test (broadcaster analysis)
    if args.test_relay:
        print("\n=== running controlled relay test (broadcaster analysis) ===")
        bstats = run_test_relay(
            args.in_url,
            args.out_url,
            cfg,
            duration_s=args.test_duration,
            audiodelay=args.audiodelay,
            video_bitrate=args.video_bitrate,
            gop=args.gop,
            audio_copy=audio_copy,
        )
        report["broadcaster_test"] = bstats
        print(json.dumps(bstats, indent=2))

        # Give output a moment to stabilize post-test
        time.sleep(1.0)

    # Quality sampling: input + output
    print("\n=== sampling frames via Ingestor (quality analysis) ===")
    in_frames, _ = sample_frames(args.in_url, cfg)
    out_frames, _ = sample_frames(args.out_url, cfg)

    print(f"Captured input frames:  {len(in_frames)}")
    print(f"Captured output frames: {len(out_frames)}")

    report["quality"] = analyze_pairwise(in_frames, out_frames, sample_fps=cfg.sample_fps)

    save_artifacts(out_dir, in_frames, out_frames, pairs=args.pairs)

    # Probe output after sampling (might differ if test-relay ran)
    report["output_probe_after"] = ffprobe_summary(args.out_url)

    # Write report
    report_path = os.path.join(out_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== ffprobe (output/after) ===")
    print(json.dumps(report["output_probe_after"], indent=2))

    print("\n=== quality summary ===")
    q = report["quality"]
    if q and q.get("ok"):
        m = q["metrics"]
        print(f"Frames compared:    {q['frames_compared']}")
        print(f"MAE mean:           {m['mae_mean']:.3f} (p95={m['mae_p95']:.3f})")
        print(f"PSNR mean:          {m['psnr_gray_mean']:.2f} dB (p05={m['psnr_gray_p05']:.2f} dB)")
        print(f"Hist dist mean:     {m['hist_bhattacharyya_mean']:.4f}")
        print(f"Brightness in/out:  {m['in_brightness_mean']:.2f} / {m['out_brightness_mean']:.2f}")
        print(f"Contrast in/out:    {m['in_contrast_std_mean']:.2f} / {m['out_contrast_std_mean']:.2f}")
        print(f"Watermark ROI Δ:    {m['watermark_roi_mae_mean']:.3f}")
        print(f"Latency estimate:   {q.get('latency_estimate_seconds')} seconds (best-effort)")
    else:
        print("Quality analysis failed:", (q or {}).get("error"))

    if report.get("broadcaster_test"):
        b = report["broadcaster_test"]
        print("\n=== broadcaster summary ===")
        print(f"Test duration:      {b['duration_s']:.2f}s")
        print(f"Frames grabbed:     {b['frames_grabbed']}")
        print(f"Frames pushed:      {b['frames_pushed']}")
        print(f"Write failures:     {b['write_failures']}")
        print(f"Pushed FPS:         {b['pushed_fps']:.2f}")
        print(f"Broadcaster alive:  {b['broadcaster_alive_end']}")

    print(f"\nArtifacts written to: {out_dir}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
