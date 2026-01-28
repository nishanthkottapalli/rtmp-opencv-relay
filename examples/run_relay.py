import os
import logging
import argparse

from rtmp_opencv.relay import Ingestor, Broadcastor

os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("process")
logger.setLevel(logging.INFO)

if not logger.handlers:
    fh = logging.FileHandler("logs/process.log")
    fh.setFormatter(logging.Formatter("%(asctime)s:%(process)d:%(levelname)s:%(message)s"))
    logger.addHandler(fh)


def parse_args():
    p = argparse.ArgumentParser(description="Headless RTMP -> OpenCV -> RTMP relay")
    p.add_argument("--in-base", required=True, help='Input RTMP base, e.g. rtmp://host/app/')
    p.add_argument("--in-key", required=True, help="Input stream key")
    p.add_argument("--out-base", required=True, help='Output RTMP base, e.g. rtmp://host/app/')
    p.add_argument("--out-key", required=True, help="Output stream key")
    p.add_argument("--audio-url", default="", help="RTMP URL to read audio from (defaults to full input URL)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--audiodelay", type=float, default=2.0)
    p.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg binary name/path")
    p.add_argument("--loglevel", default="info", help="ffmpeg loglevel (quiet|error|warning|info|debug)")
    p.add_argument("--watermark", default="Copyright YYYY, Company Name Goes Here.")

    # 7-seg deterministic timecode overlay (no OCR)
    p.add_argument(
        "--timecode",
        action="store_true",
        help="Enable 7-seg epoch_ms overlay on frames (for deterministic latency analysis).",
    )
    p.add_argument("--timecode-position", default="bl", choices=["bl", "tl"], help="Overlay position.")
    p.add_argument("--timecode-seg-len", type=int, default=18, help="7-seg segment length (px).")
    p.add_argument("--timecode-seg-th", type=int, default=4, help="7-seg segment thickness (px).")
    p.add_argument("--timecode-spacing", type=int, default=6, help="Digit spacing (px).")
    p.add_argument("--timecode-margin", type=int, default=10, help="Overlay margin from edge (px).")

    return p.parse_args()


def main():
    args = parse_args()

    in_url = args.in_base + args.in_key
    out_url = args.out_base + args.out_key
    audio_url = args.audio_url.strip() or in_url

    logger.info("Starting relay")
    logger.info("IN=%s OUT=%s AUDIO=%s %sx%s", in_url, out_url, audio_url, args.width, args.height)
    logger.info(
        "FFMPEG=%s LOGLEVEL=%s WATERMARK=%s TIMEcode=%s",
        args.ffmpeg,
        args.loglevel,
        bool(args.watermark),
        args.timecode,
    )

    ingestor = Ingestor(
        args.in_base,
        args.in_key,
        args.width,
        args.height,
        ffmpeg_bin=args.ffmpeg,
        loglevel=args.loglevel,
        watermark_text=args.watermark,
        timecode_7seg_overlay=args.timecode,
        timecode_position=args.timecode_position,
        timecode_seg_len=args.timecode_seg_len,
        timecode_seg_th=args.timecode_seg_th,
        timecode_digit_spacing=args.timecode_spacing,
        timecode_margin_px=args.timecode_margin,
        timecode_bg=True,
    ).initialize()

    broadcastor = Broadcastor(
        args.out_base,
        args.out_key,
        args.width,
        args.height,
        audio_url,
        ffmpeg_bin=args.ffmpeg,
        loglevel=args.loglevel,
        audiodelay=args.audiodelay,
    ).initialize()

    try:
        while True:
            if not ingestor.grab():
                logger.warning("Ingestor ended / could not read full frame.")
                break

            frame = ingestor.read()

            if not broadcastor.write(frame):
                logger.warning("Broadcastor write failed (ffmpeg likely exited).")
                break
    finally:
        try:
            ingestor.stop()
        except Exception:
            pass
        try:
            broadcastor.stop()
        except Exception:
            pass

    logger.info("Relay stopped")


if __name__ == "__main__":
    main()
