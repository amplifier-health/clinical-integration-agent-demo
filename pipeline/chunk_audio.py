#!/usr/bin/env python3
"""Split a mono speech recording into fixed windows for streaming analysis.

Usage:
    python pipeline/chunk_audio.py INPUT.wav OUTPUT_DIR [--window 30] [--hop 15]

Produces OUTPUT_DIR/c000_0000s.wav, c001_0015s.wav, ... Chunks shorter than the
API minimum (15s) are skipped. Requires ffmpeg / ffprobe on PATH.
"""
import argparse, os, subprocess, math

MIN_SECONDS = 15  # Amplifier API minimum accepted duration

def duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip()
    return float(out or 0)

def chunk(input_wav, out_dir, window=30, hop=15):
    os.makedirs(out_dir, exist_ok=True)
    dur = duration(input_wav)
    made, skipped = 0, 0
    starts = [0.0] if dur <= window else [i * hop for i in range(math.ceil((dur - window) / hop) + 1)]
    for i, s in enumerate(starts):
        length = min(window, dur - s)
        if length < MIN_SECONDS:
            skipped += 1
            continue
        out = os.path.join(out_dir, f"c{i:03d}_{int(s):04d}s.wav")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(s), "-t", str(window),
                        "-i", input_wav, "-ac", "1", "-ar", "16000", out], check=True)
        made += 1
    print(f"{input_wav}: {made} chunks written, {skipped} short tail(s) skipped ({dur:.0f}s total)")
    return made

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input_wav")
    p.add_argument("out_dir")
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--hop", type=int, default=15)
    a = p.parse_args()
    chunk(a.input_wav, a.out_dir, a.window, a.hop)
