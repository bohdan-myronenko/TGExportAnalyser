#!/usr/bin/env python3
import argparse
import shutil
import subprocess
import sys
import math
import os
from typing import List


def get_duration(filepath: str) -> float:
    """
    Return the total duration of the audio file in seconds, using ffprobe.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        filepath
    ]
    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
    return float(output)


def split_into_chunks(
    input_file: str,
    output_dir: str,
    chunk_length: int = 30,
    min_length: int = 5
) -> List[str]:
    """
    Split `input_file` into consecutive chunks of `chunk_length` seconds.
    If the last chunk is shorter than chunk_length but >= min_length, it is kept.
    Returns a list of generated file paths.
    """
    # Ensure external tools are available
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        sys.exit(
            "Error: ffmpeg and/or ffprobe not found on your PATH.\n"
            "Install FFmpeg (e.g. `choco install ffmpeg` or "
            "`conda install -c conda-forge ffmpeg`) and retry."
        )

    os.makedirs(output_dir, exist_ok=True)
    total_duration = get_duration(input_file)
    num_chunks = math.ceil(total_duration / chunk_length)
    generated_files: List[str] = []

    for i in range(num_chunks):
        start = i * chunk_length
        end = min((i + 1) * chunk_length, total_duration)
        segment_duration = end - start

        # Drop segments shorter than min_length
        if segment_duration < min_length:
            break

        base_name = os.path.splitext(os.path.basename(input_file))[0]
        out_path = os.path.join(output_dir, f"{base_name}_part{i+1:03d}.ogg")

        cmd = [
            "ffmpeg",
            "-n",
            "-i", input_file,
            "-acodec", "copy",
            "-ss", str(start),
            "-to", str(end),
            out_path
        ]
        subprocess.run(cmd, check=True)
        generated_files.append(out_path)

    return generated_files


def main():
    parser = argparse.ArgumentParser(
        description="Split an audio file into fixed-duration chunks."
    )
    parser.add_argument(
        "input_file",
        help="Path to the source audio file (e.g. input.mp3)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=".",
        help="Directory to save chunks (default: current directory)"
    )
    parser.add_argument(
        "-l", "--length",
        type=int,
        default=30,
        help="Chunk length in seconds (default: 30)"
    )
    parser.add_argument(
        "-m", "--min-length",
        type=int,
        default=5,
        help="Minimum duration (in seconds) to keep the last chunk (default: 5)"
    )
    args = parser.parse_args()

    try:
        files = split_into_chunks(
            input_file=args.input_file,
            output_dir=args.output_dir,
            chunk_length=args.length,
            min_length=args.min_length
        )
        if files:
            print("Generated chunks:")
            print("\n".join(files))
        else:
            print("No chunks generated (file shorter than min-length).")
    except subprocess.CalledProcessError as e:
        sys.exit(f"FFmpeg failed: {e}")
    except Exception as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
