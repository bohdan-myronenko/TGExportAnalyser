#!/usr/bin/env python3
import shutil
import sys
import argparse
import functools

import whisper


def transcribe_audio(
    filepath: str,
    model_size: str = "tiny",
    language: str = "en"
) -> str:
    """
    Load an audio file via ffmpeg, run Whisper transcription in a forced language,
    and return the detected text.

    Args:
        filepath (str): Path to the audio file.
        model_size (str): Whisper model size (e.g. "tiny", "base", "small", "medium", "large", "turbo").
        language (str): ISO 639-1 code to force transcription (e.g. "en", "fr", "de").

    Returns:
        str: The transcription result.
    """
    # Ensure ffmpeg/avconv is installed
    if not (shutil.which("ffmpeg") or shutil.which("avconv")):
        sys.exit(
            "Error: ffmpeg (or avconv) not found on your PATH.\n"
            "Please install FFmpeg (e.g. `choco install ffmpeg` on Windows "
            "or `conda install -c conda-forge ffmpeg`) and retry."
        )

    whisper.torch.load = functools.partial(whisper.torch.load, weights_only=True)
    # Load Whisper model
    model = whisper.load_model(model_size, device="cuda")

    # Prepare audio
    audio = whisper.load_audio(filepath)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
    mel = mel.to(model.device)

    # Force the chosen language
    options = whisper.DecodingOptions(language=language)
    result = whisper.decode(model, mel, options)
    return result.text


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file using OpenAI Whisper."
    )
    parser.add_argument(
        "filepath",
        help="Path to the audio file (e.g. input/audio.mp3)"
    )
    parser.add_argument(
        "-m", "--model",
        default="tiny",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size to use (default: tiny)"
    )
    parser.add_argument(
        "-l", "--language",
        default="en",
        help="ISO 639-1 code to force transcription language (default: en)"
    )

    args = parser.parse_args()
    try:
        transcription = transcribe_audio(
            filepath=args.filepath,
            model_size=args.model,
            language=args.language
        )
        print(transcription)
    except Exception as e:
        sys.exit(f"Transcription failed: {e}")


if __name__ == "__main__":
    main()
