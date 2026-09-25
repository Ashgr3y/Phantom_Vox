"""
Captures the OTHER person's voice during a call by recording your
speaker/system output (loopback) instead of your microphone, then sends
each chunk to your FastAPI /detect endpoint for spoof detection.

Windows only (uses WASAPI loopback).

Setup:
    pip install PyAudioWPatch requests

Usage:
    1. Make sure `uvicorn main:app --reload` is running in another terminal.
    2. Start your call (any app — Zoom, Discord, a phone bridged through
       your PC, whatever) so audio is playing through your speakers/headset.
    3. Run this script: python record_loopback.py
    4. Ctrl+C to stop.
"""

import io
import wave
from datetime import datetime
from pathlib import Path

import pyaudiowpatch as pyaudio
import requests

API_URL = "http://127.0.0.1:8000/detect"
CHUNK_SECONDS = 5          # how much audio to capture before sending
FORMAT = pyaudio.paInt16
FRAMES_PER_BUFFER = 1024

# Every captured chunk gets saved here so you can play it back and verify
# the script is actually hearing what you think it's hearing.
SAVE_CHUNKS = True
CHUNKS_DIR = Path("captured_chunks")


def record_chunk(p: "pyaudio.PyAudio", device: dict, seconds: float) -> io.BytesIO:
    channels = device["maxInputChannels"]
    rate = int(device["defaultSampleRate"])

    stream = p.open(
        format=FORMAT,
        channels=channels,
        rate=rate,
        input=True,
        input_device_index=device["index"],
        frames_per_buffer=FRAMES_PER_BUFFER,
    )

    frames = []
    num_reads = int(rate / FRAMES_PER_BUFFER * seconds)
    for _ in range(num_reads):
        frames.append(stream.read(FRAMES_PER_BUFFER))

    stream.stop_stream()
    stream.close()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(rate)
        wf.writeframes(b"".join(frames))
    buf.seek(0)
    return buf


def get_default_loopback_device(p: "pyaudio.PyAudio") -> dict:
    """Find the loopback device matching your current default speakers."""
    wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])

    if default_speakers["isLoopbackDevice"]:
        return default_speakers

    for loopback in p.get_loopback_device_info_generator():
        if default_speakers["name"] in loopback["name"]:
            return loopback

    raise RuntimeError(
        "No loopback device found for your default speakers. "
        "Check Windows Sound settings -> make sure a playback device is active."
    )


def main():
    p = pyaudio.PyAudio()

    try:
        device = get_default_loopback_device(p)
        print(f"Capturing from: {device['name']}")
        print(f"Sending {CHUNK_SECONDS}s chunks to {API_URL}")
        if SAVE_CHUNKS:
            CHUNKS_DIR.mkdir(exist_ok=True)
            print(f"Saving each chunk to: {CHUNKS_DIR.resolve()}")
        print("Ctrl+C to stop.\n")

        while True:
            wav_buffer = record_chunk(p, device, CHUNK_SECONDS)

            if SAVE_CHUNKS:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                chunk_path = CHUNKS_DIR / f"chunk_{timestamp}.wav"
                chunk_path.write_bytes(wav_buffer.getvalue())
                wav_buffer.seek(0)  # rewind so it can still be uploaded below

            response = requests.post(
                API_URL,
                files={"file": ("chunk.wav", wav_buffer, "audio/wav")},
            )

            if response.ok:
                result = response.json()
                saved_note = f" [{chunk_path.name}]" if SAVE_CHUNKS else ""
                print(
                    f"[{result['final_prediction']}] "
                    f"spoof={result['mean_spoof_score']:.3f} "
                    f"ratio={result['spoof_window_ratio']:.2f} "
                    f"-> {result['action']}: {result['message']}{saved_note}"
                )
            else:
                print(f"Request failed: {response.status_code} {response.text}")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        p.terminate()


if __name__ == "__main__":
    main()
