#!/usr/bin/env python3
"""Robust audio WebSocket client

Key features
------------
* **Single playback worker** driven by an ``asyncio.Queue`` – avoids
  concurrent PortAudio streams that used to corrupt memory.
* **Stereo‑safe decoder**: reshapes WAV frames to ``(frames, channels)``
  so CoreAudio receives the correct layout and doesn’t raise -10851 /
  PaError ‑9986.
* **Automatic PortAudio recovery**: on ``sounddevice.PortAudioError`` we
  reset the stream and retry once.
* **HTTP endpoints** (``/`` and ``/recording``) let external code signal
  *start/stop speaking*.
* **Configurable port**: set ``ROBOT_HTTP_PORT`` env‑var to avoid clashes.

Tested on macOS 14.4 with Python 3.12 + sounddevice 0.4.6.
"""

from __future__ import annotations

import asyncio
import base64
import errno
import io
import json
import os
import sys
import time
import wave
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd
import websockets
from aiohttp import web

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
ROBOT_ID = "robot_1"

# PRIMARY_WS_URI = (
#     "wss://app-ragbackend-dev-wus-001.azurewebsites.net/ws/{ROBOT_ID}/before/lecture"
# )
# LECTURE_WS_URI = (
#     f"wss://app-ragbackend-dev-wus-001.azurewebsites.net/ws/{ROBOT_ID}/lesson_audio"
# )

PRIMARY_WS_URI = (
    f"ws://localhost:8000/ws/{ROBOT_ID}/before/lecture"
)
LECTURE_WS_URI = (
    f"ws://localhost:8000/ws/{ROBOT_ID}/lesson_audio"
)


HTTP_PORT = int(os.getenv("ROBOT_HTTP_PORT", 5000))

# ---------------------------------------------------------------------------
# GLOBAL STATE
# ---------------------------------------------------------------------------
playback_queue: asyncio.Queue[Tuple[str, bytes]] = asyncio.Queue()
stop_playback_event = asyncio.Event()  # set() → interrupt current clip
spoken_text: str = ""  # what portion of TTS actually played

# ---------------------------------------------------------------------------
# WAV HELPERS
# ---------------------------------------------------------------------------

def _decode_wav(audio_bytes: bytes) -> Tuple[int, np.ndarray]:
    """Return (samplerate, audio[f, ch]) in float32 ∈ [-1, 1]."""
    with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
        sr = wf.getframerate()
        n_ch = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())
    
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if n_ch > 1:
        audio = audio.reshape(-1, n_ch)
    return sr, audio

# ---------------------------------------------------------------------------
# PLAYBACK WORKER
# ---------------------------------------------------------------------------
async def playback_worker() -> None:
    """Continuously pull clips from the queue and play them serially."""
    global spoken_text

    while True:
        text, audio_bytes = await playback_queue.get()
        stop_playback_event.clear()
        if isinstance(audio_bytes, bytes):  # Ensure the audio data is in bytes
            sr, audio = _decode_wav(audio_bytes)  # Decode and play the audio
        else:
            print(f"Received invalid audio data")
            continue
        duration = len(audio) / sr
        channels = 1 if audio.ndim == 1 else audio.shape[1]
        print(f"🔊 clip: {duration:.2f}s @ {sr} Hz, {channels}ch")

        if isinstance(text, dict):
            text = text.get("text", "")  # Adjust this line based on your data structure

        words = text.split()
        word_dur = duration / len(words) if words else 0.0

        for attempt in (1, 2):  # try at most twice
            try:
                sd.stop()
                sd.play(audio, sr, blocking=False)
                start = time.time()
                while True:
                    if stop_playback_event.is_set():
                        sd.stop()
                        print("⏹️  Playback interrupted by user")
                        break
                    if (time.time() - start) >= duration:
                        break
                    await asyncio.sleep(0.05)
                break  # success – break out of retry loop
            except sd.PortAudioError as exc:
                if attempt == 2:
                    print(f"❌ PortAudio failure: {exc}")
                    break
                print(f"⚠️  PortAudio error ({exc}); resetting …")
                sd.stop(); sd.wait()
                await asyncio.sleep(0.05)  # let CoreAudio settle
                # second attempt uses blocking=True to fully reset
                sd.play(audio, sr, blocking=True)
                # loop will exit immediately after blocking play
                start = time.time() - duration
                break

        # figure out how much was spoken
        elapsed = min(time.time() - start, duration)
        spoken_words = int(elapsed / word_dur) if word_dur else 0
        spoken_text = " ".join(words[:spoken_words])
        if spoken_text:
            print(f"🗣️  Spoken text: '{spoken_text}'")
        else:
            print("✅ Playback finished")

# ---------------------------------------------------------------------------
# ENQUEUE FUNCTION
# # ---------------------------------------------------------------------------
# async def enqueue_clip(audio_bytes: str, text: str = "") -> None:
#     """Place a clip on the playback queue, dropping old clips if stopped."""
#     try:
#         print(f"Received audio of length: {len(audio_bytes)} bytes")
#         playback_queue.put_nowait((text, audio_bytes))
#     except asyncio.QueueFull:
#         print("⚠️  Playback queue full – dropping clip")

# # ---------------------------------------------------------------------------
# HTTP ENDPOINTS
# ---------------------------------------------------------------------------
async def handle_start_speaking(request: web.Request):
    data = await request.json()
    
    if data.get("message") == "Start speaking":
        print("Start speaking – interrupting playback")
        stop_playback_event.set()
        # Flush anything pending so new answer plays immediately
        while not playback_queue.empty():
            try:
                playback_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        return web.Response(text="true")


async def handle_recording_end(request: web.Request):
    data = await request.json()
    if data.get("message") == "Recording Complete":
        print("Recording Complete – mic closed")
    return web.Response(text=spoken_text)


async def start_http_server() -> None:
    app = web.Application()
    app.router.add_post("/", handle_start_speaking)
    app.router.add_post("/recording", handle_recording_end)

    runner = web.AppRunner(app)
    await runner.setup()

    try:
        site = web.TCPSite(runner, "localhost", HTTP_PORT)
        await site.start()
    except OSError as e:
        if e.errno == errno.EADDRINUSE:
            print(f"❌ Port {HTTP_PORT} already in use – choose another via $ROBOT_HTTP_PORT")
            sys.exit(1)
        raise
    print(f"🚀 HTTP server listening on http://localhost:{HTTP_PORT}")

# ---------------------------------------------------------------------------
# WEBSOCKET CLIENTS
# ---------------------------------------------------------------------------
async def _generic_socket(uri: str, label: str) -> None:
    retry = 0
    while True:
        try:
            print(f"🔄 Connecting to {uri} …")
            async with websockets.connect(uri, ping_interval=None, close_timeout=10, max_size=None) as ws:
                print(f"🔗 Connected ({label})")
                retry = 0
                if label == "primary":
                    await ws.send(json.dumps({
                        "type": "register",
                        "data": {
                            "client": "audio"      # lets server distinguish roles
                        },
                        "ts": time.time(),
                    }))
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        print(f"❌ JSON error: {exc}")
                        continue

                    if label == "primary" and msg.get("robot_id") != ROBOT_ID:
                        continue

                    text = msg.get("text", "")
                    if text:
                        print(f"📝 {text}")
                    audio_list = msg.get("audio")
                    if audio_list:
                        try:
                            # Convert list of bytes back to bytes
                            audio_bytes = bytes(audio_list)
                            print(f"Received audio of length: {len(audio_bytes)} bytes")
                            await enqueue_clip(audio_bytes, text)
                        except Exception as exc:
                            print(f"❌ Error processing audio data: {exc}")
        except Exception as exc:
            retry += 1
            delay = min(60, 2 ** retry)
            print(f"⚠️  {label} socket error: {exc} – reconnecting in {delay}s …")
            await asyncio.sleep(delay)


auth_primary = lambda: _generic_socket(PRIMARY_WS_URI, "primary")
auth_lecture = lambda: _generic_socket(LECTURE_WS_URI, "lecture")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
async def main() -> None:
    print("🚀 Starting WebSocket client …  (Ctrl‑C to quit)")
    await start_http_server()
    await asyncio.gather(playback_worker(), auth_primary(), auth_lecture())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Bye!")



# # ---------------------------------------------------------------------------
# # GLOBAL STATE
# # ---------------------------------------------------------------------------
# playback_queue: asyncio.Queue[Tuple[str, str]] = asyncio.Queue()
# stop_playback_event = asyncio.Event()  # set() → interrupt current clip
# spoken_text: str = ""  # what portion of TTS actually played

# # ---------------------------------------------------------------------------
# # WAV HELPERS
# # ---------------------------------------------------------------------------

# def _decode_wav(b64: str) -> Tuple[int, np.ndarray]:
#     """Return (samplerate, audio[f, ch]) in float32 ∈ [-1, 1]."""
#     buf = base64.b64decode(b64)
#     with wave.open(io.BytesIO(buf), "rb") as wf:
#         sr = wf.getframerate()
#         n_ch = wf.getnchannels()
#         frames = wf.readframes(wf.getnframes())

#     audio = (
#         np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
#     )
#     if n_ch > 1:
#         audio = audio.reshape(-1, n_ch)  # shape (frames, channels)
#     return sr, audio

# # ---------------------------------------------------------------------------
# # PLAYBACK WORKER
# # ---------------------------------------------------------------------------
# async def playback_worker() -> None:
#     """Continuously pull clips from the queue and play them serially."""
#     global spoken_text

#     while True:
#         text, audio_b64 = await playback_queue.get()
#         stop_playback_event.clear()
#         sr, audio = _decode_wav(audio_b64)
#         duration = len(audio) / sr
#         channels = 1 if audio.ndim == 1 else audio.shape[1]
#         print(f"🔊 clip: {duration:.2f}s @ {sr} Hz, {channels}ch")

#         if isinstance(text, dict):
#             text = text.get("text", "")  # Adjust this line based on your data structure
#         words = text.split()
#         word_dur = duration / len(words) if words else 0.0

#         for attempt in (1, 2):  # try at most twice
#             try:
#                 sd.stop()
#                 sd.play(audio, sr, blocking=False)
#                 start = time.time()
#                 while True:
#                     if stop_playback_event.is_set():
#                         sd.stop()
#                         print("⏹️  Playback interrupted by user")
#                         break
#                     if (time.time() - start) >= duration:
#                         break
#                     await asyncio.sleep(0.05)
#                 break  # success – break out of retry loop
#             except sd.PortAudioError as exc:
#                 if attempt == 2:
#                     print(f"❌ PortAudio failure: {exc}")
#                     break
#                 print(f"⚠️  PortAudio error ({exc}); resetting …")
#                 sd.stop(); sd.wait()
#                 await asyncio.sleep(0.05)  # let CoreAudio settle
#                 # second attempt uses blocking=True to fully reset
#                 sd.play(audio, sr, blocking=True)
#                 # loop will exit immediately after blocking play
#                 start = time.time() - duration
#                 break

#         # figure out how much was spoken
#         elapsed = min(time.time() - start, duration)
#         spoken_words = int(elapsed / word_dur) if word_dur else 0
#         spoken_text = " ".join(words[:spoken_words])
#         if spoken_text:
#             print(f"🗣️  Spoken text: '{spoken_text}'")
#         else:
#             print("✅ Playback finished")

# # ---------------------------------------------------------------------------
# # ENQUEUE FUNCTION
# # ---------------------------------------------------------------------------
# async def enqueue_clip(audio_b64: str, text: str = "") -> None:
#     """Place a clip on the playback queue, dropping old clips if stopped."""
#     try:
#         playback_queue.put_nowait((text, audio_b64))
#     except asyncio.QueueFull:
#         print("⚠️  Playback queue full – dropping clip")

# # ---------------------------------------------------------------------------
# # HTTP ENDPOINTS
# # ---------------------------------------------------------------------------
# async def handle_start_speaking(request: web.Request):
#     data = await request.json()
    
#     if data.get("message") == "Start speaking":
#         print("Start speaking – interrupting playback")
#         stop_playback_event.set()
#         # Flush anything pending so new answer plays immediately
#         while not playback_queue.empty():
#             try:
#                 playback_queue.get_nowait()
#             except asyncio.QueueEmpty:
#                 break
#         return web.Response(text="true")


# async def handle_recording_end(request: web.Request):
#     data = await request.json()
#     if data.get("message") == "Recording Complete":
#         print("Recording Complete – mic closed")
#     return web.Response(text=spoken_text)


# async def start_http_server() -> None:
#     app = web.Application()
#     app.router.add_post("/", handle_start_speaking)
#     app.router.add_post("/recording", handle_recording_end)

#     runner = web.AppRunner(app)
#     await runner.setup()

#     try:
#         site = web.TCPSite(runner, "localhost", HTTP_PORT)
#         await site.start()
#     except OSError as e:
#         if e.errno == errno.EADDRINUSE:
#             print(f"❌ Port {HTTP_PORT} already in use – choose another via $ROBOT_HTTP_PORT")
#             sys.exit(1)
#         raise
#     print(f"🚀 HTTP server listening on http://localhost:{HTTP_PORT}")

# # ---------------------------------------------------------------------------
# # WEBSOCKET CLIENTS
# # ---------------------------------------------------------------------------
# async def _generic_socket(uri: str, label: str) -> None:
#     retry = 0
#     while True:
#         try:
#             print(f"🔄 Connecting to {uri} …")
#             async with websockets.connect(uri, ping_interval=None, close_timeout=10, max_size=None) as ws:
#                 print(f"🔗 Connected ({label})")
#                 retry = 0
#                 if label == "primary":
#                     await ws.send(json.dumps({
#                         "type": "register",
#                         "data": {
#                             "client": "audio"      # lets server distinguish roles
#                         },
#                         "ts": time.time(),
#                     }))
#                 async for raw in ws:
#                     try:
#                         msg = json.loads(raw)
#                     except json.JSONDecodeError as exc:
#                         print(f"❌ JSON error: {exc}")
#                         continue

#                     if label == "primary" and msg.get("robot_id") != ROBOT_ID:
#                         continue

#                     text = msg.get("text", "")
#                     if text:
#                         print(f"📝 {text}")
#                     audio_b64 = msg.get("audio")
#                     if audio_b64:
#                         await enqueue_clip(audio_b64, text)
#         except Exception as exc:
#             retry += 1
#             delay = min(60, 2 ** retry)
#             print(f"⚠️  {label} socket error: {exc} – reconnecting in {delay}s …")
#             await asyncio.sleep(delay)


# auth_primary = lambda: _generic_socket(PRIMARY_WS_URI, "primary")
# auth_lecture = lambda: _generic_socket(LECTURE_WS_URI, "lecture")

# # ---------------------------------------------------------------------------
# # MAIN
# # ---------------------------------------------------------------------------
# async def main() -> None:
#     print("🚀 Starting WebSocket client …  (Ctrl‑C to quit)")
#     await start_http_server()
#     await asyncio.gather(playback_worker(), auth_primary(), auth_lecture())


# if __name__ == "__main__":
#     try:
#         asyncio.run(main())
#     except KeyboardInterrupt:
#         print("\n👋 Bye!")