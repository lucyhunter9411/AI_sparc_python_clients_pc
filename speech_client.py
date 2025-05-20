import pyaudio
import numpy as np
import webrtcvad
import wave
import io
import base64
import asyncio
import json
import websockets
import sys
import time
import select
import aiohttp  # Add this import at the top of your file
import datetime  # Add this import at the top of your file
from tzlocal import get_localzone  # Import get_localzone from tzlocal

# Constants for the WebSocket connection and audio processing
ROBOT_ID = "robot_1"
WEBSOCKET_URI = "wss://app-ragbackend-dev-wus-001.azurewebsites.net/ws/before/lecture"
# WEBSOCKET_URI = "ws://localhost:8000/ws/before/lecture"

# Voice Activity Detection (VAD) and audio settings
VAD_MODE = 3  # 0 = very sensitive, 3 = least sensitive
SAMPLE_RATE = 16000  # Sample rate in Hz
FRAME_DURATION = 30  # Frame duration in ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)  # Frame size in samples
CHANNELS = 1  # Mono audio
SAMPLE_WIDTH = 2  # 16-bit samples (int16)
VOLUME_THRESHOLD = 3000  # Volume threshold for speech detection
SILENCE_THRESHOLD = 66  # Number of silent frames before stopping recording

# Initialize the WebRTC VAD
vad = webrtcvad.Vad(VAD_MODE)

stash = []  # Global list to store audio data

def is_speech(audio_bytes):
    """Check if audio contains speech using WebRTC VAD."""
    return vad.is_speech(audio_bytes, SAMPLE_RATE)

async def record_audio(websocket):
    """Record audio with voice activity detection (VAD)."""
    buffer = []  # Buffer to store audio frames
    recording = False  # Flag to indicate if recording is active
    silence_count = 0  # Counter for silent frames
    start_speaking = False  # Flag to track if user srat to speaking

    # Initialize PyAudio and open a stream for recording
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=CHANNELS, rate=SAMPLE_RATE,
                    input=True, frames_per_buffer=FRAME_SIZE)

    print("🎙️ Waiting for speech...")
    last_status_time = time.time()  # Track the last time a status was sent
    start_waiting_time = time.time()  # Track the start of the waiting period

    try:
        while True:
            global spoken_text
            # Read a chunk of audio data from the stream
            audio_chunk = stream.read(FRAME_SIZE, exception_on_overflow=False)
            mono_audio = np.frombuffer(audio_chunk, dtype=np.int16)  # Convert to numpy array
            audio_bytes = mono_audio.tobytes()  # Convert to bytes

            # Check for speech using VAD and volume threshold
            if vad.is_speech(audio_bytes, SAMPLE_RATE) and np.max(np.abs(mono_audio)) >= VOLUME_THRESHOLD:
                if not recording:
                    print("🗣️ Speech detected!")
                buffer.append(audio_bytes)  # Add audio to buffer
                recording = True
                silence_count = 0  # Reset silence counter

                print("Current Audio Volume:", np.max(np.abs(mono_audio)))
                start_waiting_time = time.time()  # Reset waiting time on speech detection
            elif recording:
                buffer.append(audio_bytes)  # Continue recording during silence
                silence_count += 1
            else:
                # Check if waiting time exceeds 5 seconds
                if time.time() - start_waiting_time > 5:
                    break  # Exit the loop to trigger reconnection

                # Send a "waiting" status to the WebSocket every 10 seconds
                if time.time() - last_status_time >= 10:
                    await websocket.send(json.dumps({"robot_id": ROBOT_ID, "status": "waiting"}))
                    last_status_time = time.time()

    finally:
        # Clean up the audio stream
        stream.stop_stream()
        stream.close()
        p.terminate()

    if not buffer:
        return None  # Return None if no audio was recorded

    # Convert the recorded audio to WAV format
    byte_stream = io.BytesIO()
    wav_writer = wave.Wave_write(byte_stream)
    wav_writer.setnchannels(CHANNELS)
    wav_writer.setsampwidth(SAMPLE_WIDTH)
    wav_writer.setframerate(SAMPLE_RATE)
    wav_writer.writeframes(b''.join(buffer))

    # Return the audio as a base64-encoded string
    return base64.b64encode(byte_stream.getvalue()).decode("utf-8")

async def get_backend_choice():
    """Select a stt model for transcription."""
    print("1. Whisper")
    print("2. gpt-4o-transcribe")
    print("3. gpt-4o-mini-transcribe")
    print("4. Google-EN")
    print("5. Google-HI")
    print("6. Google-TE")

    options = {
        "1": "whisper-1",
        "2": "gpt-4o-transcribe",
        "3": "gpt-4o-mini-transcribe",
        "4": "en-US",
        "5": "hi-IN",
        "6": "te-IN"
    }

    while True:
        # Use asyncio to handle blocking input
        loop = asyncio.get_event_loop()
        choice = await loop.run_in_executor(None, input, "Enter number (1-6): ")
        choice = choice.strip()
        
        if choice in options:
            return options[choice]  # Return the selected backend
        print("Invalid selection. Try again.")

def push(audio_data):
    """Save audio data to the stash."""
    stash.append(audio_data)

async def process_audio_and_send():
    """Handle WebSocket connection and continuously send audio data."""
    retry_count = 0
    max_retries = 5
    retry_delay = 6
    backend_choice = None

    while True:
        try:
            async with websockets.connect(
                WEBSOCKET_URI,
                ping_interval=30,
                ping_timeout=20,
                close_timeout=30,
                max_size=10_000_000
            ) as websocket:
                await websocket.send(json.dumps({
                    "robot_id": ROBOT_ID
                }))
                retry_count = 0
                if backend_choice is None:
                    backend_choice = await get_backend_choice()

                # 🔁 Stay in conversation loop
                while True:
                    audio_base64 = await record_audio(websocket)

                    if audio_base64:
                        push(audio_base64)
                        for audio in stash[:]:
                            local_time = datetime.datetime.now().isoformat()
                            local_region = str(get_localzone())
                            await websocket.send(json.dumps({
                                "robot_id": ROBOT_ID,
                                "audio": audio,
                                "backend": backend_choice,
                                "spoken_text": spoken_text,
                                "local_time": local_time,
                                "local_region": local_region
                            }))
                            stash.remove(audio)
                            response = await websocket.recv()
                            print(f"📝 Transcription: {response}\n")
                    else:
                        print("🕐 No speech detected, waiting for user...")
                        await asyncio.sleep(1)  # short pause before next VAD cycle
                        continue  # 🔄 Go to next recording session

        except (websockets.exceptions.ConnectionClosedError, websockets.exceptions.InvalidMessage) as e:
            print(f"Connection error: {e}. Retrying...")
            retry_count += 1
            if retry_count <= max_retries:
                await asyncio.sleep(5)
            else:
                print("Max retries reached. Exiting.")
                break
        except OSError as e:
            print(f"OS error: {e}. Retrying...")
            retry_count += 1
            if retry_count <= max_retries:
                await asyncio.sleep(5)
            else:
                print("Max retries reached. Exiting.")
                break

async def main():
    """Main function to start the audio processing and sending loop."""
    await process_audio_and_send()

# Run the async loop
asyncio.run(main())