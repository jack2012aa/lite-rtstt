import argparse
import asyncio
import json
import os
import logging
import sys
from dataclasses import replace

import pyaudio
import uvicorn
import websockets
from fastapi import FastAPI

from lite_rtstt.network.route import create_router
from lite_rtstt.stt.config import STTConfig
from lite_rtstt.stt.rtstt_client import ThreeLayerRTSTTClient
from lite_rtstt.stt.stt_client import WhisperClient
from lite_rtstt.stt.vad_client import WebRTCClient, SileroClient


def load_service_config(base_dir: str) -> STTConfig:
    file_name = "stt_config.json"
    path = os.path.join(base_dir, file_name)
    default_config = STTConfig.default()
    if not os.path.exists(path):
        return default_config
    with open(path, "r") as f:
        content = json.load(f)
        config = replace(default_config, **content)
        return config

def run_server(args):
    if args.debug:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    # Load environmental variables
    DATA_DIR = os.environ.get("SNAP_DATA", "./")
    config = load_service_config(DATA_DIR)

    rtc = WebRTCClient(config)
    silero = SileroClient(config)
    download_root = os.path.join(DATA_DIR, "whisper")
    whisper = WhisperClient(config, download_root)
    rtstt = ThreeLayerRTSTTClient(config, rtc, silero, whisper)
    rtstt.start()

    router = create_router(rtstt)
    app = FastAPI()
    app.include_router(router)
    uvicorn.run(app, host="0.0.0.0", port=8766)

async def _stream_microphone(uri: str):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 480  # 30ms @ 16kHz

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    async with websockets.connect(uri) as websocket:
        async def receive():
            try:
                async for message in websocket:
                    data = json.loads(message)
                    if data["type"] == "text":
                        print(f"\rUser: {data['text']}")
                        print("> ", end="", flush=True)
                    elif data["type"] == "start speaking":
                        print(f"\r[Listening...]", end="", flush=True)
                    elif data["type"] == "stop speaking":
                        print(f"\r[Thinking...]", end="", flush=True)
            except websockets.exceptions.ConnectionClosed:
                print("\nServer disconnected.")

        recv_task = asyncio.create_task(receive())

        try:
            while True:
                data = stream.read(CHUNK, exception_on_overflow=False)
                await websocket.send(data)
                await asyncio.sleep(0)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            await websocket.close()
            recv_task.cancel()

def run_live(args):
    target_url = args.url or "ws://localhost:8766/rtstt"
    try:
        asyncio.run(_stream_microphone(target_url))
    except KeyboardInterrupt:
        pass

def main():
    parser = argparse.ArgumentParser(description="Lite Real-time Speech to Text")

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Available commands",
    )

    server_parser = subparsers.add_parser("run", help="Start the RTSTT server")
    server_parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    server_parser.set_defaults(func=run_server)

    live_parser = subparsers.add_parser("live", help="Transcribe from microphone")
    live_parser.add_argument("--url", type=str, help="Server to connect to")
    live_parser.set_defaults(func=run_live)

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)

if __name__ == "__main__":
    main()