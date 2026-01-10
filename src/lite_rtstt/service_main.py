import argparse
import json
import os
import logging
from dataclasses import replace

import uvicorn
from fastapi import FastAPI

from lite_rtstt.network.route import create_router
from lite_rtstt.stt.config import STTConfig
from lite_rtstt.stt.rtstt_client import ThreeLayerRTSTTClient
from lite_rtstt.stt.stt_client import WhisperClient
from lite_rtstt.stt.vad_client import WebRTCClient, SileroClient


def load_service_config(base_dir: str) -> STTConfig:
    file_name = "stt_config.json"
    path = os.path.join(base_dir, file_name)
    default_config = STTConfig(
        vad_threads=4,
        whisper_model="base",
        duration_time_ms=1200,
        aggresiveness=3,
        sample_rate=16000,
        chunk_size_ms=30,
        active_to_detection_ms=900
    )
    if not os.path.exists(path):
        return default_config
    with open(path, "r") as f:
        content = json.load(f)
        config = replace(default_config, **content)
        return config

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="STT Server")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    # Load environmental variables
    DATA_DIR = os.environ.get("SNAP_DATA", "./")
    config = load_service_config(DATA_DIR)

    rtc_vad = WebRTCClient(config)
    silero_vad = SileroClient(config)
    download_root = os.path.join(DATA_DIR, "whisper")
    whisper_vad = WhisperClient(config, download_root)
    rtstt = ThreeLayerRTSTTClient(config, rtc_vad, silero_vad, whisper_vad)
    rtstt.start()

    router = create_router(rtstt)
    app = FastAPI()
    app.include_router(router)
    uvicorn.run(app, host="0.0.0.0")
    