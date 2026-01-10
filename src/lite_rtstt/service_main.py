import argparse
import json
import os
import logging
from dataclasses import replace

from fastapi import FastAPI

from lite_rtstt.stt.config import STTConfig


def load_service_config(base_dir: str) -> STTConfig:
    file_name = "stt_config.json"
    path = os.path.join(base_dir, file_name)
    default_config = STTConfig(vad_threads=4, whisper_model="base", duration_time_ms=1200, aggresiveness=3)
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

    app = FastAPI()
    app.include_router(stt_router)
    uvicorn.run(app, port=args.port, host="0.0.0.0")
    