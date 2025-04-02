import argparse
import logging

import uvicorn
from fastapi import FastAPI

from service.route import stt_router
from service.whisper_pool import WhisperPool

if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)
    
    parser = argparse.ArgumentParser(description="STT Server")
    parser.add_argument("--port", type=int, help="Port number for the server", default=8766)
    parser.add_argument("--pool_size", type=int, help="Size of the whisper pool", default=1)
    parser.add_argument("--model_type", type=str, help="Type of the whisper model", default="tiny.en")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA for GPU acceleration")
    args = parser.parse_args()

    try:
        WhisperPool.get_instance(args.pool_size, args.model_type, args.cuda).start()
    except FileNotFoundError as e:
        exit(1)

    app = FastAPI()
    app.include_router(stt_router)
    uvicorn.run(app, port=args.port, host="0.0.0.0")
    