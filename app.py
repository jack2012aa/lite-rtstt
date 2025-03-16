import argparse

import uvicorn
from fastapi import FastAPI
from route.speech_to_text import stt_router

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="STT Server")
    parser.add_argument(
        "--port", type=int, help="Port number for the server", default=8766
    )
    args = parser.parse_args()

    app = FastAPI()
    app.include_router(stt_router)
    uvicorn.run(app, port=args.port)
    