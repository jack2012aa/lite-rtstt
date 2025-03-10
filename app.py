import logging

import uvicorn
from fastapi import FastAPI
from route.speech_to_text import stt_router

if __name__ == "__main__":
    app = FastAPI()
    app.include_router(stt_router)
    uvicorn.run(app, port="8766")
