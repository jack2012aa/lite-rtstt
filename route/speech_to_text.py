"""Speech to text route."""

import json
import asyncio
import logging
from dataclasses import asdict

import numpy
from fastapi import APIRouter, WebSocket

from service.stt_client import STTClient
from data_structure.message import (
    SpeechTranscriptMessage,
    TranscriptionStartMessage,
)

__all__ = ["stt_router"]


stt_router = APIRouter()


@stt_router.websocket("/stt")
async def speech_to_text(websocket: WebSocket) -> None:
    """Receive audio bytes and return text.

    Args:
        websocket (WebSocket): a WebSocket connection.
    """

    # TODO Check authentication.
    await websocket.accept()
    logging.info("WebSocket connection established.")

    loop = asyncio.get_event_loop()

    def on_text(text: str) -> None:
        """Callback when text is transcribed."""
        if (text == ""):
            return
        message_structure = SpeechTranscriptMessage(text=text)
        message = json.dumps(asdict(message_structure))
        asyncio.run_coroutine_threadsafe(websocket.send_json(message), loop)

    def on_transcript_start() -> None:
        """Callback when transcription starts."""
        message = json.dumps(asdict(TranscriptionStartMessage()))
        asyncio.run_coroutine_threadsafe(websocket.send_json(message), loop)

    sst_client = STTClient(on_text, on_transcript_start)
    sst_client.start()

    try: 
        while True:
            data = await websocket.receive_bytes()
            int16_array = numpy.frombuffer(data, dtype=numpy.int16)
            sst_client.feed(int16_array)
    except Exception as e:
        sst_client.stop()