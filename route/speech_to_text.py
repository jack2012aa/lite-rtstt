"""Speech to text route."""

import json
import asyncio
import logging
from dataclasses import asdict

import numpy
from fastapi import APIRouter, WebSocket

from service.stt_client import STTClient
from data_structure.message import *

__all__ = ["stt_router"]


stt_router = APIRouter()


@stt_router.websocket("/stt")
async def speech_to_text(websocket: WebSocket) -> None:
    """Receive audio bytes and return text.

    Args:
        websocket (WebSocket): a WebSocket connection.
    """

    # TODO Check authentication.
    logging.info("WebSocket connection established.")

    loop = asyncio.get_event_loop()

    def on_text(text: str) -> None:
        """Callback when text is transcribed."""
        if text == "":
            return
        message_structure = SpeechTranscriptMessage(transcript=text)
        message = asdict(message_structure)
        asyncio.run_coroutine_threadsafe(websocket.send_json(message), loop)

    mono_start_speaking_message = asdict(StartSpeakingMessage())
    mono_stop_speaking_message = asdict(StopSpeakingMessage())

    def on_start_speaking() -> None:
        """Callback when recording starts."""
        asyncio.run_coroutine_threadsafe(
            websocket.send_json(mono_start_speaking_message), loop
        )

    def on_stop_speaking() -> None:
        """Callback when recording stops."""
        asyncio.run_coroutine_threadsafe(
            websocket.send_json(mono_stop_speaking_message), loop
        )

    sst_client = STTClient(on_text, on_start_speaking, on_stop_speaking)
    sst_client.start()
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_bytes()
            int16_array = numpy.frombuffer(data, dtype=numpy.int16)
            sst_client.feed(int16_array)
    except Exception as e:
        sst_client.stop()
