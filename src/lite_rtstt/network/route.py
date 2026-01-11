"""Speech to text route."""

from datetime import datetime
import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from lite_rtstt.stt.event import StartSpeakingEvent, StopSpeakingEvent, TextEvent
from lite_rtstt.stt.rtstt_client import RTSTTClient


def create_router(rtstt_client: RTSTTClient) -> APIRouter:
    router = APIRouter()

    @router.websocket("/rtstt")
    async def real_time_speech_to_text(websocket: WebSocket) -> None:
        await websocket.accept()
        logging.info(f"WebSocket connection established at {datetime.now()} from host {websocket.client.host}.")
        queue, connection_id = rtstt_client.connect()
        is_closed = False

        async def handle_event():
            while not is_closed:
                try:
                    event = await queue.get()
                    if isinstance(event, StartSpeakingEvent):
                        await websocket.send_json({"type": "start speaking"})
                    elif isinstance(event, StopSpeakingEvent):
                        await websocket.send_json({"type": "stop speaking"})
                    elif isinstance(event, TextEvent):
                        await websocket.send_json({"type": "text", "text": event.text})
                    else:
                        logging.error(f"Unknown event type: {event}", stack_info=True)
                except asyncio.QueueShutDown:
                    return

        task = asyncio.create_task(handle_event())

        try:
            while True:
                data = await websocket.receive_bytes()
                await rtstt_client.feed(connection_id, data)
        except WebSocketDisconnect:
            rtstt_client.disconnect(connection_id)
        except Exception as e:
            logging.error(e, stack_info=True)
            rtstt_client.disconnect(connection_id)
            await websocket.close()

    return router