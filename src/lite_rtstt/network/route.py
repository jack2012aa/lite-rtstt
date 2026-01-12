"""Speech to text route."""
import base64
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

        async def handle_event():
            try:
                while True:
                    event = await queue.get()
                    if event is None:
                        break
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
                message = await websocket.receive_json()
                if message["type"] == "audio chunk":
                    bytes = base64.b64decode(message["data"])
                    await rtstt_client.feed(connection_id, bytes)
                elif message["type"] == "EOF":
                    # Since ThreeLayerRTSTTClient#feed is blocking, closing here is safe
                    break
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logging.error(e, stack_info=True)
        finally:
            rtstt_client.disconnect(connection_id)
            await queue.put(None)
            await task
            await websocket.close()

    return router