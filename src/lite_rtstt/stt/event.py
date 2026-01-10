import asyncio
import queue
from abc import ABC, abstractmethod
from typing import Any, Coroutine


class STTEvent(ABC):
    pass

class TextEvent(STTEvent):

    def __init__(self, text: str):
        self.text = text

    def text(self) -> str:
        return self.text


class StartSpeakingEvent(STTEvent):
    pass


class StopSpeakingEvent(STTEvent):
    pass


class EventFactory:

    __START_SPEAKING_EVENT = StartSpeakingEvent()
    __STOP_SPEAKING_EVENT = StopSpeakingEvent()

    @staticmethod
    def start_speaking_event() -> StartSpeakingEvent:
        return EventFactory.__START_SPEAKING_EVENT

    @staticmethod
    def stop_speaking_event() -> StopSpeakingEvent:
        return EventFactory.__STOP_SPEAKING_EVENT

    @staticmethod
    def text_event(text: str) -> TextEvent:
        return TextEvent(text)


class STTEventQueue(ABC):

    @abstractmethod
    async def put(self, event: STTEvent):
        pass

    @abstractmethod
    async def get(self) -> STTEvent:
        pass

    @abstractmethod
    async def close(self):
        pass


class SimpleSTTEventQueue(STTEventQueue):

    def __init__(self):
        self.__queue = asyncio.Queue()

    async def put(self, event: STTEvent):
        await self.__queue.put(event)

    async def get(self) -> STTEvent:
        return await self.__queue.get()

    async def close(self):
        self.__queue.shutdown()