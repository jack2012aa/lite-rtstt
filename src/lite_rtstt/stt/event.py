import queue
from abc import ABC, abstractmethod


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


class STTEventQueue(ABC):

    @abstractmethod
    def put(self, event: STTEvent):
        pass

    @abstractmethod
    def get(self) -> STTEvent:
        pass


class EventFactory:

    __START_SPEAKING_EVENT = StartSpeakingEvent()
    __STOP_SPEAKING_EVENT = StopSpeakingEvent()

    @staticmethod
    def get_start_speaking_event() -> StartSpeakingEvent:
        return EventFactory.__START_SPEAKING_EVENT

    @staticmethod
    def get_stop_speaking_event() -> StopSpeakingEvent:
        return EventFactory.__STOP_SPEAKING_EVENT

    @staticmethod
    def get_text_event(text: str) -> TextEvent:
        return TextEvent(text)


class SimpleSTTEventQueue(STTEventQueue):

    def __init__(self):
        self.queue = queue.Queue()

    def put(self, event: STTEvent):
        self.queue.put(event)

    def get(self) -> STTEvent:
        return self.queue.get()