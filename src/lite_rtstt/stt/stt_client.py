import asyncio
import threading
import queue
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import whisper
from atomicx.atomicx import AtomicBool

from lite_rtstt.stt.audiobuffer import AudioBuffer
from lite_rtstt.stt.config import STTConfig


class STTClient(ABC):
    """A speech to text service."""

    @abstractmethod
    async def transcribe(self, audio_buffer: AudioBuffer) -> str:
        """Transcribe the audio."""
        pass

    @abstractmethod
    def start(self):
        """Start the STT service."""
        pass

    @abstractmethod
    def close(self):
        """Close the STT service."""
        pass


class WhisperClient(STTClient):


    @dataclass
    class Work:
        audio_array: np.ndarray
        loop: asyncio.AbstractEventLoop
        is_done: asyncio.Semaphore
        result: str

    __TOMBSTONE = Work(None, None, None, None)

    def __init__(self, config: STTConfig, download_root: str) -> None:
        """A Whisper-based STT client."""

        self.started = False
        self.__closed = AtomicBool(False)
        self.__inputs = queue.Queue()
        self.__input_semaphore = threading.Semaphore(0)
        self.__model: whisper.Whisper = None
        self.__model_size = config.whisper_model
        self.__whisper_thread = threading.Thread(target=self.__worker, daemon=True)
        self.__download_root = download_root

    def __worker(self):
        while not self.__closed.load():
            self.__input_semaphore.acquire()
            work = self.__inputs.get()
            if work == WhisperClient.__TOMBSTONE:
                break
            if not isinstance(work, WhisperClient.Work):
                logging.error("Whisper worker received an invalid work.", stack_info=True)
                exit(1)
            result = self.__model.transcribe(audio=work.audio_array)
            if result.get("text", None) is not None:
                work.result = result["text"]
            work.loop.call_soon_threadsafe(work.is_done.release)

    def start(self):
        if self.started:
            return
        logging.debug("Waiting for whisper models to be loaded.")
        self.__model = whisper.load_model(self.__model_size, download_root=self.__download_root)
        self.__whisper_thread.start()
        self.started = True
        logging.debug("Whisper pool starts.")

    def close(self):
        self.__closed.store(True)
        self.__inputs.put(WhisperClient.__TOMBSTONE)
        self.__input_semaphore.release()
        self.__whisper_thread.join()

    async def transcribe(self, audio_buffer: AudioBuffer) -> str:
        work = WhisperClient.Work(audio_buffer.to_ndarray(), asyncio.get_running_loop(), asyncio.Semaphore(0), "")
        self.__inputs.put(work)
        self.__input_semaphore.release()
        await work.is_done.acquire()
        return work.result
