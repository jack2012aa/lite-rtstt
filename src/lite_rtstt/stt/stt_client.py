import asyncio
import threading
import queue
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import whisper
from atomicx.atomicx import AtomicBool

from lite_rtstt.stt.audio_buffer import AudioBuffer
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


class MockSTTClient(STTClient):
    """Mock STT client."""

    def __init__(self):
        self.__started = False
        self.__closed = False
        self.__results = asyncio.Queue()

    async def append_results(self, *results: str):
        for result in results:
            await self.__results.put(result)

    def start(self):
        self.__started = True

    def close(self):
        self.__closed = True

    async def transcribe(self, audio_buffer: AudioBuffer) -> str:
        if not self.__started:
            raise RuntimeError("MockSTTClient is not started.")
        if self.__closed:
            raise RuntimeError("MockSTTClient is closed.")
        return await self.__results.get()


class WhisperClient(STTClient):

    __silence_padding = np.zeros(8000, dtype=np.float32)

    @dataclass
    class Work:
        audio_array: np.ndarray
        loop: asyncio.AbstractEventLoop
        future: asyncio.Future[str]

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
            try:
                work = self.__inputs.get()
            except queue.ShutDown:
                break
            try:
                if not isinstance(work, WhisperClient.Work):
                    raise RuntimeError("Whisper worker received an invalid work.")
                result = self.__model.transcribe(audio=work.audio_array)
                if result.get("text", None) is not None:
                    work.loop.call_soon_threadsafe(work.future.set_result, result["text"])
                else:
                    work.loop.call_soon_threadsafe(work.future.set_result, result[""])
            except Exception as e:
                work.loop.call_soon_threadsafe(work.future.set_exception, e)

    def start(self):
        if self.started:
            return
        logging.debug("Waiting for whisper models to be loaded.")
        self.__model = whisper.load_model(self.__model_size, download_root=self.__download_root)
        self.__whisper_thread.start()
        self.started = True
        logging.debug("Whisper pool starts.")

    def close(self):
        if not self.__closed.load():
            self.__closed.store(True)
            self.__inputs.shutdown(True)
            self.__input_semaphore.release()
            self.__whisper_thread.join()

    async def transcribe(self, audio_buffer: AudioBuffer) -> str:
        if not self.started:
            raise RuntimeError("Whisper is not ready.")
        if self.__closed.load():
            raise RuntimeError("Whisper is closed.")
        raw_audio = audio_buffer.to_float32_ndarray()
        padded_audio = np.concatenate([self.__silence_padding, raw_audio])
        work = WhisperClient.Work(
            padded_audio,
            asyncio.get_running_loop(),
            asyncio.Future())
        self.__inputs.put(work)
        self.__input_semaphore.release()
        return await work.future
