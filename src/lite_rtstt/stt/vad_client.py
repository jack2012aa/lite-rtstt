from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass
import logging
import queue
import threading

import numpy
import webrtcvad
from atomicx import AtomicBool
from silero_vad import load_silero_vad, get_speech_timestamps

from lite_rtstt.atomic.counter import Counter
from lite_rtstt.stt.audiobuffer import AudioBuffer
from lite_rtstt.stt.config import STTConfig


class VADClient(ABC):
    """An VAD service that tells you whether an audio chunk has human voice."""

    @abstractmethod
    def start(self):
        """Start the service."""
        pass

    @abstractmethod
    def close(self):
        """Close the service."""
        pass

    @abstractmethod
    async def is_active(self, audio_buffer: AudioBuffer) -> bool:
        """Is the given audio active?
        Args:
            audio_buffer (numpy.ndarray): Audio.
        Returns:
            bool: Is the given audio active?
        """
        pass


class MockVADClient(VADClient):

    def __init__(self) -> None:
        self.__results = asyncio.Queue()
        self.__started = False
        self.__closed = False

    async def append_result(self, *results: bool) -> None:
        for result in results:
            await self.__results.put(result)

    def start(self) -> None:
        self.__started = True

    def close(self) -> None:
        self.__closed = True

    async def is_active(self, audio_buffer: AudioBuffer) -> bool:
        if not self.__started:
            raise RuntimeError("MockVADClient is not ready.")
        if self.__closed:
            raise RuntimeError("MockVADClient is closed.")
        return await self.__results.get()

class SileroClient(VADClient):


    @dataclass
    class SileroPoolWork:
        audio: numpy.ndarray
        loop: asyncio.AbstractEventLoop
        future: asyncio.Future[bool]

    def __init__(self, config: STTConfig) -> None:
        """A VADClient that uses a Silero VAD pool for detection.

        Args:
            config (STTConfig): STT config.
        """

        self.started = False
        self.__closed = AtomicBool(False)
        self.__pool = [threading.Thread(target=self.__worker, daemon=True) for _ in range(config.vad_threads)]
        self.__inputs = queue.Queue()
        self.__input_semaphore = threading.Semaphore(0)
        self.__ready_threads = Counter()

    def __worker(self) -> None:
        """Load the model and start listening for work."""

        model = load_silero_vad()
        self.__ready_threads.increment()
        while not self.__closed.load():
            self.__input_semaphore.acquire()
            try:
                work = self.__inputs.get()
            except queue.ShutDown:
                break
            try:
                if not isinstance(work, SileroClient.SileroPoolWork):
                    raise RuntimeError("Silero worker received an invalid work.")
                result = get_speech_timestamps(work.audio, model)
                work.loop.call_soon_threadsafe(work.future.set_result, len(result) > 0)
            except Exception as e:
                work.loop.call_soon_threadsafe(work.future.set_exception, e)

    def start(self) -> None:
        """Start the pool. You should call this method before using the pool."""
        if self.started:
            return
        for thread in self.__pool:
            thread.start()
        logging.debug("Waiting for silero pool to be ready.")
        self.__ready_threads.wait_for(len(self.__pool))
        self.started = True
        logging.debug("Silero pool started.")

    def close(self) -> None:
        if not self.__closed.load():
            self.__closed.store(True)
            self.__inputs.shutdown(True)
            for _ in range(len(self.__pool)):
                self.__input_semaphore.release()
            for thread in self.__pool:
                thread.join()
            self.__pool.clear()

    async def is_active(self, audio_buffer: AudioBuffer) -> bool:
        if not self.started:
            raise RuntimeError("Silero pool is not ready.")
        if self.__closed.load():
            raise RuntimeError("Silero pool is closed.")
        work = SileroClient.SileroPoolWork(
            audio_buffer.to_float32_ndarray(),
            asyncio.get_running_loop(),
            asyncio.get_running_loop().create_future(),
        )
        self.__inputs.put(work)
        self.__input_semaphore.release()
        return await work.future


class WebRTCClient(VADClient):

    def __init__(self, config: STTConfig):
        """A light-weighted VAD based on web rtc VAD"""
        self.__vad = webrtcvad.Vad(config.aggresiveness)
        self.__sample_rate = config.sample_rate
        self.__started = False
        self.__closed = False

    def start(self):
        self.__started = True

    def close(self):
        self.__closed = True

    async def is_active(self, audio_buffer: AudioBuffer) -> bool:
        if not self.__started:
            raise RuntimeError("WebRTCClient is not ready.")
        if self.__closed:
            raise RuntimeError("WebRTCClient is closed.")
        return self.__vad.is_speech(audio_buffer.to_bytes(), self.__sample_rate)
