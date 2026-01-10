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


class SileroClient(VADClient):


    @dataclass
    class SileroPoolWork:
        audio: numpy.ndarray
        loop: asyncio.AbstractEventLoop
        is_done: asyncio.Semaphore
        result: bool

    __TOMBSTONE = SileroPoolWork(None, None, None, False)

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
        self.__ready_threads = threading.Semaphore(-config.vad_threads + 1)

    def __worker(self) -> None:
        """Load the model and start listening for work."""

        model = load_silero_vad()
        self.__ready_threads.release()
        while not self.__closed.load():
            self.__input_semaphore.acquire()
            work = self.__inputs.get()
            if work == SileroClient.__TOMBSTONE:
                break
            if not isinstance(work, SileroClient.SileroPoolWork):
                logging.error("Silero worker received an invalid work.", stack_info=True)
                exit(1)
            result = get_speech_timestamps(work.audio, model)
            work.result = len(result) > 0
            work.loop.call_soon_threadsafe(work.is_done.release)

    def start(self) -> None:
        """Start the pool. You should call this method before using the pool."""
        if self.started:
            return
        for thread in self.__pool:
            thread.start()
        logging.debug("Waiting for silero pool to be ready.")
        self.__ready_threads.acquire()
        self.started = True
        logging.debug("Silero pool started.")

    def close(self) -> None:
        self.__closed.store(True)
        self.__inputs.put(SileroClient.__TOMBSTONE)
        self.__input_semaphore.release()
        for thread in self.__pool:
            thread.join()

    async def is_active(self, audio_buffer: AudioBuffer) -> bool:
        if not self.started:
            logging.error("Silero pool is not started yet.", stack_info=True)
            exit(1)
        work = SileroClient.SileroPoolWork(
            audio_buffer.to_ndarray(),
            asyncio.get_running_loop(),
            asyncio.Semaphore(0),
            False
        )
        self.__inputs.put(work)
        self.__input_semaphore.release()
        await work.is_done.acquire()
        return work.result


class WebRTCClient(VADClient):

    def __init__(self, config: STTConfig):
        """A light-weighted VAD based on web rtc VAD"""
        self.__vad = webrtcvad.Vad(config.aggresiveness)
        self.__sample_rate = config.sample_rate

    def start(self):
        pass

    def close(self):
        self.__vad = None

    async def is_active(self, audio_buffer: AudioBuffer) -> bool:
        return self.__vad.is_speech(audio_buffer.to_bytes(), self.__sample_rate)
