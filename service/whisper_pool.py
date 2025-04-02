__all__ = ["WhisperPool"]

import os
import threading
import queue
import logging

import numpy as np
import whisper

from service.structure import TranscriptionWork, OnTextMessage


class WhisperPool:

    __instance = None

    @staticmethod
    def get_instance(size: int = 1, model_type: str = "tiny.en") -> "WhisperPool":
        """Get the singleton instance of the whisper pool.

        Args:
            size (int, optional): Size of the pool. Defaults to 1.
            model_type (str, optional): Type of the Whipser model. Defaults to "tiny.en".

        Returns:
            WhisperPool: The instance.
        """

        if WhisperPool.__instance is None:
            WhisperPool.__instance = WhisperPool(size, model_type)
        return WhisperPool.__instance
    

    def __init__(self, size: int = 1, model_type: str = "tiny.en") -> None:
        """Create a whisper pool. You shouldn't call this constructure. Use get_instance() instead.

        Args:
            size (int, optional): Size of the pool. Defaults to 1.
            model_type (str, optional): Type of the Whipser model. Defaults to "tiny.en".
        Raises:
            FileNotFoundError: When the warmup audio file is not found.
        """


        self.started = False
        warmup_audio_path = os.path.join("service", "7s_f32.pcm")
        if not os.path.exists(warmup_audio_path):
            raise FileNotFoundError(
                f"Warmup audio file not found: {warmup_audio_path}"
            )
        with open(warmup_audio_path, "rb") as f:
            self.__warmup_audio = np.frombuffer(f.read(), dtype=np.float32)

        self.__inputs = queue.Queue()
        self.__model_type = model_type
        self.__size = size
        # Wait for every model to be loaded
        self.__ready_threads = threading.Semaphore()
        logging.debug("Whisper pool created.")
        self.__pool = [
            threading.Thread(target=self.__worker, daemon=True)
            for _ in range(size)
        ]

    def __worker(self) -> None:
        """Load the model and start listening for work."""

        model = whisper.load_model(self.__model_type)
        # Warmup the model
        model.transcribe(self.__warmup_audio)
        self.__ready_threads.release()
        while True:
            work = self.__inputs.get()
            if not isinstance(work, TranscriptionWork):
                logging.warning("Whisper worker received an invalid work.")
                continue
            result = model.transcribe(work.audio)
            if result.get("text", None) is not None:
                work.return_queue.put(OnTextMessage(result["text"]))

    def start(self) -> None:
        """Start the pool."""

        if self.started:
            return
        for thread in self.__pool:
            thread.start()
        logging.debug("Waiting for whisper models to be loaded.")
        for _ in range(self.__size):
            self.__ready_threads.acquire()
        self.started = True
        logging.debug("Whisper pool starts.")

    def add_work(self, work: TranscriptionWork) -> None:
        """Add a work to the inputs buffer.

        Args:
            work (TranscribeWork): A work structure containing the audio and the return queue.

        Raises:
            RuntimeError: When the pool is not started.
        """

        if not self.started:
            raise RuntimeError("Whisper pool is not started.")
        self.__inputs.put(work)
