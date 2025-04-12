__all__ = ["WhisperPool"]

import os
import threading
import queue
import logging
import multiprocessing
import uuid

import numpy as np
import whisper

from service.structure import TranscriptionWork, OnTextMessage, OnTextMessageWithId, ProcessTranscriptionWork


def worker(
    input_queue: multiprocessing.Queue, 
    return_queue: multiprocessing.Queue, 
    is_ready: threading.Event,
    model_type: str, 
    cuda: bool, 
    warmup_audio: np.ndarray
):
    """Worker function for each process in the pool.

    Args:
        input_queue (multiprocessing.Queue): An queue for input messages.
        return_queue (multiprocessing.Queue): An queue for return messages.
        is_ready (multiprocessing.Event): An event to signal when the model is ready.
        model_type (str): Whisper model type.
        cuda (bool): Using cuda or not.
        warmup_audio (np.ndarray): Warmup audio for the model.
    """

    if cuda:
        model = whisper.load_model(model_type, device="cuda")
    else:
        model = whisper.load_model(model_type)
    # Warmup the model
    model.transcribe(warmup_audio)
    is_ready.set()
    while True:
        work = input_queue.get()
        if not isinstance(work, ProcessTranscriptionWork):
            logging.warning("Whisper worker received an invalid work.")
            continue
        result = model.transcribe(work.audio)
        if result.get("text", None) is not None:
            return_queue.put(OnTextMessageWithId(result["text"], work.id))


class WhisperPool:

    __instance = None

    @staticmethod
    def get_instance(size: int = 1, model_type: str = "tiny.en", cuda: bool = False) -> "WhisperPool":
        """Get the singleton instance of the whisper pool.

        Args:
            size (int, optional): Size of the pool. Defaults to 1.
            model_type (str, optional): Type of the Whipser model. Defaults to "tiny.en".
            cuda (bool, optional): Use cuda. Defaults to False.

        Returns:
            WhisperPool: The instance.
        """

        if WhisperPool.__instance is None:
            WhisperPool.__instance = WhisperPool(size, model_type, cuda)
        return WhisperPool.__instance
    

    def __init__(self, size: int = 1, model_type: str = "tiny.en", cuda:bool = False) -> None:
        """Create a whisper pool. You shouldn't call this constructure. Use get_instance() instead.

        Args:
            size (int, optional): Size of the pool. Defaults to 1.
            model_type (str, optional): Type of the Whipser model. Defaults to "tiny.en".
            cuda (bool, optional): Use cuda. Defaults to False.
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

        self.__input_queue = multiprocessing.Queue()
        self.__return_queue = multiprocessing.Queue()
        self.__request_dict = {}
        self.__request_dict_lock = threading.Lock()
        self.__is_ready = [multiprocessing.Event() for _ in range(size)]
        self.__pool = [multiprocessing.Process(
            target=worker, 
            args=[
                self.__input_queue, 
                self.__return_queue, 
                self.__is_ready[i], 
                model_type, 
                cuda, 
                self.__warmup_audio
            ], 
            daemon=True
        ) for i in range(size)]
        self.__handle_return_thread = threading.Thread(target=self.handle_return, daemon=True)

        # self.__inputs = queue.Queue()
        # self.__model_type = model_type
        # self.__size = size
        # self.__cuda = cuda
        # # Wait for every model to be loaded
        # self.__ready_threads = threading.Semaphore()
        # logging.debug("Whisper pool created.")
        # self.__pool = [
        #     threading.Thread(target=self.__worker, daemon=True)
        #     for _ in range(size)
        # ]

    # def __worker(self) -> None:
    #     """Load the model and start listening for work."""

    #     if self.__cuda:
    #         model = whisper.load_model(self.__model_type, device="cuda")
    #     else:
    #         model = whisper.load_model(self.__model_type)
    #     # Warmup the model
    #     model.transcribe(self.__warmup_audio)
    #     self.__ready_threads.release()
    #     while True:
    #         work = self.__inputs.get()
    #         if not isinstance(work, TranscriptionWork):
    #             logging.warning("Whisper worker received an invalid work.")
    #             continue
    #         result = model.transcribe(work.audio)
    #         if result.get("text", None) is not None:
    #             work.return_queue.put(OnTextMessage(result["text"]))

    def start(self) -> None:
        """Start the pool."""

        if self.started:
            return
        for process in self.__pool:
            process.start()
        logging.debug("Waiting for whisper models to be loaded.")
        for event in self.__is_ready:
            event.wait()
        self.__handle_return_thread.start()

        # for thread in self.__pool:
        #     thread.start()
        # logging.debug("Waiting for whisper models to be loaded.")
        # for _ in range(self.__size):
        #     self.__ready_threads.acquire()
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
        
        id = uuid.uuid4()
        work_with_id = ProcessTranscriptionWork(
            audio=work.audio, id=id
        )
        with self.__request_dict_lock:
            self.__request_dict[id] = work.return_queue
        self.__input_queue.put(work_with_id)

        # self.__inputs.put(work)

    def handle_return(self):
        """Handle the return queue and put the result in the return queue of the work."""

        while True:
            result = self.__return_queue.get()
            if not isinstance(result, OnTextMessageWithId):
                logging.warning("Whisper pool received an invalid result.")
                continue
            with self.__request_dict_lock:
                if result.id not in self.__request_dict:
                    logging.warning("Whisper pool received a result with an unknown ID.")
                    continue
                return_queue = self.__request_dict[result.id]
                del self.__request_dict[result.id]
            return_queue.put(OnTextMessage(result.text))
