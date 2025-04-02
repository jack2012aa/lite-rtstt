import logging
import queue
import threading

from silero_vad import *

from service.structure import TranscriptionWork, SileroIsSpeech


class SileroPool:

    __instance = None

    @staticmethod
    def get_instance(size: int = 1) -> "SileroPool":
        """Get the singleton instance of the silero pool.

        Args:
            size (int, optional): Size of the pool. Defaults to 1.

        Returns:
            SileroPool: The instance.
        """
        if SileroPool.__instance is None:
            SileroPool.__instance = SileroPool(size)
        return SileroPool.__instance


    def __init__(self, size: int = 1) -> None:
        """Creteate a silero pool. You shouldn't call this constructor. Use get_instance() instead.

        Args:
            size (int, optional): Size of the pool. Defaults to 1.
        """

        self.started = False
        self.__pool = [
            threading.Thread(target=self.__worker, daemon=True)
            for _ in range(size)
        ]
        self.__inputs = queue.Queue()
        self.__ready_threads = threading.Semaphore(-size + 1)
        logging.debug("Silero pool created.")

    def __worker(self) -> None:
        """Load the model and start listening for work."""

        model = load_silero_vad()
        self.__ready_threads.release()
        while True:
            work = self.__inputs.get()
            if not isinstance(work, TranscriptionWork):
                logging.warning("Silero worker received an invalid work.")
                continue
            result = get_speech_timestamps(work.audio, model)
            if len(result) > 0:
                work.return_queue.put(SileroIsSpeech(work.audio))

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

    def add_work(self, work: TranscriptionWork) -> None:
        """Add work to the pool.

        Args:
            work (TranscriptionWork): Work to add.
        """
        if not self.started:
            raise RuntimeError("Silero pool is not started yet.")
        self.__inputs.put(work)