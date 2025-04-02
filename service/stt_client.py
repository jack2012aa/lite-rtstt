"""An AudioToTextRecorder client."""

import logging
import queue
from threading import Thread, Event
from typing import Callable

# from RealtimeSTT import AudioToTextRecorder

from service.structure import *
# from service.silero_pool import SileroPool
from service.whisper_pool import WhisperPool
from service.webrtcvad_client import WebrtcvadClient

__all__ = ["STTClient", "STTClientV2"]


# class STTClient:
#     """An AudioToRecorder client.

#     Args:
#         on_text (Callable[[str], None]): Callback when text is transcribed. \
#             It will be called in another thread, so make sure it is thread-safe.
#         on_recording_start (Callable[[None], None]): Callback when recording starts (VAD detects voice starts). \
#             It will be called in another thread, so make sure it is thread-safe.
#         on_recording_stop (Callable[[None], None]): Callback when recording stops (VAD detects voice stops). \
#             It will be called in another thread, so make sure it is thread-safe.
#     """

#     def __init__(
#         self,
#         on_text: Callable[[str], None],
#         on_recording_start: Callable[[None], None],
#         on_recording_stop: Callable[[None], None],
#     ) -> None:
#         self.on_text = on_text
#         self.thread: Thread = None
#         self.thread_stop = Event()
#         self.recorder = AudioToTextRecorder(
#             model="base",
#             language="en",
#             spinner=False,
#             device="cpu",
#             use_microphone=False,
#             webrtc_sensitivity=3,
#             compute_type="float32",
#             enable_realtime_transcription=True,
#             realtime_model_type="tiny.en",
#             post_speech_silence_duration=1.2,
#             # on_realtime_transcription_stabilized=on_text,
#             on_recording_start=on_recording_start,
#             on_recording_stop=on_recording_stop,
#         )

#     def __start(self) -> None:
#         """Start an infinite loop of transcribing audio."""
#         while not self.thread_stop.is_set():
#             self.recorder.text(self.on_text)

#     def start(self) -> None:
#         """Start the recorder in another thread."""
#         self.thread = Thread(target=self.__start)
#         self.thread.start()

#     def stop(self) -> None:
#         """Stop the recorder."""
#         if (self.thread is not None) and self.thread.is_alive():
#             self.recorder.shutdown()
#             self.thread_stop.set()
#             self.thread.join()

#     def feed(self, audio: numpy.ndarray) -> None:
#         """Feed audio bytes to the recorder.

#         Args:
#             audio (bytes): audio bytes.
#         """
#         if (self.thread is None) or not self.thread.is_alive():
#             self.start()
#         # The recorder only accepts int16 numpy array.
#         self.recorder.feed_audio(audio)


class STTClientV2:

    @staticmethod
    def calculate_duration_frame(
        duration_time: float, chunk_time: int = 10
    ) -> int:
        """Calculate the nubmer of duration frames from the duration time and sample rate.

        Args:
            duration_time (float): Target duration time in second.
            chunk_size (int, optional): Chunk size of the audio. Defaults to 10.

        Returns:
            int: The number of duration frames.
        """
        return int(duration_time * 1000 / chunk_time)

    def __init__(
        self,
        on_text: Callable[[str], None],
        on_recording_start: Callable[[None], None],
        on_recording_stop: Callable[[None], None],
        duration_frames: int = calculate_duration_frame(1.2, 10),
        aggresiveness: int = 3,
    ) -> None:
        """Start a realtime speech-to-text service.
        This class uses WebrtcVAD to detect speech and Whisper to transcribe it.
        Multiple whisper models are loaded in different threads. Once a block of speech is recognized, it will be fed to the whisper pool.
        It also creates another thread for handling events (on_text, on_recording_start, on_recording_stop).
        Initialize and start the pool before using this class.

        Args:
            on_text (Callable[[str], None]): Callback when text is transcribed. \
                It will be called in another thread, so make sure it is thread-safe.
            on_recording_start (Callable[[None], None]): Callback when recording starts (VAD detects voice starts). \
                It will be called in another thread, so make sure it is thread-safe.
            on_recording_stop (Callable[[None], None]): Callback when recording stops (VAD detects voice stops). \
                It will be called in another thread, so make sure it is thread-safe.
            duration_frames (int): Number of empty frames allowed in a break of a speech. This value depends on your audio chunk size.
            aggresiveness (int, optional): Threshold of the vad. Defaults to 1.
        """

        self.on_text = on_text
        self.on_recording_start = on_recording_start
        self.on_recording_stop = on_recording_stop
        self.__event_queue = queue.Queue()
        self.__vad = WebrtcvadClient(
            duration_frames,
            aggresiveness,
            return_queue=self.__event_queue,
        )
        # self.__silero = SileroPool.get_instance()
        self.__whisper = WhisperPool.get_instance()
        self.__stop_flag = Event()
        self.__event_thread = Thread(target=self.__handle_event, daemon=True)
        self.__event_thread.start()

    def __handle_event(self) -> None:
        """Handle events from the vad and whisper pool."""

        while not self.__stop_flag.isSet():
            try:
                # Set a timeout to avoid blocking forever
                event = self.__event_queue.get(timeout=1)
            except queue.Empty:
                continue
            if isinstance(event, OnTextMessage):
                self.on_text(event.text)
            elif isinstance(event, OnSpeechMessage):
                self.on_recording_start()
            elif isinstance(event, OnSilenceMessage):
                self.on_recording_stop()
            elif isinstance(event, SileroIsSpeech):
                self.__whisper.add_work(
                    TranscriptionWork(event.audio, self.__event_queue)
                )
            else:
                logging.warning("Invalid STTClientV2 event received.")

    def feed(self, audio: bytes) -> None:
        """Feed audio bytse to the vad and try to transcribe it.

        Args:
            audio (bytes): 16-bit 16000Hz 10ms/20ms/30ms audio chunk.
        """

        speech = self.__vad.feed(audio)
        if speech is not None:
            try:
                self.__whisper.add_work(
                    TranscriptionWork(speech, self.__event_queue)
                )
            except RuntimeError as e:
                logging.warning(
                    f"Failed to add work to silero pool: {e}. \nProbably the pool isn't started yet."
                )

    def close(self) -> None:
        """Close the STT client. Stop the event thread and release the vad."""
        self.__stop_flag.set()
        # self.__event_thread.join()
        self.__vad = None
