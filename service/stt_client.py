"""An AudioToTextRecorder client."""

import logging

from threading import Thread, Event
from typing import Callable

import numpy

from RealtimeSTT import AudioToTextRecorder

__all__ = ["STTClient"]


class STTClient:
    """An AudioToRecorder client.

    Args:
        on_text (Callable[[str], None]): Callback when text is transcribed. \
            It will be called in another thread, so make sure it is thread-safe.
        on_recording_start (Callable[[None], None]): Callback when recording starts (VAD detects voice starts). \
            It will be called in another thread, so make sure it is thread-safe.
        on_recording_stop (Callable[[None], None]): Callback when recording stops (VAD detects voice stops). \
            It will be called in another thread, so make sure it is thread-safe.
    """

    def __init__(
        self,
        on_text: Callable[[str], None],
        on_recording_start: Callable[[None], None],
        on_recording_stop: Callable[[None], None],
    ) -> None:
        self.on_text = on_text
        self.thread: Thread = None
        self.thread_stop = Event()
        self.recorder = AudioToTextRecorder(
            model="base",
            language="en",
            spinner=False,
            device="cpu",
            use_microphone=False,
            webrtc_sensitivity=3,
            level=logging.DEBUG,
            compute_type="float32",
            enable_realtime_transcription=True,
            realtime_model_type="tiny.en",
            post_speech_silence_duration=1.0,
            on_recording_start=on_recording_start,
            on_recording_stop=on_recording_stop,
        )

    def __start(self) -> None:
        """Start an infinite loop of transcribing audio."""
        while not self.thread_stop.is_set():
            self.recorder.text(self.on_text)

    def start(self) -> None:
        """Start the recorder in another thread."""
        self.thread = Thread(target=self.__start)
        self.thread.start()

    def stop(self) -> None:
        """Stop the recorder."""
        if (self.thread is not None) and self.thread.is_alive():
            self.recorder.shutdown()
            self.thread_stop.set()
            self.thread.join()

    def feed(self, audio: numpy.ndarray) -> None:
        """Feed audio bytes to the recorder.

        Args:
            audio (bytes): audio bytes.
        """
        if (self.thread is None) or not self.thread.is_alive():
            self.start()
        # The recorder only accepts int16 numpy array.
        self.recorder.feed_audio(audio)
