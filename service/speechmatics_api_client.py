# A client using the SpeechMatics API.

import os
import asyncio
import logging
from typing import Callable

from httpx import HTTPStatusError
from speechmatics.models import (
    ConnectionSettings,
    TranscriptionConfig,
    AudioSettings,
    ServerMessageType,
)
from speechmatics.client import WebsocketClient

__all__ = ["SpeechmaticsClient"]


class AudioProcessor:
    def __init__(self):
        """A class that SpeechMatics uses to access audio data."""
        self.wave_data = bytearray()  # Buffer
        self.read_offset = 0  # Offset for reading data

    async def read(self, chunk_size):
        while self.read_offset + chunk_size > len(self.wave_data):
            await asyncio.sleep(0)
        new_offset = self.read_offset + chunk_size
        data = self.wave_data[self.read_offset : new_offset]
        self.read_offset = new_offset
        return data

    def write_audio(self, data):
        self.wave_data.extend(data)
        return


class SpeechmaticsClient:

    def __init__(
        self,
        on_text: Callable[[str], None],
        on_recording_start: Callable[[None], None],
        on_recording_stop: Callable[[None], None],
        delay: float = 1.5,
    ):
        """A client that wraps and connects to the SpeechMatics API.

        Args:
            on_text (Callable[[str], None]): Callback when text is transcribed.
            on_recording_start (Callable[[None], None]): Callback when recording starts (VAD detects voice starts).
            on_recording_stop (Callable[[None], None]): Callback when recording stops (VAD detects voice stops).
            delay (float): Delay in seconds for the transcription of each word.
        """

        self.__stream = AudioProcessor()

        # Create a websocket connection.
        URL = "wss://eu2.rt.speechmatics.com/v2/en"
        API_KEY = os.getenv("SPEECHMATICS_API_KEY")
        connection_settings = ConnectionSettings(url=URL, auth_token=API_KEY)
        connection = WebsocketClient(connection_settings)

        self.__words = []
        """A buffer for final words."""
        self.__empty_final = 0
        """Number of empty final messages received."""
        self.__empty_threshold = 2
        """Number of empty final messages before calling on_text."""
        def transcript_handler(message: dict):
            """Buffer words and call on_text once the message terns to be empty.

            Args:
                message (dict): a message from the SpeechMatics API.
            """

            # See here for the message format: https://docs.speechmatics.com/rt-api-ref#addtranscript
            if message["metadata"]["transcript"] != "":
                self.__words.append(message["metadata"]["transcript"])
            elif len(self.__words) > 0:
                if '.' not in self.__words[-1]:
                    self.__words.append('.')
                on_text("".join(self.__words))
                self.__words = []
                on_recording_stop()
                self.__empty_final = 0

        def recording_start_handler(message: dict):
            if message["metadata"]["transcript"] != "" and len(self.__words) == 0:
                on_recording_start()

        connection.add_event_handler(
            event_handler=transcript_handler, event_name=ServerMessageType.AddTranscript
        )
        connection.add_event_handler(
            event_handler=recording_start_handler, 
            event_name=ServerMessageType.AddPartialTranscript
        )
        transcription_config = TranscriptionConfig(
            language="en", enable_partials=True, max_delay=delay, max_delay_mode="flexible"
        )
        audio_settings = AudioSettings(encoding="pcm_s16le", sample_rate=16000)

        async def task_logging_wrapper():
            """A wrapper that logs error when connecting to the SpeechMatics API fails."""
            try:
                await connection.run(
                    self.__stream, transcription_config, audio_settings
                )
            except HTTPStatusError as e:
                logging.error(f"Failed to connect to SpeechMatics API: {e}")
            except Exception as e:
                logging.error(
                    f"An unexpected error occurred when connecting to SpeechMatics API: {e}"
                )

        self.task = asyncio.create_task(task_logging_wrapper())

    def feed(self, audio: bytes) -> None:
        """Feed audio bytes to the AudioProcessor.

        Args:
            audio (bytes): 16-bit 16kHz PCM audio bytes.
        Raises:
            RuntimeError: If the SpeechmaticsClient is closed and audio is fed.
        """

        if not self.task.done():
            self.__stream.write_audio(audio)
        else:
            raise RuntimeError("SpeechmaticsClient is closed, cannot feed audio.")

    def close(self):

        self.task.cancel()