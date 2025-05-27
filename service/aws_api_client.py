"""An STT client for AWS Transcribe."""

import asyncio
from typing import Callable

from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent


class EventHandler(TranscriptResultStreamHandler):

    def __init__(
        self,
        transcript_result_stream,
        on_text: Callable[[str], None],
        on_recording_start: Callable[[], None],
        on_recording_stop: Callable[[], None],
        delay: float = 1.0,
    ):
        """A handler that processes AWS Transcribe streaming results.

        Args:
            transcript_result_stream (_type_): A stream of transcript results.
            on_text (Callable[[str], None]): A callback function that is called with the transcript text.
            on_recording_start (Callable[[], None]): A callback function that is called when recording starts.
            on_recording_stop (Callable[[], None]): A callback function that is called when recording stops.
            delay (float, optional): Delay before the on_text is called when there is a transcript. Defaults to 1.0.
        """

        super().__init__(transcript_result_stream)
        self.__on_text = on_text
        self.__on_recording_start = on_recording_start
        self.__on_recording_stop = on_recording_stop
        self.__is_recording = False
        """Flag to indicate if recording is in progress."""
        self.__buffer = []
        """Buffer to hold transcripts."""
        self.__delay = delay
        """Delay in seconds before calling on_text."""
        self.__timer_task: asyncio.Task = None
        """A task that calls the on_text callback after a delay."""

    async def __on_text_delayed(self):
        """Call the on_text callback after a delay."""

        await asyncio.sleep(self.__delay)
        if len(self.__buffer) != 0:
            text = " ".join(self.__buffer)
            self.__on_text(text)
            self.__buffer = []

    async def __update_timer(self):
        """Cancel the previous timer and start a new one."""

        if self.__timer_task:
            self.__timer_task.cancel()
        self.__timer_task = asyncio.create_task(self.__on_text_delayed())

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):

        # Process the reuslt if it is not empty.
        results = transcript_event.transcript.results
        if len(results) == 0:
            return
        
        # The user is still talking, send the transcript later.
        await self.__update_timer()

        if not self.__is_recording:
            self.__on_recording_start()
            self.__is_recording = True

        if not results[0].is_partial:
            self.__buffer.append(results[0].alternatives[0].transcript)
            print(f"Transcript appended: {self.__buffer}")
            self.__on_recording_stop()
            self.__is_recording = False

    async def close(self):
        """Close the handler and cancel any pending tasks."""
        if self.__timer_task:
            self.__timer_task.cancel()


class AWSApiClient:

    def __init__(
        self,
        on_text: Callable[[str], None],
        on_recording_start: Callable[[], None],
        on_recording_stop: Callable[[], None],
        delay: float = 1.0,
    ):
        """A client for AWS Transcribe streaming service.

        Args:
            on_text (Callable[[str], None]): A callback function that is called with the transcript text.
            on_recording_start (Callable[[], None]): A callback function that is called when recording starts.
            on_recording_stop (Callable[[], None]): A callback function that is called when recording stops.
            delay (float, optional): Delay before the on_text is called when there is a transcript. Defaults to 1.0.
        """

        # Because the connection is asynchronous, build the connection in another function.
        self.__on_text = on_text
        self.__on_recording_start = on_recording_start
        self.__on_recording_stop = on_recording_stop
        self.__delay = delay
        """Delay in seconds before calling on_text."""
        self.__stream = None
        """The connection stream to AWS Transcribe."""
        self.__handler: EventHandler = None
        """The event handler for processing the transcript results."""
        self.__buffer = bytearray()
        """Buffer to hold audio data before sending it to the stream."""
        self.__handler_task = None
        """Task for handling events from the stream."""

    async def run(self):
        """Run the AWS Transcribe streaming client."""

        # Build connection.
        client = TranscribeStreamingClient(region="us-west-2")
        self.__stream = await client.start_stream_transcription(
            language_code="en-US",
            media_sample_rate_hz=16000,
            media_encoding="pcm",
            enable_partial_results_stabilization=True,
        )
        self.__handler = EventHandler(
            self.__stream.output_stream,
            self.__on_text,
            self.__on_recording_start,
            self.__on_recording_stop,
            self.__delay,
        )
        self.__handler_task = asyncio.create_task(self.__handler.handle_events())

    async def feed(self, audio_chunk: bytes):
        if self.__stream is None:
            raise RuntimeError("Stream not initialized or closed. Call run() first.")

        self.__buffer.extend(audio_chunk)
        THRESHOLD = 1024 * 8

        while len(self.__buffer) >= THRESHOLD:
            chunk_to_send = self.__buffer[:THRESHOLD]
            del self.__buffer[:THRESHOLD]
            await self.__stream.input_stream.send_audio_event(audio_chunk=chunk_to_send)

    async def close(self):
        if self.__stream is None:
            return
        await self.__stream.input_stream.end_stream()
        if self.__handler_task:
            self.__handler_task.cancel()
        if self.__handler:
            await self.__handler.close()