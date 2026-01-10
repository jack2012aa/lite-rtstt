"""An AudioToTextRecorder client."""
import asyncio
import logging
from abc import ABC, abstractmethod
from enum import Enum

from lite_rtstt.stt.audiobuffer import AudioBuffer
from lite_rtstt.stt.config import STTConfig
from lite_rtstt.stt.event import STTEventQueue, SimpleSTTEventQueue, EventFactory
from lite_rtstt.stt.stt_client import STTClient
from lite_rtstt.stt.vad_client import VADClient


class RTSTTClient(ABC):
    """A real-time speech to text service."""

    @abstractmethod
    def start(self):
        """Start the service."""
        pass

    @abstractmethod
    def connect(self) -> tuple[STTEventQueue, int]:
        """Connect to the stt service.

        Returns:
            tuple[STTEventQueue, int]: An event queue and the connection id.
        """
        pass

    @abstractmethod
    def disconnect(self, connection_id: int) -> None:
        """Disconnect from the stt service."""

    @abstractmethod
    async def feed(self, connection_id: int, audio: bytes):
        """Feed the audio chunk to the stt service.

        Args:
            connection_id (int): The connection id.
            audio (numpy.ndarray): The audio data.
        """
        pass


class ThreeLayerRTSTTClient(RTSTTClient):


    class AudioStreamStateMachine:
        """A state machine that keeps track of the state of the audio stream."""

        class State(Enum):
            SILENCE = 0
            ACTIVE = 1
            SPEAKING = 2

        def __init__(
            self,
            first_vad_client: VADClient,
            second_vad_client: VADClient,
            stt_client: STTClient,
            max_silence_chunks: int,
            min_active_to_detection_chunks: int,
            max_buffer_chunks: int,
        ) -> None:
            self.__audio_buffer = AudioBuffer()
            self.__state = self.State.SILENCE
            self.__silence_chunks = 0
            self.__first_vad_client = first_vad_client
            self.__second_vad_client = second_vad_client
            self.__stt_client = stt_client
            self.__max_silence_chunks = max_silence_chunks
            self.__min_active_to_detection_chunks = min_active_to_detection_chunks
            self.__max_buffered_chunks = max_buffer_chunks

        async def __feed_from_silence(self, new_buffer: AudioBuffer):
            is_active = await self.__first_vad_client.is_active(new_buffer)
            if is_active:
                self.__state = self.State.ACTIVE

        async def __feed_from_active(self):
            if self.__audio_buffer.get_chunks_count() >= self.__min_active_to_detection_chunks:
                is_speaking = await self.__second_vad_client.is_active(self.__audio_buffer)
                if is_speaking:
                    self.__state = self.State.SPEAKING
                else:
                    self.__state = self.State.SILENCE
                    self.__audio_buffer = AudioBuffer()

        async def __feed_from_speaking(self) -> asyncio.Task[str] | None:
            is_speaking = await self.__second_vad_client.is_active(self.__audio_buffer)
            if not is_speaking:
                self.__silence_chunks += 1
                if self.__silence_chunks >= self.__max_silence_chunks:
                    audio_buffer = self.__audio_buffer
                    self.__audio_buffer = AudioBuffer()
                    self.__state = self.State.SILENCE
                    self.__silence_chunks = 0
                    task = asyncio.create_task(self.__stt_client.transcribe(audio_buffer))
                    return task
            elif self.__audio_buffer.get_chunks_count() >= self.__max_buffered_chunks:
                audio_buffer = self.__audio_buffer
                self.__audio_buffer = AudioBuffer()
                self.__state = self.State.SILENCE
                self.__silence_chunks = 0
                task = asyncio.create_task(self.__stt_client.transcribe(audio_buffer))
                return task
            return None

        async def feed(self, audio: bytes) -> tuple['ThreeLayerRTSTTClient.AudioStreamStateMachine.State', 'ThreeLayerRTSTTClient.AudioStreamStateMachine.State', asyncio.Task[str] | None]:
            """Return (old state, new state, transcription task)"""
            current_state = self.__state
            new_buffer = AudioBuffer.from_bytes(audio)
            self.__audio_buffer.append(audio)
            task = None
            if self.__state == self.State.SILENCE:
                await self.__feed_from_silence(new_buffer)
            elif self.__state == self.State.ACTIVE:
                await self.__feed_from_active()
            elif self.__state == self.State.SPEAKING:
                task = await self.__feed_from_speaking()
            else:
                raise RuntimeError(f"Undefined audio stream state {self.__state}")
            return current_state, self.__state, task

    def __init__(
        self,
        config: STTConfig,
        first_vad_client: VADClient,
        second_vad_client: VADClient,
        stt_client: STTClient,
    ) -> None:
        """A RTSTTClient that use two layers of VAD and one layer of STT.
        The first layer VAD can be more light-weighted, to filter background noice.
        However, it should be able to signal voice in small chunk in order to cut the stream precisely.
        The second layer VAD can be more powerful to save STT's energy.
        """

        self.__first_vad_client = first_vad_client
        self.__second_vad_client = second_vad_client
        self.__stt_client = stt_client
        self.__state_machines: dict[int, "ThreeLayerRTSTTClient.AudioStreamStateMachine"] = {}
        self.__queues: dict[int, SimpleSTTEventQueue] = {}
        self.__increasing_id = 0
        self.__max_silence_chunks = int(config.duration_time_ms / config.chunk_size_ms)
        self.__min_active_to_detection_chunks = int(config.active_to_detection_ms / config.chunk_size_ms)
        self.__max_buffered_chunks = config.max_buffered_chunks

    def start(self):
        self.__first_vad_client.start()
        self.__second_vad_client.start()
        self.__stt_client.start()

    def connect(self) -> tuple[STTEventQueue, int]:
        connection_id = self.__increasing_id
        self.__increasing_id += 1
        self.__state_machines[connection_id] = self.AudioStreamStateMachine(
            self.__first_vad_client,
            self.__second_vad_client,
            self.__stt_client,
            self.__max_silence_chunks,
            self.__min_active_to_detection_chunks,
            self.__max_buffered_chunks
        )
        self.__queues[connection_id] = SimpleSTTEventQueue()
        return self.__queues[connection_id], connection_id

    def disconnect(self, connection_id: int) -> None:
        self.__state_machines.pop(connection_id, None)
        self.__queues.pop(connection_id, None)

    async def feed(self, connection_id: int, audio: bytes):
        state_machine = self.__state_machines[connection_id]
        old_state, new_state, task = await state_machine.feed(audio)
        if old_state == self.AudioStreamStateMachine.State.ACTIVE and new_state == self.AudioStreamStateMachine.State.SPEAKING:
            await self.__queues[connection_id].put(EventFactory.start_speaking_event())
        elif old_state == self.AudioStreamStateMachine.State.SPEAKING and new_state == self.AudioStreamStateMachine.State.SILENCE:
            await self.__queues[connection_id].put(EventFactory.stop_speaking_event())
            text = await task
            await self.__queues[connection_id].put(EventFactory.text_event(text))

    def close(self):
        self.__first_vad_client.close()
        self.__second_vad_client.close()
        self.__stt_client.close()