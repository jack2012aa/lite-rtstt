"""An AudioToTextRecorder client."""
from abc import ABC, abstractmethod
from enum import Enum

import numpy
from atomicx.atomicx import AtomicBool

from lite_rtstt.stt.audiobuffer import AudioBuffer
from lite_rtstt.stt.config import STTConfig
from lite_rtstt.stt.event import STTEventQueue, SimpleSTTEventQueue, StartSpeakingEvent, EventFactory
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
    def feed(self, connection_id: int, audio: bytes):
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
            min_active_to_detection_chunks: int
        ) -> None:
            self.__audio_buffer = AudioBuffer()
            self.__state = self.State.SILENCE
            self.__silence_chunks = 0
            self.__max_silence_chunks = max_silence_chunks
            self.__min_active_to_detection_chunks = min_active_to_detection_chunks

        async def


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
        self.__closed = AtomicBool(False)
        self.__buffers = {}
        self.__queues = {}
        self.__is_speaking = {}
        self.__silence_chunks = {}
        self.__increasing_id = 0
        self.__max_silence_chunks = int(config.duration_time_ms / config.chunk_size_ms)
        self.__min_active_to_detection_chunks = int(config.active_to_detection_ms / config.chunk_size_ms)

    def start(self):
        self.__first_vad_client.start()
        self.__second_vad_client.start()
        self.__stt_client.start()

    def connect(self) -> tuple[STTEventQueue, int]:
        connection_id = self.__increasing_id
        self.__increasing_id += 1
        self.__buffers[connection_id] = AudioBuffer()
        self.__queues[connection_id] = SimpleSTTEventQueue()
        self.__is_speaking[connection_id] = False
        self.__silence_chunks[connection_id] = 0
        return self.__queues[connection_id], connection_id

    async def feed(self, connection_id: int, audio: bytes):
        new_buffer = AudioBuffer.from_bytes(audio)
        audio_buffer = self.__buffers[connection_id]
        audio_buffer.append(new_buffer)
        is_active = await self.__first_vad_client.is_active(new_buffer)
        if is_active:
            if audio_buffer.get_chunks_count() >= self.__min_active_to_detection_chunks:
                is_active = await self.__second_vad_client.is_active(self.__buffers[connection_id])
            else:
                return

        if is_active:
            if not self.__is_speaking[connection_id]:
                self.__is_speaking[connection_id] = True
                self.__queues[connection_id].put(EventFactory.get_start_speaking_event())
                return

        self.__silence_chunks[connection_id] += 1
        if self.__silence_chunks[connection_id] >= self.__max_silence_chunks:
            audio_buffer = self.__buffers.pop(connection_id)
            self.__is_speaking[connection_id] = False
            self.__silence_chunks[connection_id] = 0
            self.__queues[connection_id].put(EventFactory.get_stop_speaking_event())
            text = await self.__stt_client.transcribe(audio_buffer)
            self.__queues[connection_id].put(EventFactory.get_text_event(text))

    def close(self):
        self.__first_vad_client.close()
        self.__second_vad_client.close()
        self.__stt_client.close()