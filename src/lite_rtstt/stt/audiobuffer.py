import numpy


class AudioBuffer:

    @staticmethod
    def from_bytes(buffer: bytes) -> 'AudioBuffer':
        audio_buffer = AudioBuffer()
        audio_buffer.append(buffer)
        return audio_buffer

    def __init__(self):
        """An audio buffer from 16-bit 16000Hz audio chunk."""
        self.__buffer = []

    def append(self, buffer: bytes):
        self.__buffer.append(buffer)

    def get_chunks_count(self) -> int:
        return len(self.__buffer)

    def to_bytes(self) -> bytes:
        return b"".join(self.__buffer)

    def to_ndarray(self) -> numpy.ndarray:
        return numpy.frombuffer(b"".join(self.__buffer), dtype=numpy.int16)