from dataclasses import dataclass

@dataclass(frozen=True)
class STTConfig:
    vad_threads: int
    whisper_model: str
    duration_time_ms: int
    aggresiveness: int
    sample_rate: int
    chunk_size_ms: int
    active_to_detection_ms: int
    max_buffered_chunks: int

    @staticmethod
    def default() -> "STTConfig":
        return STTConfig(
            vad_threads=4,
            whisper_model="base",
            duration_time_ms=1200,
            aggresiveness=3,
            sample_rate=16000,
            chunk_size_ms=30,
            active_to_detection_ms=900,
            max_buffered_chunks=500,
        )
