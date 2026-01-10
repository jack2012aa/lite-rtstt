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
