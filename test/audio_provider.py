from lite_rtstt.stt.audiobuffer import AudioBuffer


def from_int16_pcm(path: str, chunk_ms: int = 30) -> AudioBuffer:
    num_bytes = chunk_ms * 16 * 2
    buffer = AudioBuffer()
    with open(path, 'rb') as f:
        bytes = f.read()
    for i in range(0, len(bytes), num_bytes):
        buffer.append(bytes[i:i + num_bytes])
    return buffer

def get_silence_audio(duration_ms: int) -> AudioBuffer:
    num_bytes = duration_ms * 16 * 2
    bytes = b"\x00" * num_bytes
    return AudioBuffer.from_bytes(bytes)