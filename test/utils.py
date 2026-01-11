import re
import string
from difflib import SequenceMatcher

from lite_rtstt.stt.audio_buffer import AudioBuffer


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


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def assert_text_similar(test_case, expected, actual, threshold=0.8):
    norm_expected = normalize_text(expected)
    norm_actual = normalize_text(actual)
    ratio = SequenceMatcher(None, norm_expected, norm_actual).ratio()

    test_case.assertGreater(
        ratio,
        threshold,
        f"Text similarity {ratio:.2f} is below threshold {threshold}.\nExpected: {norm_expected}\nActual: {norm_actual}"
    )
