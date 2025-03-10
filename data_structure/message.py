"""Defines WebSocket message types."""

from dataclasses import dataclass

__all__ = ["SpeechTranscriptMessage", "TranscriptionStartMessage"]


@dataclass
class SpeechTranscriptMessage:
    """Speech transcript message."""

    title: str = "speech transcript"
    text: str = ""


@dataclass
class TranscriptionStartMessage:
    """Transcription start message."""

    title: str = "transcription start"
