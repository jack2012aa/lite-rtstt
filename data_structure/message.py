"""Defines WebSocket message types."""

from dataclasses import dataclass

__all__ = [
    "SpeechTranscriptMessage",
    "StartSpeakingMessage",
    "StopSpeakingMessage",
]


@dataclass
class SpeechTranscriptMessage:
    """Speech transcript message."""

    title: str = "speech transcript"
    transcript: str = ""


@dataclass
class StartSpeakingMessage:
    """Start speaking message."""

    title: str = "start speaking"


@dataclass
class StopSpeakingMessage:
    """Stop speaking message."""

    title: str = "stop speaking"
