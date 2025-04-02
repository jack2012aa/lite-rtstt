"""Defines WebSocket message types."""

from dataclasses import dataclass
import queue

import numpy as np

__all__ = [
    "SpeechTranscriptMessage",
    "StartSpeakingMessage",
    "StopSpeakingMessage",
    "TranscriptionWork",
    "OnSpeechMessage",
    "OnSilenceMessage",
    "OnTextMessage",
    "SileroIsSpeech",
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


@dataclass
class TranscriptionWork:
    """A work passed to the whisper worker."""

    audio: np.ndarray
    return_queue: queue.Queue


@dataclass
class OnSpeechMessage:
    """On speech message."""


@dataclass
class OnSilenceMessage:
    """On silence message."""


@dataclass
class OnTextMessage:
    """On text message."""

    text: str


@dataclass
class SileroIsSpeech:
    """Silero thinks the audio is speech."""

    audio: np.ndarray
