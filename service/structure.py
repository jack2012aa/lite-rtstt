"""Defines WebSocket message types."""

from dataclasses import dataclass
import queue
import uuid

import numpy as np

__all__ = [
    "SpeechTranscriptMessage",
    "StartSpeakingMessage",
    "StopSpeakingMessage",
    "TranscriptionWork",
    "ProcessTranscriptionWork",
    "OnSpeechMessage",
    "OnSilenceMessage",
    "OnTextMessage",
    "OnTextMessageWithId",
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
    """A work passed to the whisper worker (thread)."""

    audio: np.ndarray
    return_queue: queue.Queue


@dataclass
class ProcessTranscriptionWork:
    """A work passed to the whisper worker (process)."""

    audio: np.ndarray
    id: uuid.UUID


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
class OnTextMessageWithId:
    """On text message with ID."""

    text: str
    id: uuid.UUID


@dataclass
class SileroIsSpeech:
    """Silero thinks the audio is speech."""

    audio: np.ndarray
