__all__ = ["WebrtcvadClient"]

import queue

import webrtcvad
import numpy as np

from service.structure import OnSpeechMessage, OnSilenceMessage

class WebrtcvadClient:

    def __init__(
        self,
        duration_frames: int,
        aggresiveness=3,
        return_type: type[np.int16] | type[np.float32] = np.float32,
        return_queue: queue.Queue = None,
    ) -> None:
        """Create a webrtcvad client. It buffers the audio with speech.
        Args:
            duration_frames (int): Number of empty frames allowed in a break of a speech. This value depends on your audio chunk size.
            aggresiveness (int, optional): Threshold of the vad. Defaults to 3.
            return_type (np.dtype, optional): Type of the returned audio. Defaults to np.float32.
            return_queue (queue.Queue, optional): Queue to return the event. Defaults to None.
        """

        self.vad = webrtcvad.Vad(aggresiveness)
        self.duration_frames = duration_frames
        self.return_type = return_type
        self.__buffer = []
        self.__is_speech_start = False
        self.__speech_frames = 0
        self.__total_frames = 0
        self.__silence_counter = 0
        self.__return_queue = return_queue
        self.__CHECK_ON_SPEECH = 30 # 30 chunks = 300ms
        self.__THRESHOLD = 25 # 20 chunks = 200ms
    
    def feed(self, audio: bytes) -> None | np.ndarray:
        """Consume audio, accumulate speech, and return the audio buffer after the speech ends.

        Args:
            audio (bytes): An 16-bit 16000Hz 10ms/20ms/30ms audio chunk.

        Returns:
            None | np.ndarray: An ndarray of specific type if the speech ends, None otherwise.
        """

        is_speech = self.vad.is_speech(audio, 16000)

        if not self.__is_speech_start and is_speech:
            self.__is_speech_start = True
            self.__buffer = [audio]
            self.__speech_frames = 1
            self.__total_frames = 1
            self.__silence_counter = 0
        elif self.__is_speech_start and is_speech:
            self.__buffer.append(audio)
            self.__speech_frames += 1
            self.__total_frames += 1
            self.__silence_counter = 0
        elif self.__is_speech_start and not is_speech:
            self.__buffer.append(audio)
            self.__total_frames += 1
            self.__silence_counter += 1

            if self.__silence_counter >= self.duration_frames:
                self.__is_speech_start = False
                speech = np.frombuffer(b"".join(self.__buffer), dtype=np.int16)
                if self.return_type == np.float32:
                    speech = speech.astype(np.float32) / 32768.0
                self.__buffer = []
                self.__silence_counter = 0
                self.__return_queue.put(OnSilenceMessage())
                return speech
            
        if self.__total_frames == self.__CHECK_ON_SPEECH:
            if self.__speech_frames >= self.__THRESHOLD:
                if self.__return_queue is not None:
                    self.__return_queue.put(OnSpeechMessage())
                return None
            else:
                self.__is_speech_start = False
                return None

        return None
