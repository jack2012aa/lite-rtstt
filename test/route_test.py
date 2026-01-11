import os
import shutil
import unittest
from dataclasses import replace
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lite_rtstt.stt.config import STTConfig
from lite_rtstt.stt.rtstt_client import ThreeLayerRTSTTClient
from lite_rtstt.stt.stt_client import WhisperClient
from lite_rtstt.stt.vad_client import WebRTCClient, SileroClient
from lite_rtstt.network.route import create_router
from test.utils import assert_text_similar


class WebSocketIntegrationTest(unittest.TestCase):

    def setUp(self):
        self.__model_dir = "test_temp_ws_model"
        if not os.path.exists(self.__model_dir):
            os.makedirs(self.__model_dir)

        self.__config = replace(STTConfig.default(), **{
            "vad_threads": 1
        })

        self.__rtc = WebRTCClient(self.__config)
        self.__silero = SileroClient(self.__config)
        self.__whisper = WhisperClient(self.__config, self.__model_dir)

        self.__rtstt = ThreeLayerRTSTTClient(
            self.__config, self.__rtc, self.__silero, self.__whisper
        )
        self.__rtstt.start()

        router = create_router(self.__rtstt)
        app = FastAPI()
        app.include_router(router)

        self.client = TestClient(app)

    def tearDown(self):
        self.__rtstt.close()
        if os.path.exists(self.__model_dir):
            shutil.rmtree(self.__model_dir)

    def test_websocket_transcription_flow(self):
        pcm_path = "test/data/7s_i16.pcm"
        if not os.path.exists(pcm_path):
            self.skipTest("PCM data file not found")

        with open(pcm_path, "rb") as f:
            audio_data = f.read()

        chunk_size = int(16000 * 0.03 * 2)  # 960 bytes
        with self.client.websocket_connect("/rtstt") as websocket:
            total_chunks = len(audio_data) // chunk_size
            for i in range(total_chunks):
                start = i * chunk_size
                end = start + chunk_size
                chunk = audio_data[start:end]
                websocket.send_bytes(chunk)

            silence_chunks = int(1500 / 30)
            silence = b'\x00' * chunk_size

            for _ in range(silence_chunks):
                websocket.send_bytes(silence)

            received_start = False
            received_stop = False
            received_text = None

            try:
                while not received_text:
                    data = websocket.receive_json()
                    msg_type = data.get("type")
                    if msg_type == "start speaking":
                        received_start = True
                    elif msg_type == "stop speaking":
                        received_stop = True
                    elif msg_type == "text":
                        received_text = data.get("text")
                        break

            except Exception as e:
                self.fail(f"WebSocket communication failed or timed out: {e}")

            self.assertTrue(received_start, "Did not receive 'start speaking' event")
            self.assertTrue(received_stop, "Did not receive 'stop speaking' event")
            self.assertIsNotNone(received_text, "Did not receive transcript")

            expected_text = "You are given an integer matrix grid and an array queries of size k."
            assert_text_similar(self, expected_text, received_text, threshold=0.7)


if __name__ == "__main__":
    unittest.main()