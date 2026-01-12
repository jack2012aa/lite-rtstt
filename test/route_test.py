import asyncio
import base64
import os
import shutil
import threading
import unittest
from dataclasses import replace
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

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
        # if os.path.exists(self.__model_dir):
        #     shutil.rmtree(self.__model_dir)

    def test_websocket_transcription_flow(self):
        pcm_path = "test/data/42s_i16.pcm"
        if not os.path.exists(pcm_path):
            self.skipTest("PCM data file not found")

        with open(pcm_path, "rb") as f:
            audio_data = f.read()

        chunk_size = int(16000 * 0.03 * 2)  # 960 bytes
        with self.client.websocket_connect("/rtstt") as websocket:

            messages = [
                {"type": "start speaking"},
                {"type": "stop speaking"},
                {"type": "text", "text": ' After giving an integer matrix grid and an array queries of size k, find an array answer of size k such sets for each integer queries i, you start in the top left cell of the'},
                {"type": "start speaking"},
                {"type": "stop speaking"},
                {"type": "text",
                 "text": ' Tricks and repeats the following process. If queries i is strictly greater than the value of the current cell that you are in, then you get one point.'},
                {"type": "start speaking"},
                {"type": "stop speaking"},
                {"type": "text",
                 "text": ' If it is first time visiting this cell, and you can move to any adjacent cell in all four directions. Otherwise, you do not get any points, and you end this process.'},
            ]
            exception = []
            def handle_message():
                try:
                    for expected in messages:
                        actual = websocket.receive_json()
                        self.assertEqual(expected["type"], actual["type"])
                        if expected["type"] == "text":
                            assert_text_similar(self, expected["text"], actual["text"])
                    try:
                        extra_msg = websocket.receive_json()
                        raise AssertionError(f"Expected WebSocket to close, but received: {extra_msg}")
                    except WebSocketDisconnect:
                        pass
                except Exception as e:
                    exception.append(e)

            thread = threading.Thread(target=handle_message)
            thread.start()

            total_chunks = len(audio_data) // chunk_size
            for i in range(total_chunks):
                start = i * chunk_size
                end = start + chunk_size
                chunk = audio_data[start:end]
                b64 = base64.b64encode(chunk).decode("utf-8")
                websocket.send_json({"type": "audio chunk", "data": b64})

            silence_chunks = int(1500 / 30)
            silence = b'\x00' * chunk_size
            b64 = base64.b64encode(silence).decode("utf-8")

            for _ in range(silence_chunks):
                websocket.send_json({"type": "audio chunk", "data": b64})

            websocket.send_json({"type": "EOF"})
            thread.join()
            if len(exception) > 0:
                raise exception[0]


if __name__ == "__main__":
    unittest.main()